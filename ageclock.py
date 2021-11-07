#!/usr/bin/env python

import math
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

MODEL_PATH = Path.cwd()/'models'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Eve(optim.Optimizer):
    """
    Implements Eve Algorithm, proposed in `IMPROVING STOCHASTIC GRADIENT DESCENT WITH FEEDBACK`
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.999), eps=1e-8,
                 k=0.1, K=10, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        k=k, K=K, weight_decay=weight_decay)
        super(Eve, self).__init__(params, defaults)

    def step(self, closure):
        """
        :param closure: closure returns loss. see http://pytorch.org/docs/optim.html#optimizer-step-closure
        :return: loss
        """
        loss = closure()
        _loss = loss.item()  # float

        for group in self.param_groups:

            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m_t'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['v_t'] = grad.new().resize_as_(grad).zero_()
                    # f hats, smoothly tracked objective functions
                    # \hat{f}_0 = f_0
                    state['ft_2'], state['ft_1'] = _loss, None
                    state['d'] = 1

                m_t, v_t = state['m_t'], state['v_t']
                beta1, beta2, beta3 = group['betas']
                k, K = group['k'], group['K']
                d = state['d']
                state['step'] += 1
                t = state['step']
                # initialization of \hat{f}_1
                if t == 1:
                    # \hat{f}_1 = f_1
                    state['ft_1'] = _loss
                # \hat{f_{t-1}}, \hat{f_{t-2}}
                ft_1, ft_2 = state['ft_1'], state['ft_2']
                # f(\theta_{t-1})
                f = _loss

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                m_t.mul_(beta1).add_(grad, alpha=1-beta1)
                v_t.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                m_t_hat = m_t / (1 - beta1 ** t)
                v_t_hat = v_t / (1 - beta2 ** t)

                if t > 1:
                    if f >= state['ft_2']:
                        delta = k + 1
                        Delta = K + 1
                    else:
                        delta = 1 / (K + 1)
                        Delta = 1 / (k + 1)

                    c = min(max(delta, f / ft_2), Delta)
                    r = abs(c - 1) / min(c, 1)
                    state['ft_1'], state['ft_2'] = c * ft_2, ft_1
                    state['d'] = beta3 * d + (1 - beta3) * r

                # update parameters
                p.data.addcdiv_(m_t_hat,
                                v_t_hat.sqrt().add_(group['eps']),
                                value=-group['lr']/state['d'])

        return loss

class CustomDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.drop(labels='AGE', axis=1).iloc[idx]
        x = torch.from_numpy(x.values)
        #returning y as a scalar might be a problem
        y = self.df.iloc[idx].AGE
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

class NN(nn.Module):
    def __init__(self, layers=[1000,500,250], ps=0.35, in_features=20, y_range=(20, 90)):
        super(NN, self).__init__()
        self.y_range = y_range
        self.layers = layers
        self.ps = ps
        layers = [in_features] + layers
        layers = list(zip(layers, layers[1:]))
        
        l = []
        for layer in layers:
            l.append(nn.Linear(*layer))
            #TODO: play with negative slope koef. of LeakyReLU
            l.append(nn.LeakyReLU())
            l.append(nn.Dropout(ps))
        l.append(nn.Linear(layers[-1][1], 1))

        self.arch = nn.Sequential(*l)
        
    def forward(self, x):
        x = self.arch(x)
        x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x
    
    def __repr__(self):
        return "Linear -> LeakyReLU -> Dropout\nlayers: {}\nps: {}\n".format(self.layers, self.ps)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, in_5_range = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).float().unsqueeze(dim=1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            in_5_range += (abs(pred - y) < 5).sum().item()
    test_loss /= num_batches
    in_5_range /= size
    print(f"Error: \n Predictions in 5 range: {(100*in_5_range):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train(dataloader, model, loss_fn, optimizer):
    def closure():
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        return loss

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float().unsqueeze(dim=1)
        if isinstance(optimizer, Eve):
            loss = optimizer.step(closure)
        else:
            loss = closure()
            optimizer.step()

def train_test(train_path='Data/train_data.csv', test_path='Data/test_data.csv',
               optimizer=None, loss_fn=nn.L1Loss(), epochs=5, lr=1e-3,
               layers=[1000,500,250], ps=0.35, train_loss=True):

    train_set = CustomDataset(pd.read_csv(train_path))
    test_set = CustomDataset(pd.read_csv(test_path))
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True)

    model = NN(layers, ps=ps).to(device)
    print(model, end='')
    if not optimizer: 
        optimizer = Eve(model.parameters())
        print('Optimizer: default EVE')
    else: 
        optimizer = optimizer(model.parameters(), lr=lr)
        print('Optimizer: {}\nLearning rate: {}'.format(optimizer.__class__, lr))
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        if train_loss:
            print('Train ', end='')
            test(train_dataloader, model, loss_fn)
        print('Test ', end='')
        test(test_dataloader, model, loss_fn)
    return model
