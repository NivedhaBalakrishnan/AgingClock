# Summary:

## Problem:  
Predicting biological age with first with methylation genome data, then blood test data and eventually with multimodal omics and lifestyle data.  
## Why:  
Aging is the most important disease of the modern world that everyone of us shall deal with at some point.
  
*"It has been estimated that the complete elimination of a single fatal disease such as cancer in the USA would merely lead to a 2.3-year population increase in life expectancy ... since the majority of overall mortality is due to age-related diseases."[[2]](#ref2)*

## Data:  
Data of around 4k entries from the paper[[1]](#ref1).  
- GTEx Portal  
- CDC website, NHANES Dataset  
## Preprocessing/Training:  
- Standartization/Normalization  
- PCA, GBR, DNN and maybe other regression models to check their performance.[[1]](#ref1)  
- Measuring the results based on healthy individuals (assuming their chronological age is close to their biological age)  
## Testing/Validating:  
Since first I think it is more appropriate to play with genome data and apply already made sklearn models, and only then when we will add other types of data like lifestyle information then it is more appropriate to integrate a more low-level level framework like pytorch.  


# Presentation transcript:

Let me first answer the most important question. 

> **Why is it important?** 

Well, from my perspective life is the most valuable thing that any of us has and in this way trying to increase healthspan of you, your loved ones and the
humanity as a whole is a worthwhile goal. 

> **Why focus on aging research in particular?**

First of all, concerns all of us -- minute by minute we are all aging right now and as of now there is no way to stop or reverse this process.  
Looking at my premises you may argue that there are other important diseases to focus on that are a leading cause of death like cancer or heart diseases.
However, curing cancer will only prolong the average lifespan by a few years, and what's even worse is the amount of active years (healthspan) is going to be the same.   
So, instead of thinking about the effects mentioned before like heart diseases, cancer, or Alzheimer's we should focus on the cause of all them which is aging. 

One of the prominent theories of aging these days says that we age because our epigenetic information is lost as we age. 
So, imagine that the DNA or genome is like a hardware of our body and epigenetic information is a software that says what programs to run on this hardware basically saying which parts of the genetic 
code should be actively executed. With time this software 'rots' and the wrong code is executed and the important parts of the program are commented out.  
In biological terms you can see this as a cell in your brain is due to the wrong epigenetic 
information behaves like a skin cell and vice versa. 

So, having this big picture in mind, you may ask:  
> **Why deep learning?** 

Well health industry generates tons and tons of electronic data especially nowadays, and all 
of this data from clinical trials to medical records in hospitals can be used for every step in drug discovery.  
However this is too complicated for a project of this scale, so let's focus on more manageable task which is predicting accurate biological age -- one of the methods of doing so is by inspecting epigentic information mentioned before. So the goal here is to produce accurate biological age so we can measure how effective are drugs that we discover. And here we can see that all off the data can go into a deep neural network and as an output we get 
the biological age and health status.

But I would like to start with something even simpler, for example using only methylation data of the genome, that is basically one way how epigenome manifests itself. Focusing only on it can give us a good prediction of a person's age and this
data is backbone of many modern Deep Aging Clocks.

So what I'm going to describe now is primarily based on this study. Where we have a healthy group and a diseased group with their respective epigenetic information.  
The goal here is based on this info to figure out biological age. Since we assume that healthy individuals have roughly the same biological and chronological age we can use it as a ground truth.  
In this study they figured that the best regression 
model for this task is Gradient Boosting Regressor and as a first step of this project it would be great to just emulate this paper. Then we can go on to expand our dataset to some other biological data like blood work and train a deep learning network to manage the complexity of the increasing dataset.  
Data in this study is open and after applying Principal Component Analysis and other data cleaning tools we have around 4k of labeled data entries.  
That should be enough initially but for deep learning models there are huge data sets of similar information with tens or even hundreds of thousands samples. Here is one example that I found from GTExPortal however more data preprocessing is required in this case. 

Measuring data can be accomplished either by computing the error with regard to healthy individuals or other comprehensive clocks that are used in practice.

As far as engineering tools, the ones that will be used here are based on my opinion:  
Git as a version control tool, GitHub as a platform and ZenHub for managing the project.

For compute and storage Google Cloud platform is a good option since their services are arguably better than those provided by AWS especially with their
specialized Vertex AI Platform and access to custom-build Goggle TPUs. 

Docker for easier development and deployment and as a first step tools like Voila and Binder are enough for development, deployment, and hosting of a 
website. 

Later on if everything goes well I can imagine using terraform for easier cloud management and kubernetes.  


## References:  
[1]<a name="ref1"></a>
[Li, X.; Li, W.; Xu, Y. Human Age Prediction Based on DNA Methylation Using a
Gradient Boosting Regressor. Genes 2018, 9, 424.](https://doi.org/10.3390/genes9090424)

[2]<a name="ref2"></a>
[Zhavoronkov, A., Bischof, E. & Lee, KF. Artificial intelligence in
longevity medicine. Nat Aging 1, 5–7 (2021).](https://doi.org/10.1038/s43587-020-00020-4)   
