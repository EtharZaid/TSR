# TSR: Transductive Survival Ranking for Cancer Risk Stratification


This repository contains the code for the following papers:

A Transductive Approach to Survival Ranking for Cancer Risk Stratification, in The 18th Machine Learning in Computational Biology (MLCB2023) [1](https://proceedings.mlr.press/v240/alzaid24a.html#:~:text=It%20incorporates%20unlabeled%20test%20samples,processing%20or%20manual%20threshold%20selection.).

Transductive Survival Ranking for Pan-cancer Automatic Risk Stratification using Whole Slide Images, in The 29th Medical Image Understanding and Analysis (MIUA2025).


## Introduction
How can we stratify patients into meaningful risk groups by leveraging data from both patients with known survival times and event indicators and those without, while requiring no manual post-hoc thresholding of predicted risk scores? Existing survival stratification methods in computa-tional pathology train a supervised model on patients with known survival times and event indicators, then apply it to a test set to generate risk scores. These scores are typically thresholded, often at the median, to assign pa-tients to high- or low-risk groups. Such inductive pipelines overlook the large pool of unlabelled patients even though number of cases with known survival times are typically limited and observed events are even rarer. As a result, existing methods often fail to uncover meaningful risk groups. In this work, we introduce Transductive Survival Ranking (TSR) model, designed to leverage both labelled and unlabelled data for improved survival prediction. Given a dataset where only a subset of samples have associated survival time and event information, our approach (1) ranks patients by predicted survival times, (2) automatically discovers risk groups without requiring manual thresholding, and (3) transduces differential survival patterns from patients with observed events to those without events. 

<img src="TSR_overview.png" alt="Concept Diagram"/>

## Dependencies

lifelines==0.27.7 

matplotlib==3.5.3

numpy==1.21.6

pandas==1.3.5

pynverse==0.1.4.6

pysurvival==0.1.2

scikit-learn==1.0.2

scipy==1.7.3

seaborn==0.12.2

torch==1.13.1

tqdm==4.65.0

umap==0.1.1

wrapt==1.15.0

Python Version== 3.7.16

## Usage
### Data download
A sample file for BRCA is included in the `Dataset` folder with a separate file for survival information.
This was downloaded from TCGA on cBioPortal (https://www.cbioportal.org/), the gene expressions can be directly downloaded with z-score normalization.
We have included WSI embeddings for TCGA-BRCA generated using [TITAN](https://github.com/TencentAILabHealthcare/TITAN) in the `Dataset` folder.



### Single Run
In [TSR_main](TSR_main.py) (to be updated), you will be able to run the transductive survival ranking to stratify samples into low vs. high risk groups:

It will calcualte (refer to the paper for more details):

- C-index

- p-Value

And will plot the following:

- Distribution of prediction scores

- Kaplan-Meier Curves

- Heatmap of gene expressions of the stratified groups

### Bootstrap Run
In [TSR_Bootstrap](TSR_BootstrapRun.py), you can perform bootstrap runs and it will calculate the mean c-index, standard deviation and combined p-value.




