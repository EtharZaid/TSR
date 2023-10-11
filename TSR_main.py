# %%

#Ethar Alzaid

from lifelines import KaplanMeierFitter
import pandas as pd
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index as cindex
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from TransductiveSurvivalRanker import TransductiveSurvivalRanker as TSR
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from pysurvival.utils._metrics import _concordance_index as pysurv_cindex

#Specify which endpoint to work with
time = 'OSTime'
event = 'OS'


# ------Data preparation------#

gene_expressions = pd.read_excel(r'Datasets/TCGA_BRCA_.xlsx')
survival_data = pd.read_excel(r'Datasets/TCGA_BRCA_Hormones_Surv.xlsx')

#-------------

#Fetch survival data

gene_expressions['SAMPLE_ID']=[p[:12] for p in gene_expressions['SAMPLE_ID']]
gene_expressions=gene_expressions.set_index('SAMPLE_ID')

survival_data=survival_data.loc[:,['SAMPLE_ID',time,event]]
survival_data=survival_data.set_index('SAMPLE_ID')
dataset= survival_data.join(gene_expressions, on='SAMPLE_ID')

dataset.dropna(inplace=True)  # Remove rows with no values for event/time
dataset.reset_index(inplace=True, drop=True)


#%%

#Prepare required parameters

#lambda w controls the strength of the L_2 regularization and the average ranking loss for all comparable pairs in P(R)
lambda_w = 1.1

#lambda u controls the loss over the prediction scores of the samples in the test set.  
lambda_u = 0.5

#Specify the type of regularization (can be changed to 1 to perform Lasso)
p = 2


# Censoring time (10 years)
censoring = 3652

train_size = int(len(dataset) * 0.75)


# Split the data into training and test sets with replacement
train = resample(dataset, n_samples=train_size,
                 replace=True, stratify=dataset[event])
train_data = dataset.iloc[train.index]
train_data.reset_index(inplace=True, drop=True)

test_data = dataset.drop(train.index)
test_data.reset_index(inplace=True, drop=True)

# split the data into covariates, duration time, and events

T_train = np.array(train_data.loc[:, time])
E_train = np.array(train_data.loc[:, event])
X_train = np.array(train_data.drop([time, event], axis=1))

# #Censoring
E_train[T_train>censoring]=0
T_train[T_train>censoring]=censoring


# Test data
E_test = np.array(test_data.loc[:, event])
T_test = np.array(test_data.loc[:, time])
X_test = np.array(test_data.drop([time, event], axis=1))

scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

 

# ------------------Run Transductive Learning--------------

#initialize the model
tsr_model = TSR(lambda_w=lambda_w, lambda_u=lambda_u, p=p, Tmax=2000, lr=1e-4)

# Fit the model to the training set
tsr_model.fit(X_train, T_train, E_train, X_test)

# Get test set predictions and calculate the concordance index
Z_test = tsr_model.decision_function(X_test)
c_index=cindex(T_test, Z_test, E_test*(T_test < censoring))
print("\nTSR c-index: %.2f" % c_index)


#%%

#----------------KM curves and significance----------------

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

threshold = 0

#Split into high/low risk groups based on the prediction score
Results_df = pd.DataFrame(
    {'Prediction': Z_test, 'Time': T_test, 'Event': E_test})
low_group = Results_df[Results_df['Prediction'] <= threshold]
high_group = Results_df[Results_df['Prediction'] > threshold]

#logrank test to calculate significance
results = logrank_test(low_group['Time'], high_group['Time'], event_observed_A=low_group['Event'],
                           event_observed_B=high_group['Event'])


print("TSR p-value: %.4f" % results.p_value)


high_T = high_group['Time']
high_E =  high_group['Event']

low_T = low_group['Time']
low_E =  low_group['Event']


#Censoring
high_E[high_T>censoring]=0
high_T[high_T>censoring]=censoring
low_E[low_T>censoring]=0
low_T[low_T>censoring]=censoring


km_high = KaplanMeierFitter()
km_low = KaplanMeierFitter()

# Fitting the model 

ax = plt.subplot(111)
ax = km_high.fit(high_T, event_observed=high_E, label = 'Low Risk').plot_survival_function(ax=ax)
ax = km_low.fit(low_T, event_observed=low_E, label = 'High Risk').plot_survival_function(ax=ax)

add_at_risk_counts(km_high, km_low, ax=ax)
plt.title('Kaplan-Meier estimate \n')
plt.ylabel('Survival probability')
plt.show()
plt.tight_layout()

#%%

#----------------Distrobution of Predections----------------

plt.figure(figsize=(8,6))
plt.style.use('seaborn-whitegrid')

plt.hist(low_group['Prediction'], bins=25, facecolor = '#4890c1', edgecolor='#aabbcc', linewidth=0.5,label=('Low Risk'))
plt.hist(high_group['Prediction'], bins=25, facecolor = '#ecac7c', edgecolor='#fb9942', linewidth=0.5,label=('High Risk'))
plt.title('TSR Prediction Scores',fontsize=16) 
plt.xlabel('Prediction Value',fontsize=16) 
plt.ylabel('Count',fontsize=16)
plt.legend(fontsize=12)
plt.show()


#%%

import seaborn as sns
import matplotlib.pyplot as plt

