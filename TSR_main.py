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


# %%

#Prepare required parameters

#lambda w controls the strength of the L_2 regularization and the average ranking loss for all comparable pairs in P(R)
lambda_w = 1.1

#lambda u controls the loss over the prediction scores of the samples in the test set.  
lambda_u = 0.5

#Specify the type of regularization (can be changed to 1 to perform Lasso)
p = 2

n_samples = 10

# Censoring time (10 years)
censoring = 3652

train_size = int(len(dataset) * 0.75)
Bootstrap_p_Values = []
Bootstrap_cindex = []

for _ in tqdm(range(n_samples)):

    # Split the data into training and test sets with replacement
    train = resample(dataset, n_samples=train_size,
                     replace=True, stratify=dataset[event])
    
    # Split the data into training and test sets
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
    Bootstrap_cindex.append(cindex(T_test, Z_test, E_test*(T_test < censoring)))

    threshold = 0

    #Split into high/low risk groups based on the prediction score
    Results_df = pd.DataFrame(
        {'Prediction': Z_test, 'Time': T_test, 'Event': E_test})
    low_group = Results_df[Results_df['Prediction'] <= threshold]
    high_group = Results_df[Results_df['Prediction'] > threshold]

    #logrank test to calculate significance
    results = logrank_test(low_group['Time'], high_group['Time'], event_observed_A=low_group['Event'],
                               event_observed_B=high_group['Event'])
    Bootstrap_p_Values.append(results.p_value)

    

#Calculate evaluation metrics

mean_score_Tsr = np.mean(Bootstrap_cindex)
std_score_Tsr = np.std(Bootstrap_cindex)
combined_p=2*np.median(Bootstrap_p_Values)

print("\nTSR Mean ,SD,2p50:\n")
print("Mean c-index: %.2f" % mean_score_Tsr)
print("Standardd deviation of c-index: %.2f" % std_score_Tsr)
print("Combined p-Value: %.4f" % combined_p)


