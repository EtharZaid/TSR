'''
This is the code used to construct the experiments in the paper:
A Transductive Approach to Survival Ranking for Cancer Risk Stratification

This file is the TSR model, which applies transductive learning to achieve
automatic subgrouping
Transduction loss will not be calculated if the test set was omitted in the fit() function
'''
import torch
import torch.optim as optim
from lifelines.utils import concordance_index

USE_CUDA = torch.cuda.is_available() 
from torch.autograd import Variable
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def toTensor(v,dtype = torch.float,requires_grad = False):       
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))
def toNumpy(v):
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()

def TransductiveLoss(z):
    """
        ----Calculates the transductive loss using test sample prediction----
        Parameters
        ----------
        z : np array
            Prediction scores of the test set

        Return
        ----------
        Transductive loss
        """
    closs = torch.exp(-3*(z**2)) 
    return closs

class TransductiveSurvivalRanker:
    def __init__(self,model=None,lambda_w=0.1,lambda_u = 0.0,p=2,lr=1e-2,Tmax = 200):
        """
        ----Model initialization----
        Parameters
        ----------
        lambda_w : float (default=0.1)
                Controls the strength of the L_2 regularization and the average ranking loss 
                for all comparable pairs in P(R).
        lambda_u : float (default=0)
                Controls the loss over the prediction scores of the samples in the test set.
        p: int (default=2)
                Specify the type of regularization 
        lr : float (default=1e-2)
                Learning rate
        Tmax : int (default=200)
                Max training epochs
        """
        self.lambda_w = lambda_w
        self.lambda_u = lambda_u
        self.p = p
        self.Tmax = Tmax
        self.lr = lr
        self.model = model
        
        
    def fit(self,X_train,T_train,E_train,X_test = None):        
       
        """
        ----Model fitting function----
        Parameters
        ----------
        X_train : np array 
                Covariates (normalized gene expressions) of the train set
        T_train : np array
                Time duration of samples in the training set
        E_train : np array
                Event indicators of samples in the training set 
        X_test : np array (default=None)
                Covariates (normalized gene expressions) of the test set                
        
        Return
        ----------
        Instance of the trained model
        
        """
        #Handle data as tensors
        x = toTensor(X_train)
        if X_test is not None:
            X_test = toTensor(X_test)
        y = toTensor(T_train)
        e = toTensor(E_train)
        
        #NN input and output size
        N,D_in = x.shape        
        H, D_out = D_in, 1
        
        #Create a new model if none exist using a linear nn with tanh activation
        if self.model is None:                    
            self.model = torch.nn.Sequential( 
                torch.nn.Linear(H, D_out,bias=True),
                torch.nn.Tanh()
            )
        model = self.model
        model=cuda(model)
        learning_rate = self.lr
        
        #initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0)
        epochs = self.Tmax  
        lambda_w = self.lambda_w 
        p = self.p
        L = [] #collects losses for all epochs
        
        #Create pairs of samples where E_j=1 AND T_i > T_j
        dT = T_train[:, None] - T_train[None, :] 
        dP = (dT>0)*E_train
        dP = toTensor(dP,requires_grad=False)>0

        self.bias = 0.0
        loss_uv = 0.0
        for t in (range(epochs)):
            
            #Get predictions on the training set and calculate Ranking Loss and add it to the overall loss
            y_pred = model(x).flatten()
            dZ = (y_pred.unsqueeze(1) - y_pred)[dP]  
            loss = torch.mean(torch.max(toTensor([0],requires_grad=False),1.0-dZ))
            
            #Get predictions on the test set and calculate Transductive Loss and add it to the overall loss
            if X_test is not None and self.lambda_u > 0:
                test_predictions = model(X_test).flatten()
                loss_u = torch.mean(TransductiveLoss(test_predictions))
                transductive_loss=self.lambda_u*loss_u
                loss+=transductive_loss
                loss_uv = loss_u.item()

            w = model[0].weight.view(-1) #only input layer weights (exclude bias from regularization)
            
            #Calculate the regularization term and add it to the overall loss
            regularization_term=lambda_w*torch.norm(w, p)**p 
            loss+=regularization_term
            L.append([loss.item(),loss_uv])
            
            # Calculate the gradient during the backward pass and perform a single optimization step (update weights w)
            model.zero_grad()
            loss.backward() 
            optimizer.step() 
                       
        w = model[0].weight.view(-1)
        self.w = w
        self.L = L
        self.model = model
        return self
    
    
    def decision_function(self,x):
        """
         ----Predictor function----
         Parameters
         ----------
         x : np array
             Test samples covariates
          
        Return
        ----------
        Survival prediction scores  
          
         """
        x = toTensor(x)
        return toNumpy(self.model(x)-self.bias).flatten()
        
    def getW(self):
        """
         ----gets the optimized weights of the model----
        Return
        ----------
        np array of all weights of the model
          
         """
        
        return toNumpy(self.w/torch.linalg.norm(self.w,ord=1))
