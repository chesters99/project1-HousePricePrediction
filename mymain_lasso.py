import pandas as pd
import numpy as np
from scipy.stats import skew

def rmse(predictions, actuals):
    return np.sqrt(np.mean( (predictions - actuals)**2 ))

def scale(data, mean_, std_, standardize=True):
    if not standardize:
        return data
    data = (data-mean_) / std_
    return data

def one_step_lasso(r, x, lam): # translated to python from code per template
    xx = np.sum(x**2) 
    xr = np.sum(r*x)
    b = (np.abs(xr) - lam/2) / xx 
    b = np.sign(xr) * (b if b>0 else 0)
    return(b)

def mylasso(X, y, lam, n_iter=50, standardize=True): # translated to python from code template
    # X: n-by-p design matrix without the intercept
    # y: n-by-1 response vector
    # lam: lambda value
    # n.iter: number of iterations
    # standardize: if True, center and scale X and y. 
    
    # Initial values for residual and coefficient vector b
    p = X.shape[1]
    b = np.zeros(p)
    r = y

    for _ in range(n_iter):
        for j in range(p):
            # 1) Update the residual vector to be the one
            # in blue on p37 of [lec_W3_VariableSelection.pdf]. 
            r = r + X[:,j] * b[j]    
            
            # 2) Apply one_step_lasso to update beta_j
            b[j] = one_step_lasso(r, X[:, j], lam)
            
            # 3) Update the current residual vector
            r = r - X[:,j] * b[j] 
            
    # scale back b and add intercept b0      
    b = b / std_x
    b0 = mean_y - np.dot(mean_x, b)
    return b0, b

train = pd.read_csv('train.csv') 
y_train = np.log(train.Sale_Price).values

train = train.drop(['Sale_Price','PID','Utilities', 'Land_Slope', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF',
                    'Pool_Area', 'Misc_Val', 'Longitude', 'Latitude'], axis=1)
train.loc[train.Garage_Yr_Blt.isnull(),"Garage_Yr_Blt"] = 0

# consolidate rare values into one dummy value
limit = 2
for col in train.select_dtypes(include=["object"]).columns:
    variables = train[col].value_counts()
    for i, v in zip(variables.index, variables.values):
        if int(v) < limit: #2, 6 are good
            train.loc[train[col]==i, col] = 'xxxx'

train = pd.get_dummies(train) # convert categoricals to dummy variables
train = train.reindex(sorted(train.columns), axis=1) # sort columns to be in same sequence as test

# log transform all variables with skew > 2 to improve LASSO performance
numerics = train.select_dtypes(exclude=["object"])
skewness = numerics.apply(lambda x: skew(x))
skewed_cols = skewness[abs(skewness) >= 5].index 
skewed_cols = skewed_cols[skewed_cols!='Sale_Price']
train.loc[:, skewed_cols] = np.log1p(train[skewed_cols])

X_train = train.values # convert to numpy array

# Process test file in same way as train
testO = pd.read_csv('test.csv') 
test = testO.drop(['PID','Utilities', 'Land_Slope', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area',
                  'Misc_Val', 'Longitude', 'Latitude'], axis=1)
test.loc[test.Garage_Yr_Blt.isnull(),"Garage_Yr_Blt"] = 0

# consolidate rare values into one dummy value
for col in test.select_dtypes(include=["object"]).columns:
    variables = test[col].value_counts()
    for i, v in zip(variables.index, variables.values):
        if int(v) < limit:
            test.loc[test[col]==i, col] = 'xxxx'

test = pd.get_dummies(test) # convert categoricals to dummy variables

all_columns = train.columns.union(test.columns) # add columns to test that are in train but not test
test = test.reindex(columns=all_columns).fillna(0)
test = test.reindex(sorted(train.columns), axis=1) # sort columns to be in same sequence as train
test.loc[:, skewed_cols] = np.log1p(test[skewed_cols])

X_test = test.values # convert to numpy array

# calculate means and std and then scale X and center Y if standardize is True
mean_x = X_train.mean(axis=0)
std_x  = X_train.std(axis=0)
X = scale(X_train, mean_x, std_x, True)

mean_y = y_train.mean()
y = scale(y_train, mean_y, 1, True)  # Only mean centering, not standardizing

# Train model and make predictions
b0, b = mylasso(X, y, lam=10, n_iter=100, standardize=True)
predsL = b0 + np.dot(X_test, b) 

# Save predictions to mysubmisison3.txt file
LassoManual_df = pd.DataFrame({'PID': testO.PID, 'Sale_Price': np.exp(predsL).round(2)})
LassoManual_df.to_csv('mysubmission3.txt', index=False)
print('Created mysubmission3.txt, rows=', LassoManual_df.shape[0])
