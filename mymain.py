#!/usr/bin/env python

import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.decomposition import PCA
np.warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb

# function to add additional variables to train and test
def add_derived(data):       
    # create derived values to help with modelling
    data["zTotalHouse"] = data["Total_Bsmt_SF"] + data["First_Flr_SF"] + data["Second_Flr_SF"]   
    data["zTotalArea"]  = data["Total_Bsmt_SF"] + data["First_Flr_SF"] + data["Second_Flr_SF"] + data["Garage_Area"]
    data["zTotalHouse_OverallQual"] = data["zTotalHouse"] * data["oOverall_Qual"]
    data["zGrLivArea_OverallQual"] = data["Gr_Liv_Area"] * data["oOverall_Qual"]
    data["zoMSZoning_TotalHouse"] = data["oMS_Zoning"] * data["zTotalHouse"]
    data["zMSZoning_OverallQual"] = data["oMS_Zoning"] + data["oOverall_Qual"]
    data["zMSZoning_YearBuilt"] = data["oMS_Zoning"] + data["Year_Built"]
    data["zNeighborhood_TotalHouse"] = data["oNeighborhood"] * data["zTotalHouse"]
    data["zNeighborhood_OverallQual"] = data["oNeighborhood"] + data["oOverall_Qual"]
    data["zNeighborhood_YearBuilt"] = data["oNeighborhood"] + data["Year_Built"]
    data["zBsmtFinSF1_OverallQual"] = data["BsmtFin_SF_1"] * data["oOverall_Qual"]
    data["zFunctional_TotalHouse"] = data["oFunctional"] * data["zTotalHouse"]
    data["zFunctional_OverallQual"] = data["oFunctional"] + data["oOverall_Qual"]
    data["zLotArea_OverallQual"] = data["Lot_Area"] * data["oOverall_Qual"]
    data["zTotalHouse_LotArea"] = data["zTotalHouse"] + data["Lot_Area"]
    data["zCondition1_TotalHouse"] = data["oCondition_1"] * data["zTotalHouse"]
    data["zCondition1_OverallQual"] = data["oCondition_1"] + data["oOverall_Qual"]
    data["zBsmt"] = data["BsmtFin_SF_1"] + data["BsmtFin_SF_2"] + data["Bsmt_Unf_SF"]
    data["zRooms"] = data["Full_Bath"]+data["TotRms_AbvGrd"]
    data["zPorchArea"] = data["Open_Porch_SF"]+data["Enclosed_Porch"]+data["Three_season_porch"]+data["Screen_Porch"]
    data["zTotalPlace"] =data["Total_Bsmt_SF"]+data["First_Flr_SF"]+data["Second_Flr_SF"] + data["Garage_Area"] + \
                         data["Open_Porch_SF"]+data["Enclosed_Porch"]+data["Three_season_porch"]+data["Screen_Porch"] 
    return data

def rmse(predictions, actuals):
    return np.sqrt(np.mean( (predictions - actuals)**2 ))

#function to process the test and train datasets to prepare for modeling
def process(trainO, testO):
    train = trainO.copy()
    train = train.drop(['PID','Utilities','Condition_2',], axis=1)
    train['Sale_Price'] = np.log1p(train.Sale_Price)
    train.loc[train.Garage_Yr_Blt.isnull(),"Garage_Yr_Blt"] = train.Year_Built
    should_be_categoricals = ['Bsmt_Full_Bath','Bsmt_Half_Bath','Full_Bath','Half_Bath','Bedroom_AbvGr',
                              'Kitchen_AbvGr','TotRms_AbvGrd','Fireplaces','Garage_Cars']
    for col in should_be_categoricals:
        train[col] = train[col].astype(str)
        
#     create lookup table to add new columns with mean sale price of categorical column
    tables = []
    for col in train.select_dtypes(include=["object"]).columns:
        lookup = train.groupby([col])[['Sale_Price']].agg(['mean']).add_suffix("_val").reset_index()
        lookup.columns = ['value', 'number']
        lookup['column'] = col
        tables.append(lookup)
    lookup = pd.concat(tables)
        
    for col in train.select_dtypes(include=["object"]).columns: # create new columns with mean of sale price
        train['o'+col] = train[col].map(lookup[lookup.column==col].set_index('value')['number'])
        
#     consolidate rare values into one dummy value on train dataset
    for col in train.select_dtypes(include=["object"]).columns:
        variables = train[col].value_counts()
        for i, v in zip(variables.index, variables.values):
            if int(v) < 4:
                train.loc[train[col]==i, col] = 'xxxx'

#     get skewed columns from train data and apply log to each
    numerics = train.select_dtypes(exclude=["object"])
    skewness = numerics.apply(lambda x: skew(x))
    skewed_cols = skewness[abs(skewness) >= 5].index # affects 5 columns
    skewed_cols = skewed_cols[skewed_cols!='Sale_Price']
    train.loc[:, skewed_cols] = np.log1p(train[skewed_cols])

    train = add_derived(train)     # apply transformations to train datafram
    train = pd.get_dummies(train) # create dataframe with dummy variables replacing categoricals

    train = train.reindex(sorted(train.columns), axis=1) # sort columns to be in same sequence at train

    # split into X and y for train
    y_train = train.loc[:, 'Sale_Price']
    train = train.drop(['Sale_Price'], axis=1)

    #  test.csv  apply same transformations as for train dataset
    test = testO.copy()
    test = test.drop(['PID','Utilities','Condition_2'], axis=1)
    test.loc[test.Garage_Yr_Blt.isnull(),"Garage_Yr_Blt"] = test.Year_Built
    for col in should_be_categoricals:
        test[col] = test[col].astype(str)
    
    for col in test.select_dtypes(include=["object"]).columns:
        test['o'+col] = test[col].map(lookup[lookup.column==col].set_index('value')['number'])
    
    test[skewed_cols] = np.log1p(test[skewed_cols]) # apply log1p to skewed columns as per train
    test = add_derived(test) # apply transformations to test dataframe

#     loop through each test column and replace any values that arent in train with the most common train value
    for col in testO.select_dtypes(include=["object"]).columns:
        testO.loc[testO[~testO[col].isin(trainO[col].unique())].index, col] = 'xxxx'

    test = pd.get_dummies(test) # create dataframe with dummy variables replacing categoricals

    all_columns = train.columns.union(test.columns) # add columns to test that are in train but not test
    test = test.reindex(columns=all_columns).fillna(0)
    test = test.reindex(sorted(train.columns), axis=1) # sort columns to be in same sequence at train
    
    # Add first X principal components 
    pca = PCA(n_components=5) # was 5
    train = np.concatenate([train, pca.fit_transform(train) ], axis=1)
    test  = np.concatenate([test, pca.transform(test)], axis=1)
#     print('Test  dataset dimensions (original and processed)', testO.shape, test.shape,'\n')
    
    return train, y_train, test


# MAIN SECTION - READ FILES
trainO = pd.read_csv('train.csv')
testO = pd.read_csv('test.csv')
X_train, y_train, X_test  = process(trainO, testO)

# TRAIN FIRST MODEL AND CREATE SUBMISSION FILE
seed=42
np.random.seed(seed)
m1 = xgb.XGBRegressor(learning_rate=0.03, n_estimators=1100,min_child_weight=7,max_depth=5,gamma=0,subsample=0.8, 
                     colsample_bytree=0.5, reg_lambda=0.3, reg_alpha=0.4,  n_jobs=-1, random_state=seed)
_ = m1.fit(X_train, y_train)
m1_preds = np.expm1( m1.predict(X_test) ) 
m1_df = pd.DataFrame({'PID': testO.PID, 'Sale_Price': m1_preds.round(2)})
m1_df.to_csv('mysubmission1.txt', index=False)
print('Created mysubmission1.txt, rows=', m1_df.shape[0],', Model=', type(m1).__name__)

# TRAIN SECOND MODEL AND CREATE SUBMISSION FILE
seed=42
np.random.seed(seed)
m2 =  GradientBoostingRegressor(learning_rate=0.02, n_estimators=1200, max_depth=4, min_samples_split=2,
                min_samples_leaf=5, min_weight_fraction_leaf=0, subsample=0.9, max_features='sqrt', random_state=seed)
_ = m2.fit(X_train, y_train)
m2_preds = np.expm1( m2.predict(X_test) ) 
m2_df = pd.DataFrame({'PID': testO.PID, 'Sale_Price': m2_preds.round(2)})
m2_df.to_csv('mysubmission2.txt', index=False)
print('Created mysubmission2.txt, rows=', m2_df.shape[0], ', Model=', type(m2).__name__) 

