import pandas as pd
import csv as csv
import numpy as np
import matplotlib.pyplot as pl
from datetime import datetime
from datetime import date
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn import ensemble
from sklearn import model_selection, metrics
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

data = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/data/train_areas.csv')
predictions = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/data/test_areas.csv')

prices = data['Selling Price']
predictions.drop('Selling Price', axis = 1, inplace = True)
predictions.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)
data.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)
origList_Price = predictions['Listing Price']
orig_address = predictions['Address']


#This shows when a part of the code is processed
def status(feature):

    print 'Processing',feature,': ok'

#If House has pool, then assign a 1, otherwise a 0. 
def get_pool():

    Pool_dict = {
                    "YESY": "1",
                    "NONO": "0"
                }
    data['Pool'] = data.Pool.map(Pool_dict).astype(int)
    predictions['Pool'] = predictions.Pool.map(Pool_dict).astype(int)
    status('Pool')
get_pool()

#Combine Half bathroom with full bathrooms
def get_bath():
    data['Bathrooms - Half'] = data['Bathrooms - Half'] / 2
    data['Total Bathrooms'] = data['Bathrooms - Full'] + data['Bathrooms - Half']

    predictions['Bathrooms - Half'] = predictions['Bathrooms - Half'] / 2
    predictions['Total Bathrooms'] = predictions['Bathrooms - Full'] + predictions['Bathrooms - Half']


get_bath()

#get the number of bedrooms from data
def get_bedroom():
    data['Bedrooms'] = data['Bedrooms_And_Possible_Bedrooms'].map(lambda rooms: rooms.split('(')[0].strip()).astype(int)
    predictions['Bedrooms'] = predictions['Bedrooms_And_Possible_Bedrooms'].map(lambda rooms: rooms.split('(')[0].strip()).astype(int)
    status('Bedrooms')

get_bedroom()
#group the data by zip codes to fill in any null values by area
grouped = data.groupby(data['Address - Zip Code']).mean().astype(int)

#Filling empty lot sizes based off mean of zipcodes. Generalized to include all zip codes. 
def process_lotsize():
    

    def fillLotSize(row):
        zips = list(grouped.index.values)
        lot_mean = list(grouped['Lot Size - Sq Ft'].values)
        for idx, val in enumerate(zips):
            if (row['Address - Zip Code'] == val).any():
                return lot_mean[idx]
    data['Lot Size - Sq Ft'] = data.apply(lambda r: fillLotSize(r) if np.isnan(r['Lot Size - Sq Ft']) else r['Lot Size - Sq Ft'], axis = 1)
    data['Lot Size - Sq Ft'] = data.apply(lambda r: fillLotSize(r) if r['Lot Size - Sq Ft'] == 0 else r['Lot Size - Sq Ft'], axis = 1)
    predictions['Lot Size - Sq Ft'] = predictions.apply(lambda r: fillLotSize(r) if np.isnan(r['Lot Size - Sq Ft']) else r['Lot Size - Sq Ft'], axis = 1)
    predictions['Lot Size - Sq Ft'] = predictions.apply(lambda r: fillLotSize(r) if r['Lot Size - Sq Ft'] == 0 else r['Lot Size - Sq Ft'], axis = 1)
    status('Lot Size - Sq Ft')
process_lotsize()    

#Filling the year built, generalized for all zip codes
def process_yearBuilt():

    def fillYearBuilt(row):
        zips = list(grouped.index.values)
        yearBuilt_mean = list(grouped['Year Built'].values)
        for idx, val in enumerate(zips):
            if (row['Address - Zip Code'] == val).any():
                return yearBuilt_mean[idx]
    data['Year Built'] = data.apply(lambda r: fillYearBuilt(r) if np.isnan(r['Year Built']) else r['Year Built'], axis = 1)
    predictions['Year Built'] = predictions.apply(lambda r: fillYearBuilt(r) if np.isnan(r['Year Built']) else r['Year Built'], axis = 1)
    status('Year Built')

process_yearBuilt()

#Drop variables we don't need
def drop_strings():

    data.drop('Selling Date', axis = 1, inplace = True)
    data.drop('Address', axis = 1, inplace = True)

    predictions.drop('Address', axis = 1, inplace = True)
    predictions.drop('Pending Date', axis = 1, inplace = True)
    predictions.drop('Selling Date', axis = 1, inplace = True)

    status('Dropped Strings')

drop_strings()

#Split the listing date into months and years
def get_dates():

    #get the months and years, not interested in the days
    data['Month'] = data['Listing Date'].map(lambda listdate: listdate.split('/')[0].strip())
    data['Year'] = data['Listing Date'].map(lambda listdate: listdate.split('/')[2].strip())

    predictions['Month'] = predictions['Listing Date'].map(lambda listdate: listdate.split('/')[0].strip())
    predictions['Year'] = predictions['Listing Date'].map(lambda listdate: listdate.split('/')[2].strip())

    status("Get Dates")

get_dates()

#Now we're going to process our dates
def process_dates():

    #Create column with years in integer before dropping the string version
    data['Year Sold'] = data.Year.astype(int)
    predictions['Year Sold'] = predictions.Year.astype(int)

    #Now we drop all the variables that we don't need
    data.drop('Year', axis = 1, inplace = True)
    data.drop('Month', axis = 1, inplace = True)
    
    predictions.drop('Year', axis = 1, inplace = True)
    predictions.drop('Month', axis = 1, inplace = True)
    
    status('Dates')

process_dates()

#Need to work on creating days on market column for test data
def get_marketDate():

    #Need to create Days on Market for training data
    data['Listing Date'] = pd.to_datetime(data['Listing Date'])
    data['Pending Date'] = pd.to_datetime(data['Pending Date'])
    data['Days on Market'] = data['Pending Date'] - data['Listing Date']
    data['Days on Market'] = (data['Days on Market'] / np.timedelta64(1, 'D')).astype(int)

    predictions['temp'] = predictions['Listing Date']
    predictions['temp'] = pd.to_datetime(predictions['temp'])
    predictions['diff'] = predictions['temp'].map(lambda x: datetime.utcnow() - x)
    predictions['Days on Market'] = (predictions['diff'] / np.timedelta64(1, 'D')).astype(int)

    #Listing Date is a string we don't need anymore
    data.drop('Listing Date', axis = 1, inplace = True)
    data.drop('Pending Date', axis = 1, inplace = True)
    data.drop('Bedrooms_And_Possible_Bedrooms', axis = 1, inplace = True)

    predictions.drop('temp', axis = 1, inplace = True)
    predictions.drop('diff', axis = 1, inplace = True)
    predictions.drop('Listing Date', axis = 1, inplace = True)
    predictions.drop('Bedrooms_And_Possible_Bedrooms', axis = 1, inplace = True)
    
    status('Market Day Diff')

get_marketDate()

features = data.drop('Selling Price', axis = 1, inplace = True)

X = data
y = prices

#create a model to evaluate how well the XGBoost algorithm is performing
def modelfitXGB(alg, X, y, useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label = y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold = cv_folds,
            metrics = 'rmse', early_stopping_rounds = early_stopping_rounds, verbose_eval = 99)
        alg.set_params(n_estimators = cvresult.shape[0])
        cv_score = model_selection.cross_val_score(alg, X, y, cv = cv_folds)
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    #Fit the algorithm on the data
    alg.fit(X, y, eval_metric = 'rmse')
        
    #Predict training set:
    train_predictions = alg.predict(X)
        
    #Print model report:
    print "\nModel Report"
    print "Mean Squared Error : %.4g" % metrics.mean_squared_error(y, train_predictions)
    print "r2 Score: %f" % metrics.r2_score(y, train_predictions)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending = False)
    feat_imp.plot(kind = 'bar', title = 'Feature Importances')
    pl.ylabel('Feature Importance Score')
    pl.show()

xgb1 = XGBRegressor(
 learning_rate = 0.05,
 n_estimators = 600,
 max_depth = 5,
 min_child_weight = 1,
 gamma = 0,
 subsample = 0.8,
 colsample_bytree = 0.8,
 reg_alpha = 1,
 reg_lambda = 1,
 objective = 'reg:linear',
 nthread = 4,
 scale_pos_weight = 1,
 seed = 10)

modelfitXGB(xgb1, X, y)

#After viewing the CV score for the model, output predictions to a CSV file for viewing. 
xgb1.fit(X,y)
Y_xgb_pred = xgb1.predict(predictions).astype(int)
xgbOut = Y_xgb_pred
df_xgb = pd.DataFrame()
df_xgb['Listing Price'] = origList_Price
df_xgb['Predicted Selling Price'] = xgbOut
df_xgb['Address'] = orig_address
df_xgb[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/data/XGBoostPredictions.csv',index=False)