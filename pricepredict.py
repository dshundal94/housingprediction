# ***This script creates a model to predict housing prices in Lodi, CA with
# zipcodes 95240 and 95242. In the future, if model is accurate based on mean squared error
# then will generalize to include many more zip codes. I find that Random Forest does not work
# quite well to model the prediction, and my best bet would be to fine tune regression paramaters
# or try using Ridge Regression to deal with multicollinearity. **** --By Damanjit Hundal 



import pandas as pd
import csv as csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
import statsmodels.formula.api as sm
from datetime import datetime
from datetime import date
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#Reading in the csv file, this is the training set
data = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/sold3.csv')
test = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/testing3.csv')

#extract and remove targets from training data 
targets = data['Selling Price']
origList_Price = test['Listing Price']
orig_address = test['Address']
data.drop('Selling Price', axis = 1, inplace = True)
test.drop('Selling Price', axis = 1, inplace = True)
data.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)
test.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)

#This will tell us if our particular part of code ran
def status(feature):

    print 'Processing',feature,': ok'

#If House has pool, then assign a 1, otherwise a 0. 
def get_pool():

    Pool_dict = {
                    "YESY": "1",
                    "NONO": "0"
                }
    data['Pool'] = data.Pool.map(Pool_dict).astype(int)
    test['Pool'] = test.Pool.map(Pool_dict).astype(int)

get_pool()

#Combine Half bathroom with full bathrooms
def get_bath():
    data['Bathrooms - Half'] = data['Bathrooms - Half'] / 2
    data['Total Bathrooms'] = data['Bathrooms - Full'] + data['Bathrooms - Half']

    test['Bathrooms - Half'] = test['Bathrooms - Half'] / 2
    test['Total Bathrooms'] = test['Bathrooms - Full'] + test['Bathrooms - Half']

    #Drop Half and Full bathrooms, because feature redundant
    data.drop('Bathrooms - Full', axis = 1, inplace = True)
    data.drop('Bathrooms - Half', axis = 1, inplace = True)

    test.drop('Bathrooms - Full', axis = 1, inplace = True)
    test.drop('Bathrooms - Half', axis = 1, inplace = True)

get_bath()

#Hardcoding the lot size based off median of zipcode. will generalize later to include all zip codes. 
def process_lotsize():
    global data
    global test

    def fillLotSize(row):
        if row['Address - Zip Code'] == '95240':
            return 5756
        elif row['Address - Zip Code'] == '95242':
            return 6456
    data['Lot Size - Sq Ft'] = data.apply(lambda r: fillLotSize(r) if np.isnan(r['Lot Size - Sq Ft']) else r['Lot Size - Sq Ft'], axis = 1)
    test['Lot Size - Sq Ft'] = test.apply(lambda r: fillLotSize(r) if np.isnan(r['Lot Size - Sq Ft']) else r['Lot Size - Sq Ft'], axis = 1)
    status('Lot Size - Sq Ft')
process_lotsize()



#Harcode the year built, will generalize for all zip codes later
def process_yearBuilt():
    global data
    global test

    def fillYearBuilt(row):
        if row['Address - Zip Code'] == 95240:
            return 1963
        elif row['Address - Zip Code'] == 95242:
            return 1985
    data['Year Built'] = data.apply(lambda r: fillYearBuilt(r) if np.isnan(r['Year Built']) else r['Year Built'], axis = 1)
    test['Year Built'] = test.apply(lambda r: fillYearBuilt(r) if np.isnan(r['Year Built']) else r['Year Built'], axis = 1)
    status('Year Built')

process_yearBuilt()

#Use binary option to show zipcodes
def process_area():
    global data
    global test

    #Clean the address variable
    data.drop('Address', axis = 1, inplace = True)
    test.drop('Address', axis = 1, inplace = True)

    #encode dummy variables
    zipcode_dummies = pd.get_dummies(data['Address - Zip Code'], prefix = 'Zip Code')
    data = pd.concat([data, zipcode_dummies], axis = 1)

    zipcode_test_dummies = pd.get_dummies(test['Address - Zip Code'], prefix = 'Zip Code')
    test = pd.concat([test, zipcode_test_dummies], axis = 1)
    #remove the zip code title
    data.drop('Address - Zip Code', axis = 1, inplace = True)
    test.drop('Address - Zip Code', axis = 1, inplace = True)

    status('Zip Code')

process_area()



#Drop variables we don't need
def drop_strings():

    global data
    global test

    data.drop('Selling Date', axis = 1, inplace = True)

    #The test data does not have a pending date or a sold date
    test.drop('Pending Date', axis = 1, inplace = True)
    test.drop('Selling Date', axis = 1, inplace = True)

drop_strings()

#Split the listing date into months and years
def get_dates():

    global data
    global test

    #get the months and years, not interested in the days
    data['Month'] = data['Listing Date'].map(lambda listdate: listdate.split('/')[0].strip())
    data['Year'] = data['Listing Date'].map(lambda listdate: listdate.split('/')[2].strip())

    #Now for the test data
    test['Month'] = test['Listing Date'].map(lambda listdate: listdate.split('/')[0].strip())
    test['Year'] = test['Listing Date'].map(lambda listdate: listdate.split('/')[2].strip())


    status("Get Dates")

get_dates()

#Now we're going to process our dates
def process_dates():

    global data
    global test

    data['Season'] = data['Month']
    test['Season'] = test['Month']

    Date_Dictionary = {
                        "1":    "Winter",
                        "2":    "Winter",
                        "3":    "Winter",
                        "4":    "Spring",
                        "5":    "Spring",
                        "6":    "Spring",
                        "7":    "Summer",
                        "8":    "Summer",
                        "9":    "Summer",
                        "10":   "Fall",
                        "11":   "Fall",
                        "12":   "Fall"
                        }

    # We have to map each season
    data['Season'] = data.Season.map(Date_Dictionary)
    test['Season'] = test.Season.map(Date_Dictionary)

    #Encode dummy variables for the seasons
    season_dummies = pd.get_dummies(data['Season'], prefix = 'Season')
    data = pd.concat([data, season_dummies], axis =1)

    season_test_dummies = pd.get_dummies(test['Season'], prefix = 'Season')
    test = pd.concat([test, season_test_dummies], axis =1)

    #Create column with years in integer before dropping the string version
    data['intYear'] = data.Year.astype(int)
    test['intYear'] = test.Year.astype(int)

    #Now we drop all the variables that we don't need
    data.drop('Year', axis = 1, inplace = True)
    data.drop('Month', axis = 1, inplace = True)

    test.drop('Year', axis = 1, inplace = True)
    test.drop('Month', axis = 1, inplace = True)

    status('Dates')

process_dates()


#Need to work on creating days on market column for test data
def get_marketDate():

    global test
    global data

    #Can use datetime in panda series
    test['temp'] = test['Listing Date']
    test['temp'] = pd.to_datetime(test['temp'])
    test['diff'] = test['temp'].map(lambda x: datetime.utcnow() - x)
    test['Days on Market'] = (test['diff'] / np.timedelta64(1, 'D')).astype(int)

    #Need to create Days on Market for training data
    data['Listing Date'] = pd.to_datetime(data['Listing Date'])
    data['Pending Date'] = pd.to_datetime(data['Pending Date'])
    data['Days on Market'] = data['Pending Date'] - data['Listing Date']
    data['Days on Market'] = (data['Days on Market'] / np.timedelta64(1, 'D')).astype(int)

    #Now we drop all the variables we don't need
    test.drop('temp', axis = 1, inplace = True)
    test.drop('diff', axis = 1, inplace = True)
    test.drop('Listing Date', axis = 1, inplace = True)
    # test.drop('Market Days', axis = 1, inplace = True)

    #Listing Date is a string we don't need anymore
    data.drop('Listing Date', axis = 1, inplace = True)
    data.drop('Pending Date', axis = 1, inplace = True)
    
    status('Market Day Diff')

get_marketDate()

#Input test data with today's date in sold column 
# and change type of data's sold date from string to datetime
def create_dates():
    global data
    global test


    #Want to feature seasons multiplied with years sold
    #Don't need this anymore

    data['featuredDate'] = data['Season']
    test['featuredDate'] = test['Season']

    season_dict = {
                    "Winter":   "1",
                    "Spring":   "2",
                    "Summer":   "3", 
                    "Fall":     "4"
                    }

    #Map this onto the featuredData column
    data["featuredDate"] = data.featuredDate.map(season_dict).astype(int)
    test['featuredDate'] = test.featuredDate.map(season_dict).astype(int)

    # Multiply intYear by featuredDate to get new category
    data['featuredDate'] = data['featuredDate'] * data['intYear']
    test['featuredDate'] = test['featuredDate'] * test['intYear'] 
    
    #We don't need seasons anymore
    data.drop('Season', axis = 1, inplace = True)
    test.drop('Season', axis = 1, inplace = True)
    #And don't need intYear anymore ***Actually let's keep this in and see what's up***
    # data.drop('intYear', axis = 1, inplace = True)
    # test.drop('intYear', axis = 1, inplace = True)


    status('All Dates')

create_dates()

#Need to scale all the features so they are normalized
def scale_all_features():
    
    global data
    global test
    
    features = list(data.columns)
    data[features] = data[features].apply(lambda x: (x - x.mean()) / x.std(), axis = 0)
    
    features_test= list(test.columns)
    test[features_test] = test[features_test].apply(lambda x: (x - x.mean()) / x.std(), axis = 0)

    print 'Features scaled successfully !'

scale_all_features()

def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)
    return np.mean(xval)

def recover_train_test_target():
    global data
    global test
    
    train0 = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/sold3.csv')
    
    targets = train0['Selling Price']
    train = data
    
    return train,test,targets


train, test, targets = recover_train_test_target()

clf = ExtraTreesRegressor(n_estimators = 500)
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_

print features.sort_values(['importance'], ascending = False)

model = SelectFromModel(clf, prefit = True)
train_new = model.transform(train)
train_new.shape

test_new = model.transform(test)
test_new.shape

#Gradient Boosting 
params = {
         'n_estimators': 500, 
         'max_depth': 7, 
         'min_samples_split': 2,
         'learning_rate': 0.01,
         'loss': 'ls'
         }

gradBoost = ensemble.GradientBoostingRegressor(**params)
gradBoost.fit(train_new, targets)
Y_grad_pred = gradBoost.predict(test_new)


#Linear Regression
linReg = linear_model.LinearRegression()
linReg.fit(train_new, targets)
Y_lin_pred = linReg.predict(test_new)
print linReg.score(train_new, targets)

#K-nearest neighbours
knn = KNeighborsRegressor()
knn.fit(train_new, targets)
KNeighborsRegressor(algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
           metric_params = None, n_jobs = 1, n_neighbors = 7, p = 2,
           weights =  'uniform')
Y_prediction = knn.predict(test_new)
print knn.score(train_new, targets)


#hyperparameters tuning

forest = RandomForestRegressor(max_features = 'auto')

parameter_grid = {
                 'max_depth' : [5,6,7],
                 'n_estimators': [300,340,370,400]
                 }

# cross_validation = StratifiedKFold(targets, n_folds = 5)

grid_search = GridSearchCV(forest,
                           param_grid = parameter_grid,
                           cv = 5)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['Listing Price'] = origList_Price
df_output['Predicted Selling Price'] = output
df_output['Address'] = orig_address
df_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/rfPred.csv',index=False)

#For Gradient Boosting 
gradOut = Y_grad_pred
df_grad = pd.DataFrame()
df_grad['Listing Price'] = origList_Price
df_grad['Predicted Selling Price'] = gradOut
df_grad['Address'] = orig_address
df_grad[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/gradPred.csv',index=False)

#For Linear Regression
linOut = Y_lin_pred
dfx_output = pd.DataFrame()
dfx_output['Listing Price'] = origList_Price
dfx_output['Predicted Selling Price'] = linOut
dfx_output['Address'] = orig_address
dfx_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/linRegPred.csv',index=False)


#For K-Nearest Neighbors prediction
kOut = Y_prediction
df2_output = pd.DataFrame()
df2_output['Listing Price'] = origList_Price
df2_output['Predicted Selling Price'] = kOut
df2_output['Address'] = orig_address
df2_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/knnPred.csv',index=False)
