# ***This script creates a model to predict housing prices in Lodi, CA with
# zipcodes 95240 and 95242. In the future, if model is accurate based on mean squared error
# then will generalize to include many more zip codes. I find that Random Forest does not work
# quite well to model the prediction, and my best bet would be to fine tune regression paramaters
# or try using Ridge Regression to deal with multicollinearity. **** --By Damanjit Hundal 



import pandas as pd
import csv as csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import statsmodels.formula.api as sm
from datetime import datetime
from datetime import date
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model


#Reading in the csv file, this is the training set
data = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/sold3.csv')
test = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/testing3.csv')

#extract and remove targets from training data 
origList_Price = test['Listing Price']
train = data.loc[0:3600, :]
cv = data.loc[3601:, :]
orig_cv_price = cv['Listing Price']
targets = train['Selling Price']
test_check = cv['Selling Price']
orig_check_address = cv['Address']
orig_address = test['Address']
train.drop('Selling Price', axis = 1, inplace = True)
cv.drop('Selling Price', axis = 1, inplace = True)
test.drop('Selling Price', axis = 1, inplace = True)
train.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)
cv.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)
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
    train['Pool'] = train.Pool.map(Pool_dict).astype(int)
    cv['Pool'] = cv.Pool.map(Pool_dict).astype(int)
    test['Pool'] = test.Pool.map(Pool_dict).astype(int)

get_pool()

#Combine Half bathroom with full bathrooms
def get_bath():
    train['Bathrooms - Half'] = train['Bathrooms - Half'] / 2
    train['Total Bathrooms'] = train['Bathrooms - Full'] + train['Bathrooms - Half']

    cv['Bathrooms - Half'] = cv['Bathrooms - Half'] / 2
    cv['Total Bathrooms'] = cv['Bathrooms - Full'] + cv['Bathrooms - Half']

    test['Bathrooms - Half'] = test['Bathrooms - Half'] / 2
    test['Total Bathrooms'] = test['Bathrooms - Full'] + test['Bathrooms - Half']

    #Drop Half and Full bathrooms, because feature redundant
    train.drop('Bathrooms - Full', axis = 1, inplace = True)
    train.drop('Bathrooms - Half', axis = 1, inplace = True)

    cv.drop('Bathrooms - Full', axis = 1, inplace = True)
    cv.drop('Bathrooms - Half', axis = 1, inplace = True)

    test.drop('Bathrooms - Full', axis = 1, inplace = True)
    test.drop('Bathrooms - Half', axis = 1, inplace = True)

get_bath()

#Hardcoding the lot size based off median of zipcode. will generalize later to include all zip codes. 
def process_lotsize():
    global train
    global cv
    global test

    def fillLotSize(row):
        if row['Address - Zip Code'] == '95240':
            return 5756
        elif row['Address - Zip Code'] == '95242':
            return 6456
    train['Lot Size - Sq Ft'] = train.apply(lambda r: fillLotSize(r) if np.isnan(r['Lot Size - Sq Ft']) else r['Lot Size - Sq Ft'], axis = 1)
    cv['Lot Size - Sq Ft'] = cv.apply(lambda r: fillLotSize(r) if np.isnan(r['Lot Size - Sq Ft']) else r['Lot Size - Sq Ft'], axis = 1)
    test['Lot Size - Sq Ft'] = test.apply(lambda r: fillLotSize(r) if np.isnan(r['Lot Size - Sq Ft']) else r['Lot Size - Sq Ft'], axis = 1)
    status('Lot Size - Sq Ft')
process_lotsize()



#Harcode the year built, will generalize for all zip codes later
def process_yearBuilt():
    global train
    global cv
    global test

    def fillYearBuilt(row):
        if row['Address - Zip Code'] == 95240:
            return 1963
        elif row['Address - Zip Code'] == 95242:
            return 1985
    train['Year Built'] = train.apply(lambda r: fillYearBuilt(r) if np.isnan(r['Year Built']) else r['Year Built'], axis = 1)
    cv['Year Built'] = cv.apply(lambda r: fillYearBuilt(r) if np.isnan(r['Year Built']) else r['Year Built'], axis = 1)
    test['Year Built'] = test.apply(lambda r: fillYearBuilt(r) if np.isnan(r['Year Built']) else r['Year Built'], axis = 1)
    status('Year Built')

process_yearBuilt()

#Use binary option to show zipcodes
def process_area():
    global train
    global cv
    global test

    #Clean the address variable
    train.drop('Address', axis = 1, inplace = True)
    cv.drop('Address', axis = 1, inplace = True)
    test.drop('Address', axis = 1, inplace = True)

    #encode dummy variables
    zipcode_dummies = pd.get_dummies(train['Address - Zip Code'], prefix = 'Zip Code')
    train = pd.concat([train, zipcode_dummies], axis = 1)

    zipcode_dummies = pd.get_dummies(cv['Address - Zip Code'], prefix = 'Zip Code')
    cv = pd.concat([cv, zipcode_dummies], axis = 1)

    zipcode_test_dummies = pd.get_dummies(test['Address - Zip Code'], prefix = 'Zip Code')
    test = pd.concat([test, zipcode_test_dummies], axis = 1)
    #remove the zip code title
    train.drop('Address - Zip Code', axis = 1, inplace = True)
    cv.drop('Address - Zip Code', axis = 1, inplace = True)
    test.drop('Address - Zip Code', axis = 1, inplace = True)

    status('Zip Code')

process_area()



#Drop variables we don't need
def drop_strings():

    global train
    global cv
    global test

    train.drop('Selling Date', axis = 1, inplace = True)
    cv.drop('Selling Date', axis = 1, inplace = True)

    #The test data does not have a pending date or a sold date
    test.drop('Pending Date', axis = 1, inplace = True)
    test.drop('Selling Date', axis = 1, inplace = True)

drop_strings()

#Split the listing date into months and years
def get_dates():

    global train
    global cv
    global test

    #get the months and years, not interested in the days
    train['Month'] = train['Listing Date'].map(lambda listdate: listdate.split('/')[0].strip())
    train['Year'] = train['Listing Date'].map(lambda listdate: listdate.split('/')[2].strip())

    cv['Month'] = cv['Listing Date'].map(lambda listdate: listdate.split('/')[0].strip())
    cv['Year'] = cv['Listing Date'].map(lambda listdate: listdate.split('/')[2].strip())

    #Now for the test data
    test['Month'] = test['Listing Date'].map(lambda listdate: listdate.split('/')[0].strip())
    test['Year'] = test['Listing Date'].map(lambda listdate: listdate.split('/')[2].strip())


    status("Get Dates")

get_dates()

#Now we're going to process our dates
def process_dates():

    global train
    global cv
    global test

    train['Season'] = train['Month']
    cv['Season'] = cv['Month']
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
    train['Season'] = train.Season.map(Date_Dictionary)
    cv['Season'] = cv.Season.map(Date_Dictionary)
    test['Season'] = test.Season.map(Date_Dictionary)

    #Encode dummy variables for the seasons
    season_dummies = pd.get_dummies(train['Season'], prefix = 'Season')
    train = pd.concat([train, season_dummies], axis =1)

    season_dummies = pd.get_dummies(cv['Season'], prefix = 'Season')
    cv = pd.concat([cv, season_dummies], axis =1)

    season_test_dummies = pd.get_dummies(test['Season'], prefix = 'Season')
    test = pd.concat([test, season_test_dummies], axis =1)

    #Create column with years in integer before dropping the string version
    train['intYear'] = train.Year.astype(int)
    cv['intYear'] = cv.Year.astype(int)
    test['intYear'] = test.Year.astype(int)

    #Now we drop all the variables that we don't need
    train.drop('Year', axis = 1, inplace = True)
    train.drop('Month', axis = 1, inplace = True)
    
    cv.drop('Year', axis = 1, inplace = True)
    cv.drop('Month', axis = 1, inplace = True)

    test.drop('Year', axis = 1, inplace = True)
    test.drop('Month', axis = 1, inplace = True)

    status('Dates')

process_dates()


#Need to work on creating days on market column for test data
def get_marketDate():

    global train
    global cv
    global data

    #Can use datetime in panda series
    test['temp'] = test['Listing Date']
    test['temp'] = pd.to_datetime(test['temp'])
    test['diff'] = test['temp'].map(lambda x: datetime.utcnow() - x)
    test['Days on Market'] = (test['diff'] / np.timedelta64(1, 'D')).astype(int)

    #Need to create Days on Market for training data
    train['Listing Date'] = pd.to_datetime(train['Listing Date'])
    train['Pending Date'] = pd.to_datetime(train['Pending Date'])
    train['Days on Market'] = train['Pending Date'] - train['Listing Date']
    train['Days on Market'] = (train['Days on Market'] / np.timedelta64(1, 'D')).astype(int)

    cv['Listing Date'] = pd.to_datetime(cv['Listing Date'])
    cv['Pending Date'] = pd.to_datetime(cv['Pending Date'])
    cv['Days on Market'] = cv['Pending Date'] - cv['Listing Date']
    cv['Days on Market'] = (cv['Days on Market'] / np.timedelta64(1, 'D')).astype(int)

    #Now we drop all the variables we don't need
    test.drop('temp', axis = 1, inplace = True)
    test.drop('diff', axis = 1, inplace = True)
    test.drop('Listing Date', axis = 1, inplace = True)
    # test.drop('Market Days', axis = 1, inplace = True)

    #Listing Date is a string we don't need anymore
    train.drop('Listing Date', axis = 1, inplace = True)
    train.drop('Pending Date', axis = 1, inplace = True)

    cv.drop('Listing Date', axis = 1, inplace = True)
    cv.drop('Pending Date', axis = 1, inplace = True)
    
    status('Market Day Diff')

get_marketDate()

#Input test data with today's date in sold column 
# and change type of data's sold date from string to datetime
def create_dates():
    global train
    global cv
    global test


    #Want to feature seasons multiplied with years sold
    #Don't need this anymore

    # data['featuredDate'] = data['Season']
    # test['featuredDate'] = test['Season']

    # season_dict = {
    #                 "Winter":   "1",
    #                 "Spring":   "2",
    #                 "Summer":   "3", 
    #                 "Fall":     "4"
    #                 }

    # #Map this onto the featuredData column
    # data["featuredDate"] = data.featuredDate.map(season_dict).astype(int)
    # test['featuredDate'] = test.featuredDate.map(season_dict).astype(int)

    # # Multiply intYear by featuredDate to get new category
    # data['featuredDate'] = data['featuredDate'] * data['intYear']
    # test['featuredDate'] = test['featuredDate'] * test['intYear'] 
    
    #We don't need seasons anymore
    train.drop('Season', axis = 1, inplace = True)
    cv.drop('Season', axis = 1, inplace = True)
    test.drop('Season', axis = 1, inplace = True)
    #And don't need intYear anymore ***Actually let's keep this in and see what's up***
    # data.drop('intYear', axis = 1, inplace = True)
    # test.drop('intYear', axis = 1, inplace = True)


    status('All Dates')

create_dates()

#Need to scale all the features so they are normalized
def scale_all_features():
    
    global train
    global cv
    global test
    
    features = list(train.columns)
    train[features] = train[features].apply(lambda x: (x - x.mean()) / x.std(), axis = 0)

    features_cv = list(cv.columns)
    cv[features] = cv[features].apply(lambda x: (x - x.mean()) / x.std(), axis = 0)
    
    features_test= list(test.columns)
    test[features_test] = test[features_test].apply(lambda x: (x - x.mean()) / x.std(), axis = 0)

    print 'Features scaled successfully !'

scale_all_features()

def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 10,scoring=scoring)
    return np.mean(xval)

# def recover_train_test_target():
#     global data
#     global test
    
#     train0 = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/sold3.csv')
    
#     targets = train0['Selling Price']
#     train = data
    
#     return train,test,targets


# train, test, targets = recover_train_test_target()
clf = ExtraTreesClassifier(n_estimators = 150)
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_

print features.sort_values(['importance'], ascending = False)

model = SelectFromModel(clf, prefit = True)
train_new = model.transform(train)
train_new.shape

cv_new = model.transform(cv)
cv_new.shape


test_new = model.transform(test)
test_new.shape

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(train_new, targets)
Y_pred = logreg.predict(test_new)
print logreg.score(train_new, targets)

#Linear Regression (Multivariate)
lin = linear_model.LinearRegression()
lin.fit(train_new, targets)
Y_lin = lin.predict(cv_new)
print lin.score(train_new, targets)

#K-nearest neighbours
knn = KNeighborsClassifier()
knn.fit(train_new, targets)
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric='minkowski',
           metric_params = None, n_jobs = 1, n_neighbors = 5, p = 2,
           weights =  'uniform')
Y_prediction = knn.predict(test_new)
print knn.score(train_new, targets)

#Gaussian Naive Bayes, just here so I know how to implement, mainly used for classification problems
gaussian = GaussianNB()
gaussian.fit(train_new, targets)
Y_prediction1 = gaussian.predict(test_new)
print gaussian.score(train_new, targets)

#hyperparameters tuning

forest = RandomForestClassifier(max_features='auto')

# parameter_grid = {
#                  'max_depth' : [4,5,6,7,8],
#                  'n_estimators': [200,210,240,250],
#                  'criterion': ['gini','entropy']
#                  }

# cross_validation = StratifiedKFold(targets, n_folds=5)

# grid_search = GridSearchCV(forest,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)

# grid_search.fit(train_new, targets)

# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))
forest.fit(train_new, targets)
output = forest.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['Listing Price'] = origList_Price
df_output['Predicted Selling Price'] = output
df_output['Address'] = orig_address
df_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/rfPred.csv',index=False)

#For logistic regression output
logOut = Y_pred
df1_output = pd.DataFrame()
df1_output['Listing Price'] = origList_Price
df1_output['Predicted Selling Price'] = logOut
df1_output['Address'] = orig_address
df1_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/logisticPred.csv',index=False)

linOut = Y_lin
dfx_output = pd.DataFrame()
dfx_output['Listing Price'] = orig_cv_price
dfx_output['Predicted Selling Price'] = linOut
dfx_output['Address'] = orig_address
dfx_output['Actual Selling Price'] = test_check
dfx_output['Mean Squared Error'] = ((dfx_output['Actual Selling Price'] - dfx_output['Predicted Selling Price']) ** 2).mean(axis = 0)
dfx_output[['Address', 'Listing Price','Predicted Selling Price', 'Actual Selling Price', "Mean Squared Error"]].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/linRegPred.csv',index=False)

#For K-Nearest Neighbors prediction
kOut = Y_prediction
df2_output = pd.DataFrame()
df2_output['Listing Price'] = origList_Price
df2_output['Predicted Selling Price'] = kOut
df2_output['Address'] = orig_address
df2_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/knnPred.csv',index=False)

#And just for fun we will do Naive Bayes

nbOut = Y_pred
df3_output = pd.DataFrame()
df3_output['Listing Price'] = origList_Price
df3_output['Predicted Selling Price'] = nbOut
df3_output['Address'] = orig_address
df3_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/nbPred.csv',index=False)