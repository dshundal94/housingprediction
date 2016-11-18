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




#Reading in the csv file, this is the training set
data = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/sold3.csv')
test = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/testing3.csv')

print data.info()
print test.info()

#extract and remove targets from training data 
targets = data['Selling Price']
origList_Price = test['Listing Price']
orig_address = test['Address']
data.drop('Selling Price', axis = 1, inplace = True)
test.drop('Selling Price', axis = 1, inplace = True)
data.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)
test.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)

#Drawing a linear model between square feet and lot size
# square_feet = data['Square Feet']
# lot_size = data['Lot Size(AC)']
# fit = np.polyfit(square_feet, lot_size, deg = 1)
# fig, ax = plt.subplots()
# ax.plot(square_feet, fit[0] * square_feet + fit[1], color = 'red')
# ax.scatter(square_feet, lot_size)
# ax.set_xlabel('Square Feet')
# ax.set_ylabel('Lot Size(AC)')
# ax.set_title('Square Feet Vs Lot Size(Acres)')
# plt.show()

# print test.info()
# print 'first'

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


#get the zip code from the address --don't need to extract zip code anymore
# def zip_code():
#     data['Zip Code'] = data["Address"].map(lambda address: address.split('CA')[1].split('-')[0].strip())
#     test['Zip Code'] = test['Address'].map(lambda address: address.split('CA')[1].split('-')[0].strip())

# zip_code()
# # print test.info()
# # print 'second'

# grouped = data.groupby('Zip Code')
# groupedTest = test.groupby('Zip Code')
# grouped.median()
# groupedTest.median()

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

# print test.info()
# print 'third'

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
# print test.info()
# print 'fourth'

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

# print test.info()
# print 'fifth'

#Drop variables we don't need

def drop_strings():

    global data
    global test

    # data.drop('Pending Date', axis = 1, inplace = True)
    data.drop('Selling Date', axis = 1, inplace = True)
    # data.drop('Type', axis = 1, inplace = True)

    #The test data does not have a pending date or a sold date
    # test.drop('Type', axis = 1, inplace = True)
    test.drop('Pending Date', axis = 1, inplace = True)
    test.drop('Selling Date', axis = 1, inplace = True)

drop_strings()
# print test.info()
# print 'sixth'

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

# print test.info()
# print 'seventh'

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

    #Now we encode dummy variables for the years ****Don't need dummy variables anymore, since data has more samples, then just 2015, 2016***
    # year_dummies = pd.get_dummies(data['Year'], prefix = 'Year')
    # data = pd.concat([data, year_dummies], axis = 1)

    # year_test_dummies = pd.get_dummies(test['Year'], prefix = 'Year')
    # test = pd.concat([test, year_test_dummies], axis = 1)

    #Create column with years in integer before dropping the string version
    data['intYear'] = data.Year.astype(int)
    test['intYear'] = test.Year.astype(int)

    #Now we drop all the variables that we don't need
    data.drop('Year', axis = 1, inplace = True)
    # data.drop('Season', axis = 1, inplace = True)
    data.drop('Month', axis = 1, inplace = True)

    test.drop('Year', axis = 1, inplace = True)
    # test.drop('Season', axis = 1, inplace = True)
    test.drop('Month', axis = 1, inplace = True)

    status('Dates')

process_dates()
# print test.info()
# print 'EIGHT'

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


    # #create a list to store the difference in times
    # temp = list()

    #Need to convert listing date string to date format
    #Didn't work because hard to convert to panda series after
    # for i in range(len(data)):
    #     formatDate = data['Listing Date']
    #     dateList = datetime.strptime(formatDate[i], '%m/%d/%Y')
    #     marketDate = datetime.utcnow() - dateList
    #     data['Days on Market'] = marketDate.days
    #     print dayDiff
    # data['Days on Market'] = pd.Series(marketDate.days)
    # print data['Days on Market']
    
    status('Market Day Diff')

get_marketDate()
# print test.info()
# print 'Nine'

#Input test data with today's date in sold column 
# and change type of data's sold date from string to datetime
def create_dates():
    global data
    global test

    #These are in datetime64[ns] type format, instead of object
    # test['Pending Date'] = test['Pending Date'].fillna(date.today())
    # test['Pending Date'] = pd.to_datetime(test['Pending Date'])
    # data['Pending Date'] = pd.to_datetime(data['Pending Date'])

    #Want to feature seasons multiplied with years sold

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

#I want to be able to create learning methods for time series data



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
print data.info()
print test.info()

# def drop_least_important():

#     global data
#     global test

#     data.drop('Zip Code_95240', axis = 1, inplace = True)
#     data.drop('Season_Spring', axis = 1, inplace = True)
#     data.drop('Season_Winter', axis = 1, inplace = True)
#     data.drop('Season_Summer', axis = 1, inplace = True)
#     data.drop('Season_Fall', axis = 1, inplace = True)
#     data.drop('Zip Code_95242', axis = 1, inplace = True)
#     data.drop('Year_2015', axis = 1, inplace = True)
#     data.drop('Year_2016', axis = 1, inplace = True)

#     test.drop('Zip Code_95240', axis = 1, inplace = True)
#     test.drop('Season_Spring', axis = 1, inplace = True)
#     test.drop('Season_Winter', axis = 1, inplace = True)
#     test.drop('Season_Summer', axis = 1, inplace = True)
#     test.drop('Season_Fall', axis = 1, inplace = True)
#     test.drop('Zip Code_95242', axis = 1, inplace = True)
#     test.drop('Year_2015', axis = 1, inplace = True)
#     test.drop('Year_2016', axis = 1, inplace = True)

#     status('Drop Least Important')

# drop_least_important()

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


train,test,targets = recover_train_test_target()
clf = ExtraTreesClassifier(n_estimators = 500)
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_

print features.sort_values(['importance'], ascending = False)

model = SelectFromModel(clf, prefit = True)
train_new = model.transform(train)
train_new.shape

print data.info()
print test.info()

test_new = model.transform(test)
test_new.shape

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(train, targets)
Y_pred = logreg.predict(test)
print logreg.score(train, targets)

#K-nearest neighbours
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_new, targets)
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