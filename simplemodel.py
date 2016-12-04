import pandas as pd
import csv as csv
import numpy as np
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from datetime import datetime
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import transforms
import statsmodels.api as sm
import statsmodels.formula.api as smf


data = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/sold3.csv')
prices = data['Selling Price']
predictions = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/testing3.csv')

#Plot to see relationship between the listed price and the final sale price
x_axis = data['Listing Price']
y_axis = data['Selling Price']
pl.figure()
pl.title('Training Data: Listing Price vs. Selling Price')
m, b = np.polyfit(x_axis, y_axis, 1)
plt.plot(x_axis, y_axis, '.')
plt.plot(x_axis, m*x_axis + b, '-')
pl.legend()
pl.xlabel('Listing Price')
pl.ylabel('Selling Price')
pl.show()
results = sm.OLS(y_axis,sm.add_constant(x_axis)).fit()
print results.summary()
print "y=%.6fx+(%.6f)"%(m,b)

predictions.drop('Selling Price', axis = 1, inplace = True)
predictions.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)
origList_Price = predictions['Listing Price']
orig_address = predictions['Address']

def performance_metric(y_true, y_predict):
    # Calculates and returns the performance score between 
    # true and predicted values based on the metric chosen.
    
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score
data.drop('Selling_Price_Per_Sqft', axis = 1, inplace = True)

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

get_pool()

#Combine Half bathroom with full bathrooms
def get_bath():
    data['Bathrooms - Half'] = data['Bathrooms - Half'] / 2
    data['Total Bathrooms'] = data['Bathrooms - Full'] + data['Bathrooms - Half']

    predictions['Bathrooms - Half'] = predictions['Bathrooms - Half'] / 2
    predictions['Total Bathrooms'] = predictions['Bathrooms - Full'] + predictions['Bathrooms - Half']

    #Drop Half and Full bathrooms, because feature redundant
    data.drop('Bathrooms - Full', axis = 1, inplace = True)
    data.drop('Bathrooms - Half', axis = 1, inplace = True)

    predictions.drop('Bathrooms - Full', axis = 1, inplace = True)
    predictions.drop('Bathrooms - Half', axis = 1, inplace = True)

get_bath()

#Hardcoding the lot size based off median of zipcode. will generalize later to include all zip codes. 
def process_lotsize():
    

    def fillLotSize(row):
        if row['Address - Zip Code'] == '95240':
            return 5756
        elif row['Address - Zip Code'] == '95242':
            return 6456
    data['Lot Size - Sq Ft'] = data.apply(lambda r: fillLotSize(r) if np.isnan(r['Lot Size - Sq Ft']) else r['Lot Size - Sq Ft'], axis = 1)
    predictions['Lot Size - Sq Ft'] = predictions.apply(lambda r: fillLotSize(r) if np.isnan(r['Lot Size - Sq Ft']) else r['Lot Size - Sq Ft'], axis = 1)
    status('Lot Size - Sq Ft')
process_lotsize()



#Harcode the year built, will generalize for all zip codes later
def process_yearBuilt():

    def fillYearBuilt(row):
        if row['Address - Zip Code'] == 95240:
            return 1963
        elif row['Address - Zip Code'] == 95242:
            return 1985
    data['Year Built'] = data.apply(lambda r: fillYearBuilt(r) if np.isnan(r['Year Built']) else r['Year Built'], axis = 1)
    predictions['Year Built'] = predictions.apply(lambda r: fillYearBuilt(r) if np.isnan(r['Year Built']) else r['Year Built'], axis = 1)
    status('Year Built')

process_yearBuilt()

#Use binary option to show zipcodes
def process_area():
    global data
    global predictions

    #Clean the address variable
    data.drop('Address', axis = 1, inplace = True)
    predictions.drop('Address', axis = 1, inplace = True)

    # #encode dummy variables
    zipcode_dummies = pd.get_dummies(data['Address - Zip Code'], prefix = 'Zip Code')
    data = pd.concat([data, zipcode_dummies], axis = 1)

    zipcode_test_dummies = pd.get_dummies(predictions['Address - Zip Code'], prefix = 'Zip Code')
    predictions = pd.concat([predictions, zipcode_test_dummies], axis = 1)
    
    #remove the zip code title
    data.drop('Address - Zip Code', axis = 1, inplace = True)
    predictions.drop('Address - Zip Code', axis = 1, inplace = True) 
    status('Zip Code')

process_area()

#Drop variables we don't need
def drop_strings():

    data.drop('Selling Date', axis = 1, inplace = True)

    predictions.drop('Pending Date', axis = 1, inplace = True)
    predictions.drop('Selling Date', axis = 1, inplace = True)

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
    data['intYear'] = data.Year.astype(int)
    predictions['intYear'] = predictions.Year.astype(int)

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

    predictions.drop('temp', axis = 1, inplace = True)
    predictions.drop('diff', axis = 1, inplace = True)
    predictions.drop('Listing Date', axis = 1, inplace = True)
    
    status('Market Day Diff')

get_marketDate()


features = data.drop('Selling Price', axis = 1, inplace = True)

X = data
y = prices
# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=50)

def fit_model(X, y):
    # """ Performs grid search over the 'max_depth' parameter for a 
    #     decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    #Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    #Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {"max_depth":range(1,10)}

    #Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    #Create the grid search object
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']) 

output = reg.predict(predictions)   
df_output = pd.DataFrame()
df_output['Listing Price'] = origList_Price
df_output['Predicted Selling Price'] = output
df_output['Address'] = orig_address
df_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/newPredictions.csv',index=False)


#Gradient Boosting 
params = {
         'n_estimators': 500, 
         'max_depth': 7, 
         'min_samples_split': 2,
         'learning_rate': 0.01,
         'loss': 'ls'
         }

gradBoost = ensemble.GradientBoostingRegressor(**params)
gradBoost.fit(X_train, y_train)
Y_grad_pred = gradBoost.predict(predictions)
features = pd.DataFrame()
features['feature'] = data.columns
features['importance'] = gradBoost.feature_importances_
print features.sort_values(['importance'], ascending = False)
    
Y_grad_pred = gradBoost.predict(predictions)
gradOut = Y_grad_pred
df_grad = pd.DataFrame()
df_grad['Listing Price'] = origList_Price
df_grad['Predicted Selling Price'] = gradOut
df_grad['Address'] = orig_address
df_grad[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/newgradPred.csv',index=False)

linReg = linear_model.LinearRegression()
linReg.fit(X_train, y_train)
Y_lin_pred = linReg.predict(predictions)
print linReg.score(X_train, y_train)
print('Coefficients: \n', linReg.coef_)

linOut = Y_lin_pred
dfx_output = pd.DataFrame()
dfx_output['Listing Price'] = origList_Price
dfx_output['Predicted Selling Price'] = linOut
dfx_output['Address'] = orig_address
dfx_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/newlinRegPred.csv',index=False)

#Neural Network

data_matrix = X_train.as_matrix()
targets_matrix = y_train.as_matrix()
test_matrix = predictions.as_matrix()

model1 = Sequential()
model1.add(Dense(14, input_dim=14, init='normal', activation='relu'))
model1.add(Dense(7, init='normal', activation='relu'))
model1.add(Dense(1, init='normal'))
# Compile model
model1.compile(loss='mean_squared_error', optimizer='adam')
larger_hist = model1.fit(data_matrix, targets_matrix, batch_size = 10, nb_epoch = 100)
larger_pred = model1.predict(test_matrix)

large_out = larger_pred
df_large = pd.DataFrame()
df_large['Listing Price'] = origList_Price
df_large['Predicted Selling Price'] = large_out
df_large['Address'] = orig_address
df_large[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/newlargePred.csv',index=False)

def fit_rf_model(X,y):
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    forest = RandomForestRegressor(max_features = 'auto')

    params = {
                     'max_depth' : [7],
                     'n_estimators': [350]
                     }

    scoring_fnc = make_scorer(performance_metric)

    grid_search = GridSearchCV(forest, params, scoring_fnc, cv = cv_sets)

    grid_search.fit(X, y)

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    return grid_search.best_estimator_

rf_reg = fit_rf_model(X_train, y_train)
output = rf_reg.predict(predictions)
df_output = pd.DataFrame()
df_output['Listing Price'] = origList_Price
df_output['Predicted Selling Price'] = output
df_output['Address'] = orig_address
df_output[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/newrfPred.csv',index=False)