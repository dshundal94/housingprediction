# housingprediction
Using Machine Learning Methods to Predict Housing Prices in Lodi
I'm looking at houses that were recently sold in the Lodi area (95240 and 95242) and using different machine learning algorithms to
predict housing prices on houses that are currently in the market. I'm a licensed real estate agent and have access to MetroList, so
that's where I'm getting my data from. So far, I've tried to predict prices using Random Forests, Decision Tree Regression,  Gradient Boosted Regression, Linear Regression,
and Neural Networks. Gradient Boosted Regression and Random Forest work the best with the training data, with neural networks and linear regression performing well, but not as
strongly as the others. 

I use pandas to read the data and notice that there are missing values, as well as data that needs to be converted from strings to some meaningful form in either integer or float.

data = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/sold3.csv')
predictions = pd.read_csv('C:/Users/Damanjit/Documents/HousingPrediction/testing3.csv')

The y value, or the value that we are trying to train is the selling price of the house. 

prices = data['Selling Price']

I plot the relationship between listing price and selling price to see if there is any correlation, and I notice that there is a very strong correlation with a R-Squared value of 0.988

I'm using a performance metric to train the data onto the test data, and then predicting house prices that are currently active on the market. 

The 14 Features that are being trained and looked at are the following: 
-Listing Price
-Days on Market 
-Price Per Square Feet
-Square Footage 
-Lot Size (Square Feet)
-Year Built 
-Year Being Sold
-# of Garage Spaces
-Bedrooms
-Home Association Dues
-Total Bathrooms
-Pool
-In Zip Code 95242?
-In Zip Code 95240?

The features are descending in the amount of importance, where Listing Price is the most important. 

Some of the values of Lot Size and Year Built are missing, so I looked at the median of the training data grouped by the Zip Code the house is located in. 

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

After cleaning all the data and inputting all the missing data, I can run some machine learning algorithms to predict housing prices. The example I'm going to show here is gradient boosting regression, 
since that performed very fast and was very good at predictions. 

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
df_grad[['Address', 'Listing Price','Predicted Selling Price']].to_csv('C:/Users/Damanjit/Documents/HousingPrediction/gradientBoostedPredictions.csv',index=False)

In these line of codes, I tune some of the parameters to give better predictions, and the metric I was using for performance was Mean- Squared Error. With these parameters, the mean-squared error was at a minimum 
the training data I was working with. I order the features in terms of its performance and then I export the predictions to an excel file where I can view the results. 

I also plotted the predicted selling price against the listing price and found again that the R-Squared value was 0.981 showing once again high correlation, as was seen with the training data. 

I want to include more zip codes and more training data to get better predictions in the future. I also want to make the program run faster, so I will check for optimization, because Random Forest is not working as fast as it could be. 
