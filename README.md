# housingprediction
Using Machine Learning Methods to Predict Housing Prices in Lodi
I'm looking at houses that were recently sold in the Lodi area (95240, 95242, 95209, and 95219 so far) and using different machine learning algorithms to predict housing prices on houses that are currently in the market. I'm a licensed real estate agent and have access to MetroList, so that's where I'm getting my data from. So far, I've tried to predict prices using Random Forests, Decision Tree Regression,  Gradient Boosted Regression, XGBoost Regression, Linear Regression, and Neural Networks. Gradient Boosted Regression and XGBoost work the best with the training data I have. 

I use pandas to read the data and notice that there are missing values, as well as data that needs to be converted from strings to some meaningful form in either integer or float.


The 15 Features that are being trained and looked at are the following: 
#
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
    -Bathroom-Full
    -Bathroom-Half
    -Pool
    -Zip Code 

Some of the values of Lot Size and Year Built are missing or are inputted as 0, so I looked at the median of the training data grouped by the Zip Code the house is located in. Most of the information of what I did to predict housing prices can be found in the Jupyter Notebook (housingpricepredictions.ipynb). 

Last Update: 12/15/2016

UPDATE: 1/10/2017
Included a general script with a larger dataset of about 75,000 training sample points. The general script imputes missing data based on the different zip codes, where 
now nothing is hard coded, but generalized to the dataset. 
