#import liraries
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#create new variables for data
dates = []
prices = []

#import data into your fuction to fill the above variables
#filename is the name of our stock prices data
def get_data(filename):
    #the code below will open our file and read it as csv
    with open(filename, 'r') as csvfile:
        #create a file reader in order to read the file
        csvFileReader = csv.reader(csvfile)
        #this will make us iterate over our csv file and return a string for the next line using the next method
        next(csvFileReader)
        for row in csvFileReader:
            #we are appending the date column into and integer and using the split function to remove the dashes from the dates
            dates.append(int(row[0].split('-')[0]))
            #we append the price column by changing the data type to float
            prices.append(float(row[1])
    return

#creat a predict function and use numpy to format our list into an nx1 matrix
def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1))

    #create models
    svr_lin = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel= 'poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma=0.1)

    #train models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    #plot graph
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='yellow', label='Linear Model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial Model')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    return svr_lin.predict(x)(0), svr_poly.predict(x)(0), svr_rbf.predict(x)(0)

#import Data
get_data('AAPL.csv')

#create a variable to store predicted price
predictedPrice = predict_prices(dates, prices, 29)

#display our results
print(predictedPrice)