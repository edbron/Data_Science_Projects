#import library
from sklearn import tree

#Classification model from scikit learn 
#a decision tree to be presice 
#because we want to classify into male of female

#Defining our Data(List)
#[shoe_size, height, dress_size]
X = [[33, 157, 8], [37, 170, 10], [40, 181, 14], [36, 176, 8], [43, 179, 12], [44, 167, 14], [41, 185, 14], [37, 161, 10], [41, 166, 12], [32, 150, 6], [42, 178, 10] ]

Y = ['female', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'female']

#variable to store our model 
G_clf = tree.DecisionTreeClassifier()

#training model with our X and Y data
G_clf = G_clf.fit(X,Y)

#test our model with data
prediction = G_clf.predict([[41, 160, 14]])

#display results
print (prediction)