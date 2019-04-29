#%%
#import library
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetching and formating datasets
data = fetch_movielens(min_rating=4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#creating our model
model = LightFM(loss='warp') 
#warp = Weight Approximate-Rank Pairwise (helps create recommendations for each user by looking at the existing user ratings and prediciting rankings for each. It also uses the Gradient Descent Algorithm to iteratively find the weights that improve our prediction overtime)

#train our model
model.fit(data['train'], epochs=30, num_threads=2)

#generating a recommendation from our model
def recommended(model, data, user_ids):
    #get no. of users and movies in training data using the shape attribute
    n_users, n_items = data['train'].shape

    #use a for loop to go through every user id we will input and generate known positves for each
    for user_id in user_ids:

        #movies the user likes
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #generate recommendations based on our model
        scores = model.predict(user_id, np.arange(n_items))
        #rank in descending order based on the most liked
        top_items = data['item_labels'][np.argsort(-scores)]

        #print results
        print("User %s" %user_id)
        print("     Known positives:")

        #top 3 movies the user has picked
        for x in known_positives[:3]:
            print("        %s" %x)

        print("    Recommended:")

        #print the top 3 that the model predicts
        for x in top_items[:3]:
            print("          %s" %x)

#call the model, data and the user id
recommended(model, data, [3, 25, 450])

    

