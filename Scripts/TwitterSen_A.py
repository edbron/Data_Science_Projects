#import library
import tweepy
from textblob import TextBlob

#access tokens from twitter
consumer_key = ''
consumer_secret = ''

accessToken = ''
accessToken_secret = ''

#creating our auth variable
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(accessToken, accessToken_secret)

#assign our API
api = tweepy.API(auth)

#serching tweets
public_tweets = api.search('Got')

#performing the analysis
for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)