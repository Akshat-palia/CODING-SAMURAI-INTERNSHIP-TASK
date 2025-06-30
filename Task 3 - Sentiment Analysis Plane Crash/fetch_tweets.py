import os
from dotenv import load_dotenv
import tweepy

# Load credentials
load_dotenv()

# Create a Tweepy client (v2)
client = tweepy.Client(
    bearer_token=os.getenv("BEARER_TOKEN")
)

# Search tweets (Ahmedabad plane crash)
query = "Ahmedabad plane crash -is:retweet lang:en"
tweet_limit = 50

# Fetch tweets using Twitter API v2
tweets = client.search_recent_tweets(query=query, max_results=50)

# Save tweet texts
tweet_texts = [tweet.text for tweet in tweets.data] if tweets.data else []

# Print and save
for i, tweet in enumerate(tweet_texts, 1):
    print(f"\nTweet {i}: {tweet}")

with open("tweets.txt", "w", encoding="utf-8") as f:
    for tweet in tweet_texts:
        f.write(tweet + "\n\n")

print(f"\n✅ {len(tweet_texts)} tweets saved to tweets.txt")
