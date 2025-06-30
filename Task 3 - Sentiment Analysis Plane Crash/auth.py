import os
import tweepy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get keys from .env
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")

# Authenticate with Twitter
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Test authentication
try:
    api.verify_credentials()
    print("✅ Authentication successful!")
except Exception as e:
    print("❌ Authentication failed:", e)
