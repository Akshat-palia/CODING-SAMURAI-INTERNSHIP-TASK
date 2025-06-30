from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

# Read tweets from file
with open("tweets.txt", "r", encoding="utf-8") as f:
    tweets = f.read().strip().split("\n\n")

# Initialize counters
positive, neutral, negative = 0, 0, 0

# Process each tweet
for tweet in tweets:
    score = analyzer.polarity_scores(tweet)["compound"]
    
    if score >= 0.05:
        positive += 1
    elif score <= -0.05:
        negative += 1
    else:
        neutral += 1

# Print result
print("📊 Sentiment Summary:")
print(f"✅ Positive: {positive}")
print(f"😐 Neutral : {neutral}")
print(f"❌ Negative: {negative}")
