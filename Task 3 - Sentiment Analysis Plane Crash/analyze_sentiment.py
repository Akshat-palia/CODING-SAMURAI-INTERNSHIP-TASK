from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Read tweets from file
with open("tweets.txt", "r", encoding="utf-8") as f:
    tweets = f.read().strip().split("\n\n")

# Analyze each tweet
results = []

for i, tweet in enumerate(tweets, start=1):
    sentiment = analyzer.polarity_scores(tweet)
    compound = sentiment["compound"]
    
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    results.append(f"{i}. [{label}] {tweet}\n")

# Save results to new file
with open("sentiment_results.txt", "w", encoding="utf-8") as f:
    f.writelines(results)

print("✅ Sentiment analysis complete! Results saved to sentiment_results.txt")
