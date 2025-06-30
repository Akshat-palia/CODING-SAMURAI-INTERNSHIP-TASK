from textblob import TextBlob

# Read tweets from file
with open("tweets.txt", "r", encoding="utf-8") as f:
    tweets = f.read().strip().split("\n\n")

# Analyze each tweet
results = []

for i, tweet in enumerate(tweets, start=1):
    blob = TextBlob(tweet)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Label sentiment
    if polarity > 0:
        label = "Positive"
    elif polarity < 0:
        label = "Negative"
    else:
        label = "Neutral"

    results.append(f"{i}. [{label}] Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}\n{tweet}\n\n")

# Save results
with open("sentiment_results_textblob.txt", "w", encoding="utf-8") as f:
    f.writelines(results)

print("✅ TextBlob sentiment analysis complete! Results saved to sentiment_results_textblob.txt")
