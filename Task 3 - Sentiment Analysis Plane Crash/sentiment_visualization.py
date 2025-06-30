from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Read tweets
with open("tweets.txt", "r", encoding="utf-8") as f:
    tweets = f.read().strip().split("\n\n")

# Analyze sentiments
analyzer = SentimentIntensityAnalyzer()
positive = neutral = negative = 0

for tweet in tweets:
    score = analyzer.polarity_scores(tweet)["compound"]
    if score >= 0.05:
        positive += 1
    elif score <= -0.05:
        negative += 1
    else:
        neutral += 1

# Data
labels = ['Positive', 'Neutral', 'Negative']
counts = [positive, neutral, negative]
colors = ['#66bb6a', '#ffee58', '#ef5350']

# Create Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Tweet Sentiment Distribution - Ahmedabad Plane Crash')
plt.axis('equal')
plt.tight_layout()
plt.savefig("sentiment_pie_chart.png")
plt.show()

# Create Bar Chart
plt.figure(figsize=(6, 4))
plt.bar(labels, counts, color=colors)
plt.title('Tweet Sentiment Count - Ahmedabad Plane Crash')
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("sentiment_bar_chart.png")
plt.show()
