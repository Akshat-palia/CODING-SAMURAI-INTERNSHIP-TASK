from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import datetime
from reportlab.platypus import Image

# Count tweets from previous step
positive = 0
neutral = 0
negative = 0

with open("tweets.txt", "r", encoding="utf-8") as f:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    tweets = f.read().strip().split("\n\n")
    for tweet in tweets:
        score = analyzer.polarity_scores(tweet)['compound']
        if score >= 0.05:
            positive += 1
        elif score <= -0.05:
            negative += 1
        else:
            neutral += 1

# Create PDF
filename = "sentiment_report.pdf"
pdf = canvas.Canvas(filename, pagesize=A4)
width, height = A4

# Title
pdf.setFont("Helvetica-Bold", 22)
pdf.drawCentredString(width / 2, height - 60, "Sentiment Analysis Report")

# Subtitle
pdf.setFont("Helvetica", 14)
pdf.drawCentredString(width / 2, height - 90, "Topic: Public Sentiment on Ahmedabad Plane Crash")

# Author and Date
pdf.setFont("Helvetica", 11)
pdf.drawString(40, height - 120, f"Prepared by: Akshat Palia")
pdf.drawString(40, height - 135, f"Date: {datetime.today().strftime('%d %B %Y')}")

# Summary
pdf.setFont("Helvetica-Bold", 13)
pdf.drawString(40, height - 170, "📊 Sentiment Summary:")

pdf.setFont("Helvetica", 12)
pdf.drawString(60, height - 190, f"✅ Positive Tweets : {positive}")
pdf.drawString(60, height - 210, f"😐 Neutral Tweets  : {neutral}")
pdf.drawString(60, height - 230, f"❌ Negative Tweets : {negative}")

# Add Pie Chart
pdf.setFont("Helvetica-Bold", 13)
pdf.drawString(40, height - 270, "📈 Pie Chart:")
pdf.drawImage("sentiment_pie_chart.png", 60, height - 520, width=200, height=200)

# Add Bar Chart
pdf.setFont("Helvetica-Bold", 13)
pdf.drawString(300, height - 270, "📊 Bar Chart:")
pdf.drawImage("sentiment_bar_chart.png", 300, height - 520, width=200, height=200)

# Footer
pdf.setFont("Helvetica-Oblique", 9)
pdf.drawCentredString(width / 2, 30, "This report was auto-generated using Python")

# Save
pdf.save()
print(f"✅ Report generated: {filename}")
