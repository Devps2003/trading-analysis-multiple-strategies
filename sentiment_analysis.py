# sentiment_analysis.py

import requests
from textblob import TextBlob

def fetch_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=YOUR_NEWS_API_KEY"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        analysis = TextBlob(article['title'] + ' ' + article['description'])
        sentiments.append(analysis.sentiment.polarity)
    return sum(sentiments) / len(sentiments) if sentiments else 0

def integrate_sentiment(data, ticker):
    articles = fetch_news(ticker)
    sentiment_score = analyze_sentiment(articles)
    data['Sentiment'] = sentiment_score
    return data
