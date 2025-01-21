from newsapi import NewsApiClient
import yfinance as yf
from datetime import date, timedelta
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import numpy as np
import pandas as pd
from scv_creation import *

model = joblib.load('btc_price_predictor_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_btc_price(news_article, prev_close):
    news_vector = vectorizer.transform([news_article])
    sentiment_score = get_sentiment(news_article)
    prediction_input = np.array([[sentiment_score, prev_close]])
    predicted_price = model.predict(prediction_input)
    return predicted_price[0]

def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    return sentiment

d1 = today.strftime("%Y-%m-%d")
d2 = date.today() - timedelta(days=1)
def main():
    print("Do you want to:")
    print("1. Input news and Bitcoin price manually")
    print("2. Fetch news and Bitcoin price automatically for today")
    
    # Debug print to check if the program reaches here
    print("Prompting for choice...")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        print("You chose to input news and price manually.")
        news_article = input("Enter today's Bitcoin-related news article: ")
        prev_close = float(input("Enter today's Bitcoin closing price: "))
    elif choice == "2":
        print("You chose to fetch news and price automatically.")
        merged_data = fetch_data(d2,d1)
        merged_data.dropna(inplace=True)
        merged_data.to_csv('btc_news_and_price_today.csv', index=False)
        if len(merged_data) == 0:
            print("No data fetched. Exiting.")
            return
        df = pd.read_csv('btc_news_and_price_today.csv')
        df.columns = ['Date', 'Original_Date', 'Close', 'Title', 'Description', 'Source']
        news_article = df['Description'].iloc[0]
        prev_close = df['Close'].iloc[0]
    else:
        print("Invalid choce. Exiting.")
        return
    
    # Convert the news article into the TF-IDF features
    news_vector = vectorizer.transform([news_article]).toarray()
    
    # Get the sentiment score for the news article
    sentiment_score = get_sentiment(news_article)
    
    # Create the input features array (TF-IDF features + sentiment + prev_close)
    new_features = np.hstack([news_vector, np.array([[prev_close, sentiment_score]])])
    
    # Make the prediction
    predicted_price = model.predict(new_features)
    print(f"Predicted Bitcoin price for tomorrow: {predicted_price[0]:.2f}")

if __name__ == "__main__":
    print("Starting the program...")
    main()
