from newsapi import NewsApiClient  # Ensure this import is at the top
import yfinance as yf
from datetime import date, timedelta
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import numpy as np
import pandas as pd
from scv_creation import *
# Load the trained model and vectorizer once at the start
model = joblib.load('btc_price_predictor_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load the pre-trained vectorizer

def predict_btc_price(news_article, prev_close):
    # Vectorize the news article and apply the model to predict the price
    news_vector = vectorizer.transform([news_article])  # Vectorizing the news article
    sentiment_score = get_sentiment(news_article)  # Get sentiment from the article

    # Assuming the model expects a feature array [sentiment_score, prev_close]
    prediction_input = np.array([[sentiment_score, prev_close]])
    
    # Predict the price using the model
    predicted_price = model.predict(prediction_input)
    return predicted_price[0]  # Return the predicted price

def get_sentiment(text):
    # Simple sentiment analysis using TextBlob
    sentiment = TextBlob(text).sentiment.polarity
    return sentiment

d1 = today.strftime("%Y-%m-%d")
d2 = date.today() - timedelta(days=1)


# Main function
def main():
    print("Do you want to:")
    print("1. Input news and Bitcoin price manually")
    print("2. Fetch news and Bitcoin price automatically for today")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Manual input
        news_article = input("Enter today's Bitcoin-related news article: ")
        prev_close = float(input("Enter today's Bitcoin closing price: "))
    elif choice == "2":
        # Automatic fetching
        merged_data = fetch_data(d2,d1) 
        merged_data.dropna(inplace=True)

        # Save the merged data to a CSV file with only one 'Date' column
        merged_data.to_csv('btc_news_and_price_today.csv', index=False)
        if len(merged_data) == 0:
            print("No data fetched. Exiting.")
            return
        df = pd.read_csv('btc_news_and_price_today.csv')
        df.columns = ['Date', 'Original_Date', 'Close', 'Title', 'Description', 'Source']
        news_article = df['Description'].iloc[0]  # Taking the description of the first article
        prev_close = df['Close'].iloc[0]  # Taking the previous close pric
    else:
        print("Invalid choice. Exiting.")
        return
    new_text = vectorizer.transform([news_article]).toarray()
    new_features = np.hstack([new_text, np.array([[prev_close, news_article]])])
    #predicted_price = predict_btc_price(news_article, prev_close)
    predicted_price = model.predict(new_features)
    print(f"Predicted Bitcoin price for tomorrow: {predicted_price:.2f}")

# Run the program
if __name__ == "__main__":
    main()
