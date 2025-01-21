# Bitcoin Price Prediction Based on News and Sentiment Analysis

## Overview
This program predicts the future Bitcoin closing price using news articles and sentiment analysis combined with historical Bitcoin price data. It leverages machine learning to integrate text-based insights with numerical market data.

---

## Features
- **News Sentiment Analysis**: Uses TextBlob for sentiment polarity scoring of Bitcoin-related news articles.
- **TF-IDF Vectorization**: Converts news article descriptions into numerical features using TfidfVectorizer.
- **Machine Learning Model**: Trains a RandomForestRegressor to predict Bitcoin prices.
- **Integrated Dataset**: Merges historical Bitcoin price data with news sentiment and text features.
- **Custom Predictions**: Predicts the future price of Bitcoin based on new news articles.

---

## Installation and Setup
## Prerequisites

1. Python 3.7+
2. Required Python libraries:
- pandas
- numpy
- textblob
- scikit-learn
- joblib
  
Install the required Python packages before running the program:

```pip install pandas numpy textblob scikit-learn joblib```

## Dataset Preparation
The program reads a CSV file named btc_news_and_price_cleaned.csv, which should include Bitcoin-related news and historical price data. The columns must be:
- Date
- Original_Date
- Close (Bitcoin's closing price)
- Title (news title)
- Description (news description)
- Source (news source)

To create or clean the dataset, ensure that:

- Text data (Description) is meaningful and relevant to Bitcoin.
- Historical Bitcoin prices (Close) are accurate.

## How to Run

Training the Model:

- Run the script to train the model using historical data.
- The program will calculate sentiment polarity, vectorize text features, and combine them with numerical features like previous closing prices.
- The model's performance is evaluated using metrics like MAE, MSE, and R².
- 
Making Predictions:

Provide a new Bitcoin-related news article as input. The program calculates the sentiment, vectorizes the text, and predicts the next day's Bitcoin closing price using the trained model.
Outputs

## Example Output

``` Mean Absolute Error:
Mean Squared Error: 550,000.50
R² Score: 0.91
```

## Limitations
Sentiment analysis using TextBlob is basic and might not capture complex sentiments.
TF-IDF vectorization is limited to the top 500 features for simplicity.
The model is dependent on the quality and relevance of input data.
Future Enhancements
- Use more advanced sentiment analysis libraries like VADER or BERT.
- Incorporate additional market indicators like trading volume or social media sentiment.
- Train on a larger dataset for improved accuracy.
