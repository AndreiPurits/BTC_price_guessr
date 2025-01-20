import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

# Step 1: Read the dataset
df = pd.read_csv('btc_news_and_price_cleaned.csv')

# Step 2: Rename columns to clean up the dataset
df.columns = ['Date', 'Original_Date', 'Close', 'Title', 'Description', 'Source']

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=500)  # Limit to top 500 features for simplicity

# Step 3: Sentiment analysis function
def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    return sentiment

# Step 4: Preprocess text data (Sentiment Analysis)
df['Sentiment'] = df['Description'].apply(lambda x: get_sentiment(str(x)))

# Step 5: Convert text data (news) into numerical features using TF-IDF
X_text = vectorizer.fit_transform(df['Description']).toarray()

# Step 6: Combine features (Bitcoin price and news features)
df['Prev_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)  # Drop rows with NaN values after shifting

# Ensure both 'X_text' and the numerical features have the same number of rows
X_text = X_text[:len(df)]  # Slice 'X_text' to match the length of df

# Prepare the final dataset for training
X = np.hstack([X_text, df[['Prev_Close', 'Sentiment']].values])  # Combine price and sentiment features
y = df['Close'].values  # Target variable (Bitcoin closing price)
print("X", X)
print("Y", y)
# Step 7: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Random Forest Regressor for training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 9: Optional - Predict Bitcoin price for a new article
new_article = "Bitcoin hits new highs as institutional adoption increases. Experts believe that the price will continue to rise."
new_sentiment = get_sentiment(new_article)
new_text = vectorizer.transform([new_article]).toarray()

# Assume the previous day's close price is the last close price from the dataset
prev_close = df['Close'].iloc[-1]
print("prev", prev_close)
# Combine the new features for prediction
new_features = np.hstack([new_text, np.array([[prev_close, new_sentiment]])])

# Predict the price
predicted_price = model.predict(new_features)
print(f"Predicted Bitcoin Price: {predicted_price[0]}")

# Step 10: Save the model and vectorizer
#joblib.dump(model, 'btc_price_predictor_model.pkl')  # Save the model
#joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')     # Save the vectorizer
