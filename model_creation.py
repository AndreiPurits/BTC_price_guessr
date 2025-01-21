import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

df = pd.read_csv('btc_news_and_price_cleaned.csv')
df.columns = ['Date', 'Original_Date', 'Close', 'Title', 'Description', 'Source']

vectorizer = TfidfVectorizer(max_features=500)

def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    return sentiment

df['Sentiment'] = df['Description'].apply(lambda x: get_sentiment(str(x)))

X_text = vectorizer.fit_transform(df['Description']).toarray()

df['Prev_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)

X_text = X_text[:len(df)]

X = np.hstack([X_text, df[['Prev_Close', 'Sentiment']].values])
y = df['Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

new_article = "Bitcoin hits new highs as institutional adoption increases. Experts believe that the price will continue to rise."
new_sentiment = get_sentiment(new_article)
new_text = vectorizer.transform([new_article]).toarray()

prev_close = df['Close'].iloc[-1]
print("prev", prev_close)

new_features = np.hstack([new_text, np.array([[prev_close, new_sentiment]])])

predicted_price = model.predict(new_features)
print(f"Predicted Bitcoin Price: {predicted_price[0]}")
