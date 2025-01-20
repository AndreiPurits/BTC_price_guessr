from newsapi import NewsApiClient
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# Init NewsAPI client
newsapi = NewsApiClient(api_key='96f6b944caaa44db88f1178f8f47f540')

def fetch_data(start, end):

    # Download Bitcoin price data for the last 30 days
    btc_data = yf.download('BTC-USD', start, end, progress=False)

    # Reset index so 'Date' becomes a column
    btc_data.reset_index(inplace=True)

    # Convert 'Date' column to datetime if not already and remove the time part
    btc_data["Date"] = pd.to_datetime(btc_data["Date"]).dt.date

    # Select only 'Date' and 'Close' columns
    btc_data = btc_data[['Date', 'Close']]

    # Fetch Bitcoin-related news articles from NewsAPI
    keywords = (
        'bitcoin OR cryptocurrency OR blockchain OR "bitcoin price" OR "bitcoin regulation" OR '
        '"crypto market" OR "cryptocurrency regulation" OR "bitcoin adoption" OR "bitcoin investment" OR '
        '"crypto exchange" OR "bitcoin futures" OR "bitcoin mining" OR "bitcoin ETF" OR "central bank digital currency" '
    )

    bitcoin_articles = newsapi.get_everything(
        q=keywords,  # Search for articles about Bitcoin
        language='en',  # English language
        from_param=start,  # Starting date
        to=end,  # Ending date
        sort_by='popularity',  # Optional: Sort by popularity
    )

    # Prepare articles data
    articles_data = []
    for article in bitcoin_articles['articles']:
        articles_data.append({
            'Date': article['publishedAt'][:10],  # Extract only the date part
            'Title': article['title'],
            'Description': article['description'],
            'Source': article['source']['name']
        })

    # Convert articles data to DataFrame
    articles_df = pd.DataFrame(articles_data)

    # Convert 'Date' column in articles DataFrame to datetime
    articles_df['Date'] = pd.to_datetime(articles_df['Date']).dt.date

    # Merge Bitcoin price data with articles data on the 'Date' column
    merged_data = pd.merge(btc_data, articles_df, on='Date', how='left')
    return merged_data
today = date.today()
d1 = today.strftime("%Y-%m-%d")
d2 = date.today() - timedelta(days=30)
merged_data = fetch_data(d2,d1) 
merged_data.dropna(inplace=True)

# Save the merged data to a CSV file with only one 'Date' column
merged_data.to_csv('btc_news_and_price_cleaned.csv', index=False)

# Display the first few rows of the merged data
print(f"Merged Data:\n{merged_data.head()}")
