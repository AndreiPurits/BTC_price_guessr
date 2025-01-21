from newsapi import NewsApiClient
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

newsapi = NewsApiClient(api_key='your api key')

def fetch_data(start, end):
    btc_data = yf.download('BTC-USD', start, end, progress=False)
    btc_data.reset_index(inplace=True)
    btc_data["Date"] = pd.to_datetime(btc_data["Date"]).dt.date
    btc_data = btc_data[['Date', 'Close']]
    
    keywords = (
        'bitcoin OR cryptocurrency OR blockchain OR "bitcoin price" OR "bitcoin regulation" OR '
        '"crypto market" OR "cryptocurrency regulation" OR "bitcoin adoption" OR "bitcoin investment" OR '
        '"crypto exchange" OR "bitcoin futures" OR "bitcoin mining" OR "bitcoin ETF" OR "central bank digital currency" '
    )

    bitcoin_articles = newsapi.get_everything(
        q=keywords,
        language='en',
        from_param=start,
        to=end,
        sort_by='popularity',
    )

    articles_data = []
    for article in bitcoin_articles['articles']:
        articles_data.append({
            'Date': article['publishedAt'][:10],
            'Title': article['title'],
            'Description': article['description'],
            'Source': article['source']['name']
        })

    articles_df = pd.DataFrame(articles_data)
    articles_df['Date'] = pd.to_datetime(articles_df['Date']).dt.date
    merged_data = pd.merge(btc_data, articles_df, on='Date', how='left')
    return merged_data

today = date.today()
d1 = today.strftime("%Y-%m-%d")
d2 = date.today() - timedelta(days=30)
merged_data = fetch_data(d2, d1)
merged_data.dropna(inplace=True)

merged_data.to_csv('btc_news_and_price_cleaned.csv', index=False)

print(f"Merged Data:\n{merged_data.head()}")
