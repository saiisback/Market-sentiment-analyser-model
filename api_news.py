import finnhub
import time
from datetime import datetime, timedelta

# Initialize Finnhub Client with API Key
API_KEY = "cvi1mvpr01qks9q7mol0cvi1mvpr01qks9q7molg"
finnhub_client = finnhub.Client(api_key=API_KEY)

# Define the stock symbol (Modify as needed)
STOCK_SYMBOL = "TSLA"

def fetch_finance_news():
    today = datetime.today()
    start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    try:
        # Fetch news from Finnhub
        news_data = finnhub_client.company_news(STOCK_SYMBOL, _from=start_date, to=end_date)

        # Display the news
        if news_data:
            return news_data
        else:
            print(f"\nNo news found for {STOCK_SYMBOL} on {end_date}.\n")

    except Exception as e:
        print(f"\nError fetching news: {e}\n")
    
    
