import api_news as api
import finnhub
import faiss
import numpy as np
import yfinance as yf
from model import Request
from sentence_transformers import SentenceTransformer
from transformers import pipeline

api_key = "cvi1mvpr01qks9q7mol0cvi1mvpr01qks9q7molg"
groq_api_key = "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"

req = Request(groq_api_key)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True)

finnhub_client = finnhub.Client(api_key=api_key)

dimension = 384
index = faiss.IndexFlatL2(dimension)
news_data = []

def fetch_news():
    global news_data
    print("Fetching news...")
    news = api.fetch_finance_news()
    
    processed_news = []
    embeddings = []
    
    for article in news:
        text = article["headline"] + " " + article.get("summary", "")
        embedding = embedding_model.encode(text)
        processed_news.append({"text": text, "embedding": embedding})
        embeddings.append(embedding)

    news_data = processed_news
    news_vectors = np.array(embeddings)
    index.add(news_vectors)
    print(f"Stored {len(news_data)} news articles in vector database.")

def retrieve_similar_news(query, top_k=5):
    query_vector = embedding_model.encode(query).reshape(1, -1)
    _, indices = index.search(query_vector, top_k)
    return [news_data[i]["text"] for i in indices[0]]

def analyze_sentiment(news_texts):
    sentiment_results = sentiment_analyzer(news_texts)
    sentiments = []
    for result in sentiment_results:
        best_match = max(result, key=lambda x: x["score"])
        sentiments.append((best_match["label"], round(best_match["score"] * 100, 2)))
    return sentiments

def extract_tickers(text):
    words = text.split()
    possible_tickers = [word for word in words if word.isupper() and 2 <= len(word) <= 5]
    
    verified_tickers = []
    for ticker in possible_tickers:
        try:
            stock_data = finnhub_client.quote(ticker)
            if stock_data and stock_data["c"] != 0:
                verified_tickers.append(ticker)
        except:
            pass  
    return verified_tickers

def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1d")["Close"].iloc[-1]
    except:
        return None

def predict_stock_drop():
    print("\nFetching relevant negative news...")
    negative_news = []
    
    for article in news_data:
        sentiment, confidence = analyze_sentiment([article["text"]])[0]
        if sentiment == "negative" and confidence > 70:
            negative_news.append(article["text"])

    if not negative_news:
        print("No significant negative news found.")
        return

    affected_stocks = {}
    for news in negative_news:
        tickers = extract_tickers(news)
        for ticker in tickers:
            if ticker not in affected_stocks:
                affected_stocks[ticker] = 1
            else:
                affected_stocks[ticker] += 1

    ai_prompt = f"""
    Based on the following financial news, predict how much stock prices might fall:
    {negative_news[:3]}  
    Consider historical patterns and market trends. Provide estimated drop % for each stock.
    """

    req.request(ai_prompt)
    stock_predictions = req.answer()

    if not stock_predictions or not isinstance(stock_predictions, dict):
        print("AI did not return valid predictions.")
        return

    print("\n📊 AI's Stock Drop Prediction vs Real Data:")
    print("───────────────────────────────────────")
    print(f"{'Stock':<10}{'Predicted Drop (%)':>20}{'Real Drop (%)':>20}")
    print("───────────────────────────────────────")
    
    for stock, predicted_drop in stock_predictions.items():
        old_price = get_stock_price(stock)
        if old_price is None:
            real_drop = "N/A"
        else:
            new_price = get_stock_price(stock)
            real_drop = round(((old_price - new_price) / old_price) * 100, 2) if new_price else "N/A"
        print(f"{stock:<10}{predicted_drop:>20}%{real_drop:>20}%")
    print("───────────────────────────────────────")

if __name__ == "__main__":
    fetch_news()
    predict_stock_drop()