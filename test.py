import faiss
import numpy as np
from moneycontrol import moneycontrol_api
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from model import Request  # Assuming Groq API is integrated via Request

# Initialize APIs and models

moneycontrol = moneycontrol_api()
groq_api_key = "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"
req = Request(groq_api_key)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # For text embeddings
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")

# FAISS vector database setup
dimension = 384  # Embedding size
index = faiss.IndexFlatL2(dimension)
news_data = []

def fetch_news():
    """Fetch latest financial news from Moneycontrol and store embeddings."""
    global news_data
    print("Fetching latest financial news...")
    news = moneycontrol.get_latest_news()
    
    processed_news = []
    embeddings = []
    
    for article in news:
        text = article["Title:"] + " " + article.get("Summary:", "")
        embedding = embedding_model.encode(text)
        processed_news.append({"text": text, "embedding": embedding})
        embeddings.append(embedding)
    
    # Store news in FAISS index
    news_data = processed_news
    news_vectors = np.array(embeddings)
    index.add(news_vectors)
    print(f"Stored {len(news_data)} news articles in vector database.")

def retrieve_similar_news(query, top_k=5):
    """Retrieve top-k relevant news articles based on query."""
    query_vector = embedding_model.encode(query).reshape(1, -1)
    _, indices = index.search(query_vector, top_k)
    return [news_data[i]["text"] for i in indices[0]]

def analyze_sentiment(news_texts):
    """Perform sentiment analysis on a list of news texts."""
    sentiments = sentiment_analyzer(news_texts)
    return [s["label"] for s in sentiments]

def extract_tickers(text):
    """Extract potential Indian stock symbols from news text."""
    words = text.split()
    return [word for word in words if word.isupper() and len(word) <= 5]  # Basic NSE/BSE ticker filter

def predict_market_impact():
    """Identify stocks that could be affected by negative news."""
    print("Analyzing market sentiment...")
    negative_news = []
    
    for article in news_data:
        sentiment = analyze_sentiment([article["text"]])[0]
        if sentiment == "negative":
            negative_news.append(article["text"])
    
    if not negative_news:
        print("No significant negative news found.")
        return []
    
    affected_stocks = {}
    for news in negative_news:
        tickers = extract_tickers(news)
        for ticker in tickers:
            affected_stocks[ticker] = affected_stocks.get(ticker, 0) + 1
    
    # Use Groq AI model to estimate market impact
    ai_prompt = f"""
    Based on the following financial news, predict how much stock prices might change:

    {negative_news[:3]}

    Consider historical patterns and market trends. Provide estimated impact for each stock.
    """
    req.request(ai_prompt)
    print("\n AI's Market Impact Prediction:")
    req.answer()

if __name__ == "__main__":
    fetch_news()
    predict_market_impact()
