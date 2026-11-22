"""
FinBERT-based news sentiment pipeline for the MoE Trading Engine.

Scrapes financial news headlines via RSS feeds and computes sentiment
scores using the ProsusAI/finbert model from HuggingFace.

Outputs a daily sentiment score between -1.0 (extreme bearish) and
+1.0 (extreme bullish) for each target asset.
"""

import os
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from ticker to search terms for news
TICKER_KEYWORDS = {
    "^NSEI": ["nifty", "nifty 50", "indian stock market", "sensex"],
    "RELIANCE.NS": ["reliance industries", "reliance", "mukesh ambani"],
    "HDFCBANK.NS": ["hdfc bank", "hdfc"],
    "SBIN.NS": ["state bank of india", "sbi"],
    "TCS.NS": ["tata consultancy", "tcs"],
    "INFY.NS": ["infosys"],
    "TATAMOTORS.NS": ["tata motors"],
    "ICICIBANK.NS": ["icici bank"],
    "BAJFINANCE.NS": ["bajaj finance"],
}

# RSS feeds for Indian financial news
RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/results.xml",
    "https://www.livemint.com/rss/markets",
]


class SentimentPipeline:
    """News sentiment analysis using FinBERT.
    
    Workflow:
    1. Fetch headlines from RSS feeds
    2. Filter by ticker-relevant keywords
    3. Score each headline with FinBERT
    4. Aggregate into a single daily score
    5. Cache results to avoid re-computation
    """

    def __init__(
        self,
        cache_dir: str = "data/sentiment",
        model_name: str = "ProsusAI/finbert",
        device: str | None = None,
    ):
        """
        Args:
            cache_dir: Directory to cache sentiment scores
            model_name: HuggingFace model name for sentiment analysis
            device: Torch device for inference
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.device = device

        self._pipeline = None  # Lazy loaded

    def _get_pipeline(self):
        """Lazy-load the FinBERT pipeline to avoid import overhead."""
        if self._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
                self._pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=self.device,
                    truncation=True,
                    max_length=512,
                )
                logger.info(f"FinBERT model loaded: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}. Using neutral sentiment.")
                self._pipeline = None
        return self._pipeline

    def fetch_headlines(self, ticker: str = "^NSEI") -> list[str]:
        """Fetch recent news headlines from RSS feeds.
        
        Args:
            ticker: Asset ticker to fetch news for
        
        Returns:
            List of headline strings
        """
        headlines = []
        keywords = TICKER_KEYWORDS.get(ticker, ["indian stock market"])

        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed. Returning empty headlines.")
            return []

        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:  # Top 20 per feed
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title}. {summary}".strip()

                    # Filter by keywords (case-insensitive)
                    text_lower = text.lower()
                    if any(kw.lower() in text_lower for kw in keywords):
                        headlines.append(text)
                    elif ticker == "^NSEI":
                        # For Nifty, also include general market news
                        market_keywords = ["market", "stock", "equity", "bull", "bear", "rally", "crash", "fii", "dii"]
                        if any(mk in text_lower for mk in market_keywords):
                            headlines.append(text)

            except Exception as e:
                logger.debug(f"Failed to fetch {feed_url}: {e}")
                continue

        # Deduplicate
        seen = set()
        unique = []
        for h in headlines:
            h_hash = hashlib.md5(h.encode()).hexdigest()
            if h_hash not in seen:
                seen.add(h_hash)
                unique.append(h)

        logger.info(f"Fetched {len(unique)} unique headlines for {ticker}")
        return unique[:50]  # Cap at 50

    def score_headlines(self, headlines: list[str]) -> float:
        """Score a list of headlines using FinBERT.
        
        Args:
            headlines: List of news headline strings
        
        Returns:
            Aggregated sentiment score in [-1.0, 1.0]
        """
        if not headlines:
            return 0.0

        pipe = self._get_pipeline()
        if pipe is None:
            return 0.0

        try:
            results = pipe(headlines, batch_size=16)
        except Exception as e:
            logger.warning(f"FinBERT inference failed: {e}")
            return 0.0

        scores = []
        for r in results:
            label = r["label"].lower()
            score = r["score"]

            if label == "positive":
                scores.append(score)
            elif label == "negative":
                scores.append(-score)
            else:  # neutral
                scores.append(0.0)

        # Weighted average (recent headlines weighted slightly more)
        if len(scores) > 1:
            weights = np.linspace(0.5, 1.0, len(scores))
            weights /= weights.sum()
            aggregated = float(np.dot(scores, weights))
        else:
            aggregated = float(scores[0]) if scores else 0.0

        return np.clip(aggregated, -1.0, 1.0)

    def get_daily_sentiment(
        self, ticker: str = "^NSEI", date: str | None = None
    ) -> float:
        """Get sentiment score for a specific date, with caching.
        
        Args:
            ticker: Asset ticker
            date: Date string (YYYY-MM-DD). None = today.
        
        Returns:
            Sentiment score in [-1.0, 1.0]
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Check cache
        cache_file = self.cache_dir / f"{ticker.replace('^', 'IDX_')}_{date}.csv"
        if cache_file.exists():
            cached = pd.read_csv(cache_file)
            if not cached.empty:
                score = cached["sentiment"].iloc[0]
                logger.debug(f"Cache hit for {ticker} on {date}: {score:.4f}")
                return float(score)

        # Fetch and score
        headlines = self.fetch_headlines(ticker)
        score = self.score_headlines(headlines)

        # Cache
        pd.DataFrame({
            "date": [date],
            "ticker": [ticker],
            "sentiment": [score],
            "num_headlines": [len(headlines)],
        }).to_csv(cache_file, index=False)

        logger.info(f"Sentiment for {ticker} on {date}: {score:.4f} ({len(headlines)} headlines)")
        return score

    def backfill_sentiment(
        self,
        ticker: str,
        dates: list[str],
        default_value: float = 0.0,
    ) -> pd.Series:
        """Get sentiment scores for a list of historical dates.
        
        For dates where real sentiment is unavailable (pre-pipeline),
        returns the default value.
        
        Args:
            ticker: Asset ticker
            dates: List of date strings
            default_value: Default sentiment for dates without data
        
        Returns:
            pandas Series with sentiment scores indexed by date
        """
        scores = {}
        for date in dates:
            cache_file = self.cache_dir / f"{ticker.replace('^', 'IDX_')}_{date}.csv"
            if cache_file.exists():
                cached = pd.read_csv(cache_file)
                scores[date] = float(cached["sentiment"].iloc[0])
            else:
                scores[date] = default_value

        series = pd.Series(scores, name="finbert_score")
        
        cached_count = sum(1 for v in scores.values() if v != default_value)
        logger.info(
            f"Sentiment backfill for {ticker}: "
            f"{cached_count}/{len(dates)} dates with real data, "
            f"rest filled with {default_value}"
        )
        return series
