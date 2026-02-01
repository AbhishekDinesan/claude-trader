"""
Sentiment Analysis Engine
Analyzes sentiment from Reddit, news, and social media
Uses free APIs and web scraping
"""

import requests
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

from config import SENTIMENT_CONFIG, SentimentConfig


@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    symbol: str
    
    # Overall sentiment (-1 to 1)
    overall_sentiment: float
    sentiment_label: str  # 'very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish'
    
    # Reddit sentiment
    reddit_sentiment: float
    reddit_mentions: int
    reddit_posts: List[Dict]
    
    # News sentiment
    news_sentiment: float
    news_articles: int
    news_headlines: List[str]
    
    # Social buzz score (0-100)
    buzz_score: float
    
    # Trend
    sentiment_trend: str  # 'improving', 'declining', 'stable'


class SentimentAnalyzer:
    """Analyzes sentiment from various sources"""
    
    def __init__(self, config: SentimentConfig = SENTIMENT_CONFIG):
        self.config = config
        
        # Initialize VADER for sentiment analysis
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
    
    def _rate_limit(self):
        """Simple rate limiting to be respectful to APIs"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of text using available libraries
        Returns score from -1 (very negative) to 1 (very positive)
        """
        if not text:
            return 0.0
        
        scores = []
        
        # VADER sentiment (specialized for social media)
        if self.vader:
            vader_score = self.vader.polarity_scores(text)
            scores.append(vader_score['compound'])
        
        # TextBlob sentiment
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            scores.append(blob.sentiment.polarity)
        
        # Return average if we have scores, else 0
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_reddit_sentiment(self, symbol: str) -> Tuple[float, int, List[Dict]]:
        """
        Get sentiment from Reddit using public JSON API (no auth needed)
        
        Returns:
            Tuple of (sentiment_score, mention_count, posts_list)
        """
        posts = []
        all_text = []
        
        # Reddit's public JSON API endpoints
        search_queries = [
            f"${symbol}",  # Ticker with $ prefix
            symbol,        # Just the ticker
        ]
        
        for subreddit in self.config.subreddits[:3]:  # Limit subreddits to avoid rate limits
            for query in search_queries:
                try:
                    self._rate_limit()
                    
                    # Use Reddit's public search JSON endpoint
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        'q': query,
                        'sort': 'new',
                        'limit': 25,
                        't': 'week',
                        'restrict_sr': 'on'
                    }
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    response = requests.get(url, params=params, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for post in data.get('data', {}).get('children', []):
                            post_data = post.get('data', {})
                            
                            title = post_data.get('title', '')
                            selftext = post_data.get('selftext', '')
                            score = post_data.get('score', 0)
                            num_comments = post_data.get('num_comments', 0)
                            
                            # Check if ticker is actually mentioned
                            combined_text = f"{title} {selftext}".upper()
                            if symbol.upper() in combined_text or f"${symbol.upper()}" in combined_text:
                                all_text.append(f"{title} {selftext}")
                                posts.append({
                                    'subreddit': subreddit,
                                    'title': title[:100],
                                    'score': score,
                                    'comments': num_comments,
                                    'sentiment': self.analyze_text(f"{title} {selftext}")
                                })
                    
                    time.sleep(0.5)  # Be nice to Reddit
                    
                except Exception as e:
                    continue
        
        # Calculate overall sentiment
        if posts:
            # Weight by engagement (upvotes + comments)
            weighted_sum = sum(
                p['sentiment'] * (1 + p['score'] * 0.01 + p['comments'] * 0.05)
                for p in posts
            )
            total_weight = sum(
                1 + p['score'] * 0.01 + p['comments'] * 0.05
                for p in posts
            )
            sentiment = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            sentiment = 0.0
        
        # Deduplicate posts
        unique_posts = []
        seen_titles = set()
        for post in posts:
            if post['title'] not in seen_titles:
                seen_titles.add(post['title'])
                unique_posts.append(post)
        
        return sentiment, len(unique_posts), unique_posts[:10]
    
    def get_news_sentiment(self, symbol: str) -> Tuple[float, int, List[str]]:
        """
        Get news sentiment using free news sources
        
        Returns:
            Tuple of (sentiment_score, article_count, headlines_list)
        """
        headlines = []
        
        try:
            # Use Google News RSS feed (free, no API key needed)
            self._rate_limit()
            
            url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Parse RSS feed manually (simple approach)
                import re
                
                # Find all titles in the RSS
                titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', response.text)
                if not titles:
                    titles = re.findall(r'<title>(.*?)</title>', response.text)
                
                # Skip the first title (feed title)
                for title in titles[1:21]:  # Get up to 20 headlines
                    # Clean up HTML entities
                    title = title.replace('&amp;', '&').replace('&quot;', '"')
                    title = title.replace('&#39;', "'").replace('&lt;', '<')
                    
                    # Filter to relevant headlines
                    if symbol.upper() in title.upper():
                        headlines.append(title)
        
        except Exception as e:
            pass
        
        # Calculate sentiment
        if headlines:
            sentiments = [self.analyze_text(h) for h in headlines]
            avg_sentiment = sum(sentiments) / len(sentiments)
        else:
            avg_sentiment = 0.0
        
        return avg_sentiment, len(headlines), headlines[:10]
    
    def analyze_symbol(self, symbol: str) -> SentimentResult:
        """
        Complete sentiment analysis for a symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            SentimentResult with all sentiment data
        """
        print(f"  [*] Analyzing sentiment for {symbol}...")
        
        # Get Reddit sentiment
        reddit_sentiment, reddit_mentions, reddit_posts = self.get_reddit_sentiment(symbol)
        
        # Get news sentiment
        news_sentiment, news_articles, news_headlines = self.get_news_sentiment(symbol)
        
        # Calculate overall sentiment (weighted average)
        if reddit_mentions + news_articles > 0:
            # Weight Reddit higher for meme stocks, news higher for established stocks
            reddit_weight = min(0.6, 0.3 + reddit_mentions * 0.02)
            news_weight = 1 - reddit_weight
            
            overall_sentiment = (
                reddit_sentiment * reddit_weight + 
                news_sentiment * news_weight
            )
        else:
            overall_sentiment = 0.0
        
        # Determine sentiment label
        if overall_sentiment > 0.5:
            sentiment_label = 'very_bullish'
        elif overall_sentiment > 0.15:
            sentiment_label = 'bullish'
        elif overall_sentiment > -0.15:
            sentiment_label = 'neutral'
        elif overall_sentiment > -0.5:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'very_bearish'
        
        # Calculate buzz score (0-100)
        buzz_score = min(100, (reddit_mentions * 5 + news_articles * 3))
        
        # Sentiment trend (simplified - would need historical data for real trend)
        if reddit_posts:
            recent_sentiment = sum(p['sentiment'] for p in reddit_posts[:3]) / min(3, len(reddit_posts))
            if recent_sentiment > overall_sentiment + 0.1:
                sentiment_trend = 'improving'
            elif recent_sentiment < overall_sentiment - 0.1:
                sentiment_trend = 'declining'
            else:
                sentiment_trend = 'stable'
        else:
            sentiment_trend = 'stable'
        
        return SentimentResult(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            sentiment_label=sentiment_label,
            reddit_sentiment=reddit_sentiment,
            reddit_mentions=reddit_mentions,
            reddit_posts=reddit_posts,
            news_sentiment=news_sentiment,
            news_articles=news_articles,
            news_headlines=news_headlines,
            buzz_score=buzz_score,
            sentiment_trend=sentiment_trend
        )
    
    def batch_analyze(self, symbols: List[str]) -> Dict[str, SentimentResult]:
        """
        Analyze sentiment for multiple symbols
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dict mapping symbol to SentimentResult
        """
        results = {}
        
        print(f"\n[*] Analyzing sentiment for {len(symbols)} stocks...")
        
        for i, symbol in enumerate(symbols):
            try:
                results[symbol] = self.analyze_symbol(symbol)
                print(f"  [{i+1}/{len(symbols)}] {symbol}: {results[symbol].sentiment_label} "
                      f"(score: {results[symbol].overall_sentiment:.2f}, "
                      f"mentions: {results[symbol].reddit_mentions})")
            except Exception as e:
                print(f"  [{i+1}/{len(symbols)}] {symbol}: Error - {e}")
        
        return results


def get_trending_tickers() -> List[str]:
    """
    Get currently trending stock tickers from various sources
    
    Returns:
        List of trending ticker symbols
    """
    tickers = set()
    
    try:
        # Try to get trending from Yahoo Finance
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(
            'https://finance.yahoo.com/trending-tickers',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            # Extract tickers using regex
            found = re.findall(r'/quote/([A-Z]{1,5})\?', response.text)
            tickers.update(found[:20])
    
    except:
        pass
    
    return list(tickers)
