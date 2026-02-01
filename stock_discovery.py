"""
Stock Discovery Module
Automatically discovers new stocks that are performing well

Features:
- Scans for trending stocks
- Identifies top movers and gainers
- Adds promising stocks to watchlist
- Tracks which discovered stocks perform well
"""

import json
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional
import yfinance as yf
from bs4 import BeautifulSoup

from config import DEFAULT_WATCHLIST


# File to track discovered stocks
DISCOVERED_FILE = os.path.join(os.path.dirname(__file__), 'discovered_stocks.json')


class StockDiscovery:
    """Discovers new stocks to track"""
    
    def __init__(self):
        self.discovered = self._load_discovered()
        self.current_watchlist = set(DEFAULT_WATCHLIST)
    
    def _load_discovered(self) -> Dict:
        """Load previously discovered stocks"""
        if os.path.exists(DISCOVERED_FILE):
            with open(DISCOVERED_FILE, 'r') as f:
                return json.load(f)
        return {
            'stocks': {},  # symbol -> discovery info
            'added_to_watchlist': [],
            'last_scan': None
        }
    
    def _save_discovered(self):
        """Save discovered stocks"""
        with open(DISCOVERED_FILE, 'w') as f:
            json.dump(self.discovered, f, indent=2)
    
    def get_yahoo_trending(self) -> List[str]:
        """Get trending tickers from Yahoo Finance"""
        tickers = []
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            # Trending tickers
            response = requests.get(
                'https://finance.yahoo.com/trending-tickers',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                import re
                found = re.findall(r'/quote/([A-Z]{1,5})\?', response.text)
                tickers.extend(found[:20])
        except:
            pass
        
        return list(set(tickers))
    
    def get_top_gainers(self) -> List[Dict]:
        """Get top gaining stocks today"""
        gainers = []
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(
                'https://finance.yahoo.com/gainers',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                import re
                # Extract tickers and their gains
                found = re.findall(r'/quote/([A-Z]{1,5})\?', response.text)
                
                for symbol in found[:15]:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period='5d')
                        if len(hist) >= 2:
                            change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
                            if change > 5:  # Only if up more than 5%
                                gainers.append({
                                    'symbol': symbol,
                                    'change_1d': change,
                                    'price': hist['Close'].iloc[-1]
                                })
                    except:
                        continue
        except:
            pass
        
        return sorted(gainers, key=lambda x: x['change_1d'], reverse=True)
    
    def get_reddit_mentions(self) -> List[str]:
        """Get frequently mentioned stocks from Reddit"""
        tickers = []
        
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        
        for sub in subreddits:
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                url = f"https://www.reddit.com/r/{sub}/hot.json?limit=50"
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    import re
                    
                    for post in data.get('data', {}).get('children', []):
                        title = post.get('data', {}).get('title', '')
                        # Find stock tickers (1-5 uppercase letters, often with $)
                        found = re.findall(r'\$([A-Z]{1,5})\b', title)
                        found += re.findall(r'\b([A-Z]{2,5})\b', title)
                        tickers.extend(found)
            except:
                continue
        
        # Count occurrences and return most mentioned
        from collections import Counter
        counts = Counter(tickers)
        
        # Filter out common words that look like tickers
        blacklist = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 
                    'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'BUY',
                    'NOW', 'HOW', 'GET', 'UP', 'SO', 'IF', 'ANY', 'CEO', 'IPO',
                    'ETF', 'GDP', 'USA', 'SEC', 'FED', 'IMO', 'DD', 'YOLO'}
        
        return [t for t, c in counts.most_common(20) if t not in blacklist and c >= 2]
    
    def validate_ticker(self, symbol: str) -> Optional[Dict]:
        """Validate a ticker is real and get basic info"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if it's a valid stock
            market_cap = info.get('marketCap', 0)
            if not market_cap or market_cap < 50_000_000:  # Min $50M market cap
                return None
            
            # Get recent performance
            hist = ticker.history(period='1mo')
            if hist.empty:
                return None
            
            change_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
            avg_volume = hist['Volume'].mean()
            
            if avg_volume < 100000:  # Min 100K average volume
                return None
            
            return {
                'symbol': symbol,
                'name': info.get('shortName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': market_cap,
                'price': hist['Close'].iloc[-1],
                'change_1m': change_1m,
                'avg_volume': avg_volume
            }
        except:
            return None
    
    def discover_new_stocks(self) -> List[Dict]:
        """
        Run discovery to find new stocks worth tracking
        
        Returns:
            List of newly discovered stocks with their info
        """
        print("\n[STOCK DISCOVERY]")
        print("="*50)
        
        all_candidates = set()
        
        # Get candidates from various sources
        print("  Scanning Yahoo trending...")
        trending = self.get_yahoo_trending()
        all_candidates.update(trending)
        print(f"    Found {len(trending)} trending")
        
        print("  Scanning top gainers...")
        gainers = self.get_top_gainers()
        all_candidates.update([g['symbol'] for g in gainers])
        print(f"    Found {len(gainers)} gainers")
        
        print("  Scanning Reddit mentions...")
        reddit = self.get_reddit_mentions()
        all_candidates.update(reddit)
        print(f"    Found {len(reddit)} Reddit mentions")
        
        # Filter out stocks we already track
        new_candidates = all_candidates - self.current_watchlist
        new_candidates -= set(self.discovered['stocks'].keys())
        
        print(f"\n  New candidates to evaluate: {len(new_candidates)}")
        
        # Validate and score new stocks
        discovered = []
        for symbol in list(new_candidates)[:20]:  # Limit to 20 per run
            info = self.validate_ticker(symbol)
            if info and info['change_1m'] > 10:  # Only if up >10% this month
                info['discovered_date'] = datetime.now().isoformat()
                info['source'] = 'auto-discovery'
                discovered.append(info)
                
                # Save to discovered list
                self.discovered['stocks'][symbol] = info
                print(f"    [+] {symbol}: +{info['change_1m']:.1f}% (1mo), ${info['price']:.2f}")
        
        self.discovered['last_scan'] = datetime.now().isoformat()
        self._save_discovered()
        
        print(f"\n  Discovered {len(discovered)} new promising stocks")
        
        return discovered
    
    def get_discovery_recommendations(self, top_n: int = 5) -> List[Dict]:
        """Get top discovered stocks to consider adding to watchlist"""
        stocks = list(self.discovered['stocks'].values())
        
        # Update current prices and performance
        updated = []
        for stock in stocks:
            try:
                ticker = yf.Ticker(stock['symbol'])
                hist = ticker.history(period='5d')
                if not hist.empty:
                    stock['current_price'] = hist['Close'].iloc[-1]
                    stock['change_since_discovery'] = (
                        (stock['current_price'] / stock['price'] - 1) * 100
                    )
                    updated.append(stock)
            except:
                continue
        
        # Sort by performance since discovery
        updated.sort(key=lambda x: x.get('change_since_discovery', 0), reverse=True)
        
        return updated[:top_n]
    
    def add_to_watchlist(self, symbol: str) -> bool:
        """Add a discovered stock to the permanent watchlist"""
        if symbol in self.current_watchlist:
            return False
        
        # Add to discovered tracking
        if symbol not in self.discovered['added_to_watchlist']:
            self.discovered['added_to_watchlist'].append(symbol)
            self._save_discovered()
        
        return True


def run_discovery() -> Dict:
    """Run stock discovery and return results"""
    discovery = StockDiscovery()
    
    # Discover new stocks
    new_stocks = discovery.discover_new_stocks()
    
    # Get recommendations from previously discovered
    recommendations = discovery.get_discovery_recommendations(top_n=5)
    
    return {
        'new_discoveries': new_stocks,
        'top_recommendations': recommendations,
        'total_discovered': len(discovery.discovered['stocks']),
        'timestamp': datetime.now().isoformat()
    }
