"""
Stock Screener
Identifies high-potential stocks based on momentum, volatility, and volume
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import SCREENER_CONFIG, ScreenerConfig, DEFAULT_WATCHLIST


@dataclass
class ScreenerResult:
    """Container for screener results"""
    symbol: str
    price: float
    market_cap: float
    volume: int
    avg_volume: int
    
    # Momentum metrics
    change_1d: float
    change_1w: float
    change_1m: float
    change_3m: float
    
    # Volatility
    volatility: float  # Average daily % range
    beta: Optional[float]
    
    # Volume analysis
    volume_ratio: float  # Current vs average
    
    # Fundamental (basic)
    pe_ratio: Optional[float]
    sector: str
    
    # Screening flags
    passes_screen: bool
    screen_reasons: List[str]


class StockScreener:
    """Screens stocks for high-potential trading opportunities"""
    
    def __init__(self, config: ScreenerConfig = SCREENER_CONFIG):
        self.config = config
    
    def screen_stock(self, symbol: str) -> Optional[ScreenerResult]:
        """
        Screen a single stock for trading potential
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            ScreenerResult if data available, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period="3mo")
            if hist.empty or len(hist) < 20:
                return None
            
            # Get info (may fail for some tickers)
            try:
                info = ticker.info
            except:
                info = {}
            
            # Current price and volume
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            # Market cap
            market_cap = info.get('marketCap', 0) or 0
            
            # Calculate momentum at different timeframes
            change_1d = self._calc_change(hist, 1)
            change_1w = self._calc_change(hist, 5)
            change_1m = self._calc_change(hist, 21)
            change_3m = self._calc_change(hist, 63)
            
            # Calculate volatility (average daily range as % of price)
            daily_range = (hist['High'] - hist['Low']) / hist['Close'] * 100
            volatility = daily_range.tail(20).mean()
            
            # Beta (if available)
            beta = info.get('beta')
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Basic fundamentals
            pe_ratio = info.get('forwardPE') or info.get('trailingPE')
            sector = info.get('sector', 'Unknown')
            
            # Run screening criteria
            passes, reasons = self._check_criteria(
                price=current_price,
                market_cap=market_cap,
                volume=current_volume,
                avg_volume=avg_volume,
                change_1w=change_1w,
                change_1m=change_1m,
                volatility=volatility,
                volume_ratio=volume_ratio
            )
            
            return ScreenerResult(
                symbol=symbol,
                price=current_price,
                market_cap=market_cap,
                volume=int(current_volume),
                avg_volume=int(avg_volume),
                change_1d=change_1d,
                change_1w=change_1w,
                change_1m=change_1m,
                change_3m=change_3m,
                volatility=volatility,
                beta=beta,
                volume_ratio=volume_ratio,
                pe_ratio=pe_ratio,
                sector=sector,
                passes_screen=passes,
                screen_reasons=reasons
            )
            
        except Exception as e:
            print(f"Error screening {symbol}: {e}")
            return None
    
    def _calc_change(self, hist: pd.DataFrame, days: int) -> float:
        """Calculate percentage change over given days"""
        if len(hist) < days + 1:
            return 0.0
        
        current = hist['Close'].iloc[-1]
        past = hist['Close'].iloc[-days-1]
        
        return ((current / past) - 1) * 100
    
    def _check_criteria(self, **kwargs) -> tuple:
        """Check if stock passes screening criteria"""
        passes = True
        reasons = []
        
        # Price filter
        if kwargs['price'] < self.config.min_price:
            passes = False
            reasons.append(f"Price ${kwargs['price']:.2f} below minimum ${self.config.min_price}")
        elif kwargs['price'] > self.config.max_price:
            passes = False
            reasons.append(f"Price ${kwargs['price']:.2f} above maximum ${self.config.max_price}")
        else:
            reasons.append(f"[+] Price ${kwargs['price']:.2f} in range")
        
        # Volume filter
        if kwargs['avg_volume'] < self.config.min_volume:
            passes = False
            reasons.append(f"Avg volume {kwargs['avg_volume']:,} below minimum {self.config.min_volume:,}")
        else:
            reasons.append(f"[+] Volume {kwargs['avg_volume']:,} sufficient")
        
        # Market cap filter
        if kwargs['market_cap'] < self.config.min_market_cap:
            passes = False
            reasons.append(f"Market cap ${kwargs['market_cap']/1e9:.2f}B below minimum")
        else:
            reasons.append(f"[+] Market cap ${kwargs['market_cap']/1e9:.2f}B sufficient")
        
        # Volatility filter (for high upside potential)
        if kwargs['volatility'] < self.config.min_volatility:
            reasons.append(f"Low volatility {kwargs['volatility']:.1f}% (less upside)")
        elif kwargs['volatility'] > self.config.max_volatility:
            reasons.append(f"[!] High volatility {kwargs['volatility']:.1f}% (risky)")
        else:
            reasons.append(f"[+] Good volatility {kwargs['volatility']:.1f}%")
        
        # Momentum - positive signals
        if kwargs['change_1w'] > self.config.min_momentum_1w:
            reasons.append(f"[+] Strong 1W momentum +{kwargs['change_1w']:.1f}%")
        
        if kwargs['change_1m'] > self.config.min_momentum_1m:
            reasons.append(f"[+] Strong 1M momentum +{kwargs['change_1m']:.1f}%")
        
        # Volume spike
        if kwargs['volume_ratio'] > 1.5:
            reasons.append(f"[+] Volume spike {kwargs['volume_ratio']:.1f}x average")
        
        return passes, reasons
    
    def screen_watchlist(self, symbols: List[str] = None, max_workers: int = 10) -> List[ScreenerResult]:
        """
        Screen multiple stocks in parallel
        
        Args:
            symbols: List of ticker symbols (uses DEFAULT_WATCHLIST if None)
            max_workers: Number of parallel threads
            
        Returns:
            List of ScreenerResult for stocks that pass criteria
        """
        if symbols is None:
            symbols = DEFAULT_WATCHLIST
        
        results = []
        
        print(f"\n[*] Screening {len(symbols)} stocks...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.screen_stock, symbol): symbol 
                for symbol in symbols
            }
            
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result and result.passes_screen:
                        results.append(result)
                        print(f"  [{completed}/{len(symbols)}] [OK] {symbol} - ${result.price:.2f}")
                    else:
                        print(f"  [{completed}/{len(symbols)}] [--] {symbol}")
                except Exception as e:
                    print(f"  [{completed}/{len(symbols)}] [--] {symbol} - Error: {e}")
        
        # Sort by momentum score
        results.sort(key=lambda x: x.change_1w + x.change_1m, reverse=True)
        
        return results
    
    def get_momentum_leaders(self, symbols: List[str] = None, top_n: int = 10) -> List[ScreenerResult]:
        """
        Get the top momentum stocks regardless of screening criteria
        
        Args:
            symbols: List of ticker symbols
            top_n: Number of top stocks to return
            
        Returns:
            List of top momentum stocks
        """
        if symbols is None:
            symbols = DEFAULT_WATCHLIST
        
        results = []
        
        print(f"\n[*] Finding momentum leaders in {len(symbols)} stocks...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self.screen_stock, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except:
                    pass
        
        # Sort by combined momentum
        results.sort(key=lambda x: x.change_1w + x.change_1m * 0.5, reverse=True)
        
        return results[:top_n]
    
    def find_breakouts(self, symbols: List[str] = None) -> List[ScreenerResult]:
        """
        Find stocks showing breakout characteristics:
        - Volume spike
        - Price near 52-week high
        - Strong recent momentum
        """
        if symbols is None:
            symbols = DEFAULT_WATCHLIST
        
        breakouts = []
        
        print(f"\n[*] Scanning for breakouts in {len(symbols)} stocks...")
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                
                if hist.empty or len(hist) < 50:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                year_high = hist['High'].max()
                current_volume = hist['Volume'].iloc[-1]
                avg_volume = hist['Volume'].tail(20).mean()
                
                # Check breakout conditions
                near_high = current_price > year_high * 0.95
                volume_spike = current_volume > avg_volume * 1.5
                momentum = (current_price / hist['Close'].iloc[-5] - 1) * 100 > 3
                
                if near_high and (volume_spike or momentum):
                    result = self.screen_stock(symbol)
                    if result:
                        result.screen_reasons.append(f"[!] BREAKOUT: {current_price/year_high*100:.1f}% of 52w high")
                        breakouts.append(result)
                        print(f"  [OK] {symbol} - Potential breakout!")
                        
            except Exception as e:
                continue
        
        return breakouts


def format_market_cap(value: float) -> str:
    """Format market cap for display"""
    if value >= 1e12:
        return f"${value/1e12:.1f}T"
    elif value >= 1e9:
        return f"${value/1e9:.1f}B"
    elif value >= 1e6:
        return f"${value/1e6:.0f}M"
    else:
        return f"${value:,.0f}"
