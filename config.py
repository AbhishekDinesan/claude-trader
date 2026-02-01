"""
Configuration for the Stock Analysis Tool
All settings in one place for easy customization
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class TechnicalConfig:
    """Technical analysis parameters"""
    # Moving Averages
    sma_short: int = 10
    sma_medium: int = 20
    sma_long: int = 50
    ema_short: int = 12
    ema_long: int = 26
    
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Volume
    volume_sma_period: int = 20
    volume_spike_threshold: float = 2.0  # 2x average volume


@dataclass
class ScreenerConfig:
    """Stock screener parameters"""
    # Minimum criteria
    min_price: float = 1.0      # Avoid penny stocks under $1
    max_price: float = 100.0    # Focus on affordable stocks for $1000 account
    min_volume: int = 500000    # Minimum daily volume for liquidity
    min_market_cap: float = 100_000_000  # $100M minimum market cap
    
    # Momentum thresholds
    min_momentum_1w: float = 5.0   # Minimum 1-week gain %
    min_momentum_1m: float = 10.0  # Minimum 1-month gain %
    
    # Volatility (for high upside potential)
    min_volatility: float = 2.0   # Minimum daily volatility %
    max_volatility: float = 15.0  # Maximum to avoid extreme risk


@dataclass
class SentimentConfig:
    """Sentiment analysis parameters"""
    # Reddit subreddits to monitor
    subreddits: List[str] = field(default_factory=lambda: [
        'wallstreetbets',
        'stocks', 
        'investing',
        'stockmarket',
        'pennystocks',
        'smallstreetbets'
    ])
    
    # Time windows
    reddit_lookback_hours: int = 72  # Look at posts from last 3 days
    news_lookback_days: int = 7      # Look at news from last week
    
    # Sentiment thresholds
    bullish_threshold: float = 0.2   # Sentiment score > 0.2 is bullish
    bearish_threshold: float = -0.2  # Sentiment score < -0.2 is bearish


@dataclass
class ScoringConfig:
    """Scoring weights for final ranking"""
    # Technical weights (should sum to 1.0)
    weight_rsi: float = 0.15
    weight_macd: float = 0.15
    weight_trend: float = 0.20
    weight_volume: float = 0.15
    weight_bb: float = 0.10
    
    # Momentum weight
    weight_momentum: float = 0.15
    
    # Sentiment weight
    weight_sentiment: float = 0.10
    
    # Score thresholds
    strong_buy_threshold: float = 75.0
    buy_threshold: float = 60.0
    hold_threshold: float = 40.0


# Default configurations
TECHNICAL_CONFIG = TechnicalConfig()
SCREENER_CONFIG = ScreenerConfig()
SENTIMENT_CONFIG = SentimentConfig()
SCORING_CONFIG = ScoringConfig()

# Stock universe - popular stocks to scan
# Mix of growth stocks, meme stocks, and volatile names for high upside
DEFAULT_WATCHLIST = [
    # Tech Growth
    'NVDA', 'AMD', 'PLTR', 'SOFI', 'HOOD', 'COIN', 'MARA', 'RIOT',
    'IONQ', 'RGTI', 'QUBT', 'SMCI', 'ARM', 'CRWD', 'NET', 'DDOG',
    
    # EV & Clean Energy
    'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'PLUG', 'FCEL',
    'ENPH', 'SEDG', 'RUN', 'NOVA',
    
    # Biotech (high risk/reward)
    'MRNA', 'BNTX', 'NVAX', 'SAVA', 'SRPT', 'BMRN', 'EXAS', 'SANA',
    
    # Meme / High Volatility
    'GME', 'AMC', 'BBBY', 'BB', 'NOK', 'SPCE', 'WISH', 'CLOV',
    'TLRY', 'SNDL', 'CGC',
    
    # Financials
    'SQ', 'PYPL', 'AFRM', 'UPST', 'LC', 'NU',
    
    # Other Growth
    'SHOP', 'SNOW', 'ROKU', 'SNAP', 'PINS', 'TTD', 'RBLX', 'U',
    'PATH', 'DOCN', 'MDB', 'CFLT', 'GTLB',
    
    # SPACs & Recent IPOs (volatile)
    'DWAC', 'BKKT', 'DNA', 'JOBY', 'LILM',
]

# High momentum ETFs for reference
MOMENTUM_ETFS = ['TQQQ', 'SOXL', 'FNGU', 'LABU', 'TECL']
