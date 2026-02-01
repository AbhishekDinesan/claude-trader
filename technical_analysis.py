"""
Technical Analysis Engine
Calculates all major technical indicators for stock analysis
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from config import TECHNICAL_CONFIG, TechnicalConfig


@dataclass
class TechnicalSignals:
    """Container for all technical signals"""
    # Trend
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: float  # 0-100
    
    # RSI
    rsi: float
    rsi_signal: str  # 'oversold', 'overbought', 'neutral'
    
    # MACD
    macd: float
    macd_signal_line: float
    macd_histogram: float
    macd_crossover: str  # 'bullish', 'bearish', 'none'
    
    # Bollinger Bands
    bb_position: float  # 0 = lower band, 1 = upper band
    bb_signal: str  # 'oversold', 'overbought', 'neutral'
    bb_squeeze: bool  # True if bands are tight (potential breakout)
    
    # Moving Averages
    above_sma20: bool
    above_sma50: bool
    golden_cross: bool  # Short MA crossed above long MA
    death_cross: bool   # Short MA crossed below long MA
    
    # Volume
    volume_ratio: float  # Current vs average
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    
    # Support/Resistance
    near_support: bool
    near_resistance: bool
    
    # Overall Score (0-100)
    technical_score: float


class TechnicalAnalyzer:
    """Calculates technical indicators and generates signals"""
    
    def __init__(self, config: TechnicalConfig = TECHNICAL_CONFIG):
        self.config = config
    
    def analyze(self, df: pd.DataFrame) -> Optional[TechnicalSignals]:
        """
        Analyze a stock's price data and return technical signals
        
        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                Index should be DatetimeIndex
        
        Returns:
            TechnicalSignals object with all indicators
        """
        if df is None or len(df) < 50:
            return None
        
        try:
            # Calculate all indicators
            df = self._add_moving_averages(df)
            df = self._add_rsi(df)
            df = self._add_macd(df)
            df = self._add_bollinger_bands(df)
            df = self._add_volume_indicators(df)
            
            # Generate signals from latest data
            return self._generate_signals(df)
        except Exception as e:
            print(f"Error in technical analysis: {e}")
            return None
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA and EMA indicators"""
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA10'] = df['Close'].rolling(window=self.config.sma_short).mean()
        df['SMA20'] = df['Close'].rolling(window=self.config.sma_medium).mean()
        df['SMA50'] = df['Close'].rolling(window=self.config.sma_long).mean()
        
        # Exponential Moving Averages
        df['EMA12'] = df['Close'].ewm(span=self.config.ema_short, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=self.config.ema_long, adjust=False).mean()
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        df = df.copy()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        df = df.copy()
        
        ema_fast = df['Close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df = df.copy()
        
        df['BB_Middle'] = df['Close'].rolling(window=self.config.bb_period).mean()
        bb_std = df['Close'].rolling(window=self.config.bb_period).std()
        
        df['BB_Upper'] = df['BB_Middle'] + (self.config.bb_std * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (self.config.bb_std * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        df = df.copy()
        
        df['Volume_SMA'] = df['Volume'].rolling(window=self.config.volume_sma_period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> TechnicalSignals:
        """Generate trading signals from calculated indicators"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Trend Analysis
        trend_direction, trend_strength = self._analyze_trend(df)
        
        # RSI Signal
        rsi = latest['RSI']
        if rsi < self.config.rsi_oversold:
            rsi_signal = 'oversold'
        elif rsi > self.config.rsi_overbought:
            rsi_signal = 'overbought'
        else:
            rsi_signal = 'neutral'
        
        # MACD Crossover
        macd_crossover = 'none'
        if prev['MACD'] < prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
            macd_crossover = 'bullish'
        elif prev['MACD'] > prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']:
            macd_crossover = 'bearish'
        
        # Bollinger Band Position
        bb_range = latest['BB_Upper'] - latest['BB_Lower']
        bb_position = (latest['Close'] - latest['BB_Lower']) / bb_range if bb_range > 0 else 0.5
        
        if bb_position < 0.2:
            bb_signal = 'oversold'
        elif bb_position > 0.8:
            bb_signal = 'overbought'
        else:
            bb_signal = 'neutral'
        
        # Bollinger Squeeze (low volatility, potential breakout)
        avg_width = df['BB_Width'].rolling(50).mean().iloc[-1]
        bb_squeeze = latest['BB_Width'] < avg_width * 0.5
        
        # Moving Average Signals
        above_sma20 = latest['Close'] > latest['SMA20']
        above_sma50 = latest['Close'] > latest['SMA50']
        
        # Golden/Death Cross (check last 5 days)
        recent = df.tail(5)
        golden_cross = any(
            (recent['SMA20'].iloc[i-1] < recent['SMA50'].iloc[i-1]) and 
            (recent['SMA20'].iloc[i] > recent['SMA50'].iloc[i])
            for i in range(1, len(recent))
        )
        death_cross = any(
            (recent['SMA20'].iloc[i-1] > recent['SMA50'].iloc[i-1]) and 
            (recent['SMA20'].iloc[i] < recent['SMA50'].iloc[i])
            for i in range(1, len(recent))
        )
        
        # Volume Analysis
        volume_ratio = latest['Volume_Ratio']
        vol_trend = df['Volume_Ratio'].tail(5).mean()
        if vol_trend > 1.2:
            volume_trend = 'increasing'
        elif vol_trend < 0.8:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'stable'
        
        # Support/Resistance (simplified)
        recent_low = df['Low'].tail(20).min()
        recent_high = df['High'].tail(20).max()
        near_support = latest['Close'] < recent_low * 1.02
        near_resistance = latest['Close'] > recent_high * 0.98
        
        # Calculate Overall Technical Score
        technical_score = self._calculate_score(
            rsi=rsi,
            rsi_signal=rsi_signal,
            macd_crossover=macd_crossover,
            macd_histogram=latest['MACD_Histogram'],
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            bb_signal=bb_signal,
            bb_squeeze=bb_squeeze,
            volume_ratio=volume_ratio,
            above_sma20=above_sma20,
            above_sma50=above_sma50,
            golden_cross=golden_cross
        )
        
        return TechnicalSignals(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            rsi=rsi,
            rsi_signal=rsi_signal,
            macd=latest['MACD'],
            macd_signal_line=latest['MACD_Signal'],
            macd_histogram=latest['MACD_Histogram'],
            macd_crossover=macd_crossover,
            bb_position=bb_position,
            bb_signal=bb_signal,
            bb_squeeze=bb_squeeze,
            above_sma20=above_sma20,
            above_sma50=above_sma50,
            golden_cross=golden_cross,
            death_cross=death_cross,
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            near_support=near_support,
            near_resistance=near_resistance,
            technical_score=technical_score
        )
    
    def _analyze_trend(self, df: pd.DataFrame) -> tuple:
        """Analyze overall trend direction and strength"""
        close = df['Close']
        sma20 = df['SMA20']
        sma50 = df['SMA50']
        
        # Calculate trend based on moving averages and price action
        latest_close = close.iloc[-1]
        latest_sma20 = sma20.iloc[-1]
        latest_sma50 = sma50.iloc[-1]
        
        # Price position relative to MAs
        above_20 = latest_close > latest_sma20
        above_50 = latest_close > latest_sma50
        ma_aligned = latest_sma20 > latest_sma50
        
        # Price momentum (rate of change)
        roc_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100
        roc_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100
        
        # Determine trend direction
        if above_20 and above_50 and ma_aligned and roc_5d > 0:
            direction = 'bullish'
        elif not above_20 and not above_50 and not ma_aligned and roc_5d < 0:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Calculate trend strength (0-100)
        strength_factors = [
            25 if above_20 else 0,
            25 if above_50 else 0,
            25 if ma_aligned else 0,
            min(25, max(0, roc_5d * 2.5)) if roc_5d > 0 else 0
        ]
        strength = sum(strength_factors)
        
        return direction, strength
    
    def _calculate_score(self, **kwargs) -> float:
        """Calculate overall technical score from 0-100"""
        score = 50.0  # Start neutral
        
        # RSI contribution (-15 to +15)
        rsi = kwargs['rsi']
        if kwargs['rsi_signal'] == 'oversold':
            score += 15  # Potential bounce
        elif kwargs['rsi_signal'] == 'overbought':
            score -= 10  # Risk of pullback
        elif 40 <= rsi <= 60:
            score += 5  # Healthy range
        
        # MACD contribution (-15 to +15)
        if kwargs['macd_crossover'] == 'bullish':
            score += 15
        elif kwargs['macd_crossover'] == 'bearish':
            score -= 15
        elif kwargs['macd_histogram'] > 0:
            score += 5
        else:
            score -= 5
        
        # Trend contribution (-20 to +20)
        if kwargs['trend_direction'] == 'bullish':
            score += kwargs['trend_strength'] * 0.2
        elif kwargs['trend_direction'] == 'bearish':
            score -= kwargs['trend_strength'] * 0.2
        
        # Bollinger Bands contribution (-10 to +10)
        if kwargs['bb_signal'] == 'oversold':
            score += 10
        elif kwargs['bb_signal'] == 'overbought':
            score -= 5
        
        if kwargs['bb_squeeze']:
            score += 10  # Potential breakout
        
        # Volume contribution (-5 to +10)
        if kwargs['volume_ratio'] > 2.0:
            score += 10  # High volume confirms move
        elif kwargs['volume_ratio'] < 0.5:
            score -= 5  # Low volume weakness
        
        # Moving Average signals
        if kwargs['above_sma20']:
            score += 5
        if kwargs['above_sma50']:
            score += 5
        if kwargs['golden_cross']:
            score += 10
        
        return max(0, min(100, score))


def get_price_data(symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """
    Fetch price data for a symbol using yfinance
    
    Args:
        symbol: Stock ticker symbol
        period: Time period (1mo, 3mo, 6mo, 1y, 2y)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            return None
        
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
