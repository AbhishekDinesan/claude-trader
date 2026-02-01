"""
Unified Stock Scanner
Combines technical analysis, screening, and sentiment for final scoring
"""

import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import SCORING_CONFIG, ScoringConfig, DEFAULT_WATCHLIST
from technical_analysis import TechnicalAnalyzer, TechnicalSignals, get_price_data
from screener import StockScreener, ScreenerResult
from sentiment import SentimentAnalyzer, SentimentResult


@dataclass
class StockOpportunity:
    """Complete analysis of a stock opportunity"""
    symbol: str
    
    # Scores (0-100)
    overall_score: float
    technical_score: float
    momentum_score: float
    sentiment_score: float
    
    # Signal
    signal: str  # 'STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    
    # Price data
    current_price: float
    target_upside: float  # Estimated % upside
    stop_loss: float  # Suggested stop loss %
    
    # Risk metrics
    risk_level: str  # 'HIGH', 'MEDIUM', 'LOW'
    volatility: float
    
    # Analysis components
    technical: Optional[TechnicalSignals]
    screener: Optional[ScreenerResult]
    sentiment: Optional[SentimentResult]
    
    # Key reasons
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)
    
    # Timestamp
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())


class UnifiedScanner:
    """Combines all analysis methods for comprehensive stock scanning"""
    
    def __init__(self, scoring_config: ScoringConfig = SCORING_CONFIG):
        self.scoring = scoring_config
        self.technical_analyzer = TechnicalAnalyzer()
        self.screener = StockScreener()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_stock(self, symbol: str, include_sentiment: bool = True) -> Optional[StockOpportunity]:
        """
        Complete analysis of a single stock
        
        Args:
            symbol: Stock ticker
            include_sentiment: Whether to include sentiment analysis (slower)
            
        Returns:
            StockOpportunity with all analysis
        """
        try:
            # Get price data
            df = get_price_data(symbol, period="6mo")
            if df is None or df.empty:
                return None
            
            current_price = df['Close'].iloc[-1]
            
            # Technical Analysis
            technical = self.technical_analyzer.analyze(df)
            
            # Screener Analysis
            screener = self.screener.screen_stock(symbol)
            
            # Sentiment Analysis (optional - takes time)
            sentiment = None
            if include_sentiment:
                sentiment = self.sentiment_analyzer.analyze_symbol(symbol)
            
            # Calculate scores
            technical_score = technical.technical_score if technical else 50.0
            
            momentum_score = self._calc_momentum_score(screener) if screener else 50.0
            
            sentiment_score = self._calc_sentiment_score(sentiment) if sentiment else 50.0
            
            # Calculate overall score (weighted)
            overall_score = (
                technical_score * 0.45 +
                momentum_score * 0.35 +
                sentiment_score * 0.20
            )
            
            # Determine signal
            signal, confidence = self._determine_signal(
                overall_score, technical, screener, sentiment
            )
            
            # Calculate risk metrics
            risk_level, volatility = self._assess_risk(df, screener)
            
            # Calculate target and stop loss
            target_upside, stop_loss = self._calc_targets(
                technical, screener, risk_level
            )
            
            # Collect bullish/bearish factors
            bullish, bearish = self._collect_factors(technical, screener, sentiment)
            
            return StockOpportunity(
                symbol=symbol,
                overall_score=overall_score,
                technical_score=technical_score,
                momentum_score=momentum_score,
                sentiment_score=sentiment_score,
                signal=signal,
                confidence=confidence,
                current_price=current_price,
                target_upside=target_upside,
                stop_loss=stop_loss,
                risk_level=risk_level,
                volatility=volatility,
                technical=technical,
                screener=screener,
                sentiment=sentiment,
                bullish_factors=bullish,
                bearish_factors=bearish
            )
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calc_momentum_score(self, screener: ScreenerResult) -> float:
        """Calculate momentum score from screener data"""
        score = 50.0
        
        # 1-day change
        if screener.change_1d > 5:
            score += 15
        elif screener.change_1d > 2:
            score += 10
        elif screener.change_1d > 0:
            score += 5
        elif screener.change_1d < -5:
            score -= 15
        elif screener.change_1d < -2:
            score -= 10
        
        # 1-week change
        if screener.change_1w > 15:
            score += 20
        elif screener.change_1w > 10:
            score += 15
        elif screener.change_1w > 5:
            score += 10
        elif screener.change_1w < -10:
            score -= 15
        
        # 1-month change
        if screener.change_1m > 30:
            score += 15
        elif screener.change_1m > 20:
            score += 10
        elif screener.change_1m > 10:
            score += 5
        elif screener.change_1m < -20:
            score -= 15
        
        # Volume spike bonus
        if screener.volume_ratio > 2.0:
            score += 10
        elif screener.volume_ratio > 1.5:
            score += 5
        
        return max(0, min(100, score))
    
    def _calc_sentiment_score(self, sentiment: SentimentResult) -> float:
        """Calculate sentiment score"""
        # Convert -1 to 1 sentiment to 0-100 score
        base_score = (sentiment.overall_sentiment + 1) * 50
        
        # Adjust for buzz (more mentions = more confidence)
        buzz_adjustment = min(10, sentiment.buzz_score * 0.1)
        
        # Trend adjustment
        if sentiment.sentiment_trend == 'improving':
            trend_adjustment = 5
        elif sentiment.sentiment_trend == 'declining':
            trend_adjustment = -5
        else:
            trend_adjustment = 0
        
        return max(0, min(100, base_score + buzz_adjustment + trend_adjustment))
    
    def _determine_signal(self, overall_score: float, technical: TechnicalSignals,
                          screener: ScreenerResult, sentiment: SentimentResult) -> tuple:
        """Determine trading signal and confidence"""
        # Signal based on score
        if overall_score >= self.scoring.strong_buy_threshold:
            signal = 'STRONG BUY'
        elif overall_score >= self.scoring.buy_threshold:
            signal = 'BUY'
        elif overall_score >= self.scoring.hold_threshold:
            signal = 'HOLD'
        elif overall_score >= 25:
            signal = 'SELL'
        else:
            signal = 'STRONG SELL'
        
        # Confidence based on alignment
        alignment_count = 0
        
        if technical and technical.technical_score > 60:
            alignment_count += 1
        if screener and screener.change_1w > 5:
            alignment_count += 1
        if sentiment and sentiment.overall_sentiment > 0.2:
            alignment_count += 1
        
        if alignment_count >= 3:
            confidence = 'HIGH'
        elif alignment_count >= 2:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return signal, confidence
    
    def _assess_risk(self, df: pd.DataFrame, screener: ScreenerResult) -> tuple:
        """Assess risk level"""
        # Calculate volatility
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * 100 * (252 ** 0.5)  # Annualized
        
        # Risk level based on volatility and beta
        if screener:
            if screener.volatility > 8 or (screener.beta and screener.beta > 2):
                risk_level = 'HIGH'
            elif screener.volatility > 4 or (screener.beta and screener.beta > 1.2):
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
        else:
            risk_level = 'MEDIUM'
        
        return risk_level, volatility
    
    def _calc_targets(self, technical: TechnicalSignals, screener: ScreenerResult,
                      risk_level: str) -> tuple:
        """Calculate target upside and stop loss"""
        # Base target on recent momentum and risk
        if screener:
            base_target = max(10, screener.change_1m * 0.5)
        else:
            base_target = 15
        
        # Adjust for technical signals
        if technical:
            if technical.trend_direction == 'bullish':
                base_target *= 1.2
            elif technical.trend_direction == 'bearish':
                base_target *= 0.7
            
            if technical.bb_squeeze:
                base_target *= 1.3  # Breakout potential
        
        # Stop loss based on risk
        if risk_level == 'HIGH':
            stop_loss = 15.0
        elif risk_level == 'MEDIUM':
            stop_loss = 10.0
        else:
            stop_loss = 7.0
        
        return min(50, base_target), stop_loss
    
    def _collect_factors(self, technical: TechnicalSignals, screener: ScreenerResult,
                         sentiment: SentimentResult) -> tuple:
        """Collect bullish and bearish factors"""
        bullish = []
        bearish = []
        
        # Technical factors
        if technical:
            if technical.rsi_signal == 'oversold':
                bullish.append("RSI oversold - potential bounce")
            elif technical.rsi_signal == 'overbought':
                bearish.append("RSI overbought - potential pullback")
            
            if technical.macd_crossover == 'bullish':
                bullish.append("MACD bullish crossover")
            elif technical.macd_crossover == 'bearish':
                bearish.append("MACD bearish crossover")
            
            if technical.golden_cross:
                bullish.append("Golden cross (MA bullish)")
            elif technical.death_cross:
                bearish.append("Death cross (MA bearish)")
            
            if technical.bb_squeeze:
                bullish.append("Bollinger squeeze - breakout imminent")
            
            if technical.volume_ratio > 1.5:
                bullish.append(f"High volume ({technical.volume_ratio:.1f}x avg)")
            
            if technical.trend_direction == 'bullish':
                bullish.append(f"Strong uptrend (strength: {technical.trend_strength:.0f})")
            elif technical.trend_direction == 'bearish':
                bearish.append(f"Downtrend (strength: {technical.trend_strength:.0f})")
        
        # Momentum factors
        if screener:
            if screener.change_1w > 10:
                bullish.append(f"Strong weekly momentum +{screener.change_1w:.1f}%")
            elif screener.change_1w < -10:
                bearish.append(f"Weak weekly momentum {screener.change_1w:.1f}%")
            
            if screener.change_1m > 20:
                bullish.append(f"Strong monthly gain +{screener.change_1m:.1f}%")
            elif screener.change_1m < -20:
                bearish.append(f"Monthly decline {screener.change_1m:.1f}%")
        
        # Sentiment factors
        if sentiment:
            if sentiment.sentiment_label in ['bullish', 'very_bullish']:
                bullish.append(f"Positive sentiment ({sentiment.reddit_mentions} Reddit mentions)")
            elif sentiment.sentiment_label in ['bearish', 'very_bearish']:
                bearish.append(f"Negative sentiment")
            
            if sentiment.buzz_score > 50:
                bullish.append(f"High social buzz (score: {sentiment.buzz_score:.0f})")
        
        return bullish, bearish
    
    def scan_watchlist(self, symbols: List[str] = None, include_sentiment: bool = True,
                       max_workers: int = 5) -> List[StockOpportunity]:
        """
        Scan entire watchlist and return sorted opportunities
        
        Args:
            symbols: List of tickers (uses DEFAULT_WATCHLIST if None)
            include_sentiment: Include sentiment analysis (slower)
            max_workers: Number of parallel threads
            
        Returns:
            List of StockOpportunity sorted by score
        """
        if symbols is None:
            symbols = DEFAULT_WATCHLIST
        
        opportunities = []
        
        print(f"\n{'='*60}")
        print(f"[COMPREHENSIVE STOCK SCANNER]")
        print(f"{'='*60}")
        print(f"Analyzing {len(symbols)} stocks...")
        print(f"Include sentiment: {include_sentiment}")
        print(f"{'='*60}\n")
        
        for i, symbol in enumerate(symbols):
            try:
                print(f"[{i+1}/{len(symbols)}] Analyzing {symbol}...")
                result = self.analyze_stock(symbol, include_sentiment)
                
                if result:
                    opportunities.append(result)
                    print(f"  -> Score: {result.overall_score:.1f} | "
                          f"Signal: {result.signal} | "
                          f"Price: ${result.current_price:.2f}")
                else:
                    print(f"  -> Skipped (insufficient data)")
                    
            except Exception as e:
                print(f"  -> Error: {e}")
        
        # Sort by overall score
        opportunities.sort(key=lambda x: x.overall_score, reverse=True)
        
        return opportunities
    
    def get_top_picks(self, symbols: List[str] = None, top_n: int = 5,
                      include_sentiment: bool = True) -> List[StockOpportunity]:
        """
        Get the top N stock picks
        
        Args:
            symbols: List of tickers
            top_n: Number of top picks to return
            include_sentiment: Include sentiment analysis
            
        Returns:
            Top N opportunities
        """
        all_opportunities = self.scan_watchlist(symbols, include_sentiment)
        
        # Filter for BUY signals only
        buy_signals = [o for o in all_opportunities if o.signal in ['STRONG BUY', 'BUY']]
        
        return buy_signals[:top_n]


def format_opportunity(opp: StockOpportunity) -> str:
    """Format opportunity for display"""
    lines = [
        f"\n{'='*60}",
        f"[{opp.symbol}] - {opp.signal} ({opp.confidence} confidence)",
        f"{'='*60}",
        f"",
        f"Price: ${opp.current_price:.2f}",
        f"Target Upside: +{opp.target_upside:.1f}%",
        f"Stop Loss: -{opp.stop_loss:.1f}%",
        f"Risk Level: {opp.risk_level}",
        f"",
        f"[SCORES]",
        f"   Overall:    {opp.overall_score:.1f}/100",
        f"   Technical:  {opp.technical_score:.1f}/100",
        f"   Momentum:   {opp.momentum_score:.1f}/100",
        f"   Sentiment:  {opp.sentiment_score:.1f}/100",
        f"",
    ]
    
    if opp.bullish_factors:
        lines.append("[+] BULLISH FACTORS")
        for factor in opp.bullish_factors[:5]:
            lines.append(f"    * {factor}")
        lines.append("")
    
    if opp.bearish_factors:
        lines.append("[!] BEARISH FACTORS")
        for factor in opp.bearish_factors[:5]:
            lines.append(f"    * {factor}")
        lines.append("")
    
    if opp.sentiment and opp.sentiment.reddit_posts:
        lines.append("[RECENT REDDIT MENTIONS]")
        for post in opp.sentiment.reddit_posts[:3]:
            sentiment_marker = "[+]" if post['sentiment'] > 0.1 else "[-]" if post['sentiment'] < -0.1 else "[=]"
            lines.append(f"   {sentiment_marker} r/{post['subreddit']}: {post['title'][:50]}...")
        lines.append("")
    
    return "\n".join(lines)
