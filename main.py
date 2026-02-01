#!/usr/bin/env python3
"""
Stock Analysis Tool - Main CLI Interface
Find high-potential stocks using technical analysis and sentiment

Usage:
    python main.py scan              # Scan default watchlist
    python main.py scan --fast       # Quick scan without sentiment
    python main.py analyze TSLA      # Analyze single stock
    python main.py top 10            # Get top 10 picks
    python main.py momentum          # Find momentum leaders
    python main.py breakouts         # Find breakout candidates
"""

import argparse
import sys
from datetime import datetime
from typing import List

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from config import DEFAULT_WATCHLIST, SCORING_CONFIG
from scanner import UnifiedScanner, StockOpportunity, format_opportunity
from screener import StockScreener, format_market_cap
from sentiment import SentimentAnalyzer, get_trending_tickers

# ML Model integration
try:
    from ml_model import (
        TradingMLModel, 
        HistoricalDataCollector, 
        train_model_from_scratch,
        format_ml_prediction
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Learning system integration
try:
    from learning import (
        AutonomousLearner,
        run_learning_cycle,
        show_learning_stats
    )
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False


def print_banner():
    """Print welcome banner"""
    banner = """
    +===============================================================+
    |                                                               |
    |   STOCK ANALYSIS TOOL v1.0                                    |
    |   Technical Analysis - Momentum Screening - Sentiment         |
    |                                                               |
    |   Goal: Find high-upside opportunities for your $1000         |
    |   Strategy: Technical + Momentum + Social Sentiment           |
    |                                                               |
    +===============================================================+
    """
    print(banner)


def print_disclaimer():
    """Print risk disclaimer"""
    print("""
    [!] DISCLAIMER: This tool is for educational purposes only.
    Trading involves substantial risk of loss. Past performance
    does not guarantee future results. Always do your own research.
    """)


def create_results_table(opportunities: List[StockOpportunity]) -> None:
    """Create and print results table"""
    if RICH_AVAILABLE:
        console = Console()
        
        table = Table(title="Stock Analysis Results", show_lines=True)
        
        table.add_column("Rank", style="cyan", justify="center")
        table.add_column("Symbol", style="bold white")
        table.add_column("Price", style="green", justify="right")
        table.add_column("Signal", style="bold", justify="center")
        table.add_column("Score", justify="center")
        table.add_column("Tech", justify="center")
        table.add_column("Mom", justify="center")
        table.add_column("Sent", justify="center")
        table.add_column("Risk", justify="center")
        table.add_column("Upside", style="green", justify="right")
        
        for i, opp in enumerate(opportunities[:20], 1):
            # Format signal with color
            if opp.signal == 'STRONG BUY':
                signal_text = "[bold green]STRONG BUY[/bold green]"
            elif opp.signal == 'BUY':
                signal_text = "[green]BUY[/green]"
            elif opp.signal == 'HOLD':
                signal_text = "[yellow]HOLD[/yellow]"
            elif opp.signal == 'SELL':
                signal_text = "[red]SELL[/red]"
            elif opp.signal == 'STRONG SELL':
                signal_text = "[bold red]STRONG SELL[/bold red]"
            else:
                signal_text = opp.signal
            
            # Format risk with color
            if opp.risk_level == 'HIGH':
                risk_text = "[red]HIGH[/red]"
            elif opp.risk_level == 'MEDIUM':
                risk_text = "[yellow]MEDIUM[/yellow]"
            elif opp.risk_level == 'LOW':
                risk_text = "[green]LOW[/green]"
            else:
                risk_text = opp.risk_level
            
            table.add_row(
                str(i),
                opp.symbol,
                f"${opp.current_price:.2f}",
                signal_text,
                f"{opp.overall_score:.0f}",
                f"{opp.technical_score:.0f}",
                f"{opp.momentum_score:.0f}",
                f"{opp.sentiment_score:.0f}",
                risk_text,
                f"+{opp.target_upside:.0f}%"
            )
        
        console.print(table)
    else:
        # Fallback plain text
        print("\n" + "="*100)
        print("[STOCK ANALYSIS RESULTS]")
        print("="*100)
        print(f"{'Rank':<5} {'Symbol':<8} {'Price':<10} {'Signal':<12} {'Score':<7} "
              f"{'Tech':<6} {'Mom':<6} {'Sent':<6} {'Risk':<8} {'Upside':<8}")
        print("-"*100)
        
        for i, opp in enumerate(opportunities[:20], 1):
            print(f"{i:<5} {opp.symbol:<8} ${opp.current_price:<9.2f} {opp.signal:<12} "
                  f"{opp.overall_score:<7.0f} {opp.technical_score:<6.0f} "
                  f"{opp.momentum_score:<6.0f} {opp.sentiment_score:<6.0f} "
                  f"{opp.risk_level:<8} +{opp.target_upside:.0f}%")
        
        print("="*100)


def print_detailed_picks(opportunities: List[StockOpportunity], top_n: int = 5):
    """Print detailed analysis for top picks"""
    print("\n" + "="*60)
    print("[TOP PICKS - DETAILED ANALYSIS]")
    print("="*60)
    
    for opp in opportunities[:top_n]:
        print(format_opportunity(opp))


def cmd_scan(args):
    """Run full watchlist scan"""
    print_banner()
    
    scanner = UnifiedScanner()
    
    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = DEFAULT_WATCHLIST
    
    include_sentiment = not args.fast
    
    print(f"\n[*] Scanning {len(symbols)} stocks...")
    print(f"    Mode: {'Fast (no sentiment)' if args.fast else 'Full analysis'}")
    print(f"    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    opportunities = scanner.scan_watchlist(
        symbols=symbols,
        include_sentiment=include_sentiment
    )
    
    if opportunities:
        create_results_table(opportunities)
        
        # Show top picks in detail
        buy_signals = [o for o in opportunities if o.signal in ['STRONG BUY', 'BUY']]
        if buy_signals:
            print_detailed_picks(buy_signals, top_n=args.top)
        
        # Log predictions for learning (if --track flag is set)
        if getattr(args, 'track', False) and LEARNING_AVAILABLE:
            print("\n[*] Logging predictions for learning...")
            learner = AutonomousLearner()
            logged = 0
            for opp in opportunities:
                if learner.log_prediction_from_opportunity(opp):
                    logged += 1
            print(f"    Logged {logged} predictions for future evaluation")
        
        # Summary
        print("\n" + "="*60)
        print("[+] SUMMARY")
        print("="*60)
        print(f"    Total analyzed: {len(opportunities)}")
        print(f"    Strong Buy: {len([o for o in opportunities if o.signal == 'STRONG BUY'])}")
        print(f"    Buy: {len([o for o in opportunities if o.signal == 'BUY'])}")
        print(f"    Hold: {len([o for o in opportunities if o.signal == 'HOLD'])}")
        print(f"    Sell/Strong Sell: {len([o for o in opportunities if 'SELL' in o.signal])}")
    else:
        print("\n[X] No opportunities found matching criteria.")
    
    print_disclaimer()


def cmd_analyze(args):
    """Analyze a single stock"""
    print_banner()
    
    symbol = args.symbol.upper()
    print(f"\n[*] Analyzing {symbol}...")
    
    scanner = UnifiedScanner()
    result = scanner.analyze_stock(symbol, include_sentiment=True)
    
    if result:
        print(format_opportunity(result))
        
        # Additional technical details
        if result.technical:
            tech = result.technical
            print("\n[TECHNICAL INDICATORS]")
            print(f"   RSI: {tech.rsi:.1f} ({tech.rsi_signal})")
            print(f"   MACD: {tech.macd:.4f} (Signal: {tech.macd_signal_line:.4f})")
            print(f"   MACD Histogram: {tech.macd_histogram:.4f}")
            print(f"   Trend: {tech.trend_direction} (strength: {tech.trend_strength:.0f})")
            print(f"   Above SMA20: {'Yes' if tech.above_sma20 else 'No'}")
            print(f"   Above SMA50: {'Yes' if tech.above_sma50 else 'No'}")
            print(f"   Bollinger Position: {tech.bb_position:.2f} ({tech.bb_signal})")
            print(f"   Volume Ratio: {tech.volume_ratio:.2f}x")
        
        # Screener details
        if result.screener:
            scr = result.screener
            print("\n[MOMENTUM DATA]")
            print(f"   1-Day Change: {scr.change_1d:+.2f}%")
            print(f"   1-Week Change: {scr.change_1w:+.2f}%")
            print(f"   1-Month Change: {scr.change_1m:+.2f}%")
            print(f"   3-Month Change: {scr.change_3m:+.2f}%")
            print(f"   Daily Volatility: {scr.volatility:.2f}%")
            print(f"   Market Cap: {format_market_cap(scr.market_cap)}")
            print(f"   Sector: {scr.sector}")
        
        # Sentiment details
        if result.sentiment:
            sent = result.sentiment
            print("\n[SENTIMENT ANALYSIS]")
            print(f"   Overall: {sent.overall_sentiment:.2f} ({sent.sentiment_label})")
            print(f"   Reddit Sentiment: {sent.reddit_sentiment:.2f}")
            print(f"   Reddit Mentions: {sent.reddit_mentions}")
            print(f"   News Sentiment: {sent.news_sentiment:.2f}")
            print(f"   News Articles: {sent.news_articles}")
            print(f"   Buzz Score: {sent.buzz_score:.0f}/100")
            print(f"   Trend: {sent.sentiment_trend}")
            
            if sent.news_headlines:
                print("\n   Recent Headlines:")
                for headline in sent.news_headlines[:5]:
                    print(f"   • {headline[:70]}...")
    else:
        print(f"\n[X] Could not analyze {symbol}. Check if the ticker is valid.")
    
    print_disclaimer()


def cmd_top(args):
    """Get top N picks"""
    print_banner()
    
    n = args.n
    print(f"\n[*] Finding top {n} stock picks...")
    
    scanner = UnifiedScanner()
    top_picks = scanner.get_top_picks(top_n=n, include_sentiment=not args.fast)
    
    if top_picks:
        create_results_table(top_picks)
        print_detailed_picks(top_picks, top_n=n)
    else:
        print(f"\n[X] No strong buy signals found.")
    
    print_disclaimer()


def cmd_momentum(args):
    """Find momentum leaders"""
    print_banner()
    
    print("\n[*] Finding momentum leaders...")
    
    screener = StockScreener()
    leaders = screener.get_momentum_leaders(top_n=args.top)
    
    if leaders:
        print("\n" + "="*80)
        print("[MOMENTUM LEADERS]")
        print("="*80)
        print(f"{'Rank':<5} {'Symbol':<8} {'Price':<10} {'1D':<8} {'1W':<8} "
              f"{'1M':<8} {'3M':<10} {'Vol Ratio':<10}")
        print("-"*80)
        
        for i, stock in enumerate(leaders, 1):
            print(f"{i:<5} {stock.symbol:<8} ${stock.price:<9.2f} "
                  f"{stock.change_1d:+.1f}%{'':<3} {stock.change_1w:+.1f}%{'':<3} "
                  f"{stock.change_1m:+.1f}%{'':<3} {stock.change_3m:+.1f}%{'':<5} "
                  f"{stock.volume_ratio:.1f}x")
        
        print("="*80)
    else:
        print("\n[X] Could not find momentum data.")
    
    print_disclaimer()


def cmd_breakouts(args):
    """Find breakout candidates"""
    print_banner()
    
    print("\n[*] Scanning for breakout candidates...")
    
    screener = StockScreener()
    breakouts = screener.find_breakouts()
    
    if breakouts:
        print("\n" + "="*80)
        print("[BREAKOUT CANDIDATES]")
        print("="*80)
        
        for stock in breakouts[:10]:
            print(f"\n{stock.symbol} - ${stock.price:.2f}")
            for reason in stock.screen_reasons:
                if "BREAKOUT" in reason or "✓" in reason:
                    print(f"   {reason}")
        
        print("\n" + "="*80)
    else:
        print("\n[X] No breakout candidates found at this time.")
    
    print_disclaimer()


def cmd_trending(args):
    """Show trending tickers"""
    print_banner()
    
    print("\n[*] Finding trending tickers...")
    
    trending = get_trending_tickers()
    
    if trending:
        print(f"\nTrending tickers: {', '.join(trending)}")
        
        if args.analyze:
            print("\nAnalyzing trending stocks...")
            scanner = UnifiedScanner()
            opportunities = scanner.scan_watchlist(
                symbols=trending[:15],
                include_sentiment=False
            )
            
            if opportunities:
                create_results_table(opportunities)
    else:
        print("\n[X] Could not fetch trending tickers.")
    
    print_disclaimer()


def cmd_portfolio(args):
    """Portfolio analysis for $1000"""
    print_banner()
    
    print("\n[PORTFOLIO BUILDER FOR $1,000]")
    print("="*60)
    
    scanner = UnifiedScanner()
    
    print("\nFinding best opportunities...")
    top_picks = scanner.get_top_picks(top_n=10, include_sentiment=not args.fast)
    
    if not top_picks:
        print("\n[X] No strong opportunities found.")
        return
    
    # Build suggested portfolio
    budget = 1000
    num_positions = min(5, len(top_picks))  # Max 5 positions for diversification
    position_size = budget / num_positions
    
    print(f"\n[SUGGESTED PORTFOLIO (${budget})]")
    print("="*60)
    print(f"Strategy: {num_positions} positions @ ~${position_size:.0f} each")
    print("-"*60)
    
    total_invested = 0
    portfolio = []
    
    for opp in top_picks[:num_positions]:
        shares = int(position_size / opp.current_price)
        if shares < 1:
            shares = 1
        
        cost = shares * opp.current_price
        target_value = cost * (1 + opp.target_upside / 100)
        stop_value = cost * (1 - opp.stop_loss / 100)
        
        portfolio.append({
            'symbol': opp.symbol,
            'shares': shares,
            'price': opp.current_price,
            'cost': cost,
            'target': target_value,
            'stop': stop_value,
            'signal': opp.signal,
            'risk': opp.risk_level
        })
        
        total_invested += cost
    
    # Display portfolio
    print(f"\n{'Symbol':<8} {'Shares':<8} {'Price':<10} {'Cost':<10} "
          f"{'Target':<10} {'Stop':<10} {'Signal':<12}")
    print("-"*70)
    
    for p in portfolio:
        print(f"{p['symbol']:<8} {p['shares']:<8} ${p['price']:<9.2f} "
              f"${p['cost']:<9.2f} ${p['target']:<9.2f} ${p['stop']:<9.2f} "
              f"{p['signal']:<12}")
    
    print("-"*70)
    print(f"{'TOTAL':<8} {'':<8} {'':<10} ${total_invested:<9.2f}")
    print(f"{'CASH':<8} {'':<8} {'':<10} ${budget - total_invested:<9.2f}")
    
    # Potential outcomes
    best_case = sum(p['target'] for p in portfolio)
    worst_case = sum(p['stop'] for p in portfolio)
    
    print("\n[POTENTIAL OUTCOMES]")
    print(f"   Best case (all hit targets): ${best_case:.2f} ({(best_case/total_invested-1)*100:+.1f}%)")
    print(f"   Worst case (all hit stops):  ${worst_case:.2f} ({(worst_case/total_invested-1)*100:+.1f}%)")
    
    print("\n[!] RISK MANAGEMENT")
    print("    * Set stop losses immediately after buying")
    print("    * Never risk more than you can afford to lose")
    print("    * Consider scaling in (buy half now, half on dips)")
    print("    * Take profits at targets, don't get greedy")
    
    print_disclaimer()


def cmd_ml(args):
    """ML model commands"""
    print_banner()
    
    if not ML_AVAILABLE:
        print("\n[!] ML dependencies not installed!")
        print("    Run: pip install xgboost lightgbm scikit-learn")
        return
    
    if args.action == 'train':
        cmd_ml_train(args)
    elif args.action == 'status':
        cmd_ml_status(args)
    elif args.action == 'predict':
        cmd_ml_predict(args)
    else:
        print(f"\n[?] Unknown ML action: {args.action}")


def cmd_ml_train(args):
    """Train the ML model"""
    print("\n" + "="*60)
    print("[ML MODEL TRAINING]")
    print("="*60)
    
    symbols = DEFAULT_WATCHLIST[:args.symbols] if args.symbols else DEFAULT_WATCHLIST[:30]
    
    print(f"Training on {len(symbols)} symbols...")
    print(f"This may take several minutes...\n")
    
    try:
        model = train_model_from_scratch(symbols=symbols)
        
        print("\n" + "="*60)
        print("[TRAINING COMPLETE]")
        print("="*60)
        
        if model.training_metrics:
            metrics = model.training_metrics
            print(f"\n[Model Performance]")
            print(f"   Accuracy:    {metrics.accuracy*100:.1f}%")
            print(f"   Precision:   {metrics.precision*100:.1f}%")
            print(f"   Win Rate:    {metrics.win_rate*100:.1f}%")
            print(f"   Avg Return:  {metrics.avg_return:.2f}%")
            print(f"   Sharpe:      {metrics.sharpe_ratio:.2f}")
            print(f"   Total Trades:{metrics.total_trades}")
        
        print("\n[+] Model saved to models/trading_model.pkl")
        print("    The model will now be used for all stock analysis.")
        
    except Exception as e:
        print(f"\n[!] Training failed: {e}")


def cmd_ml_status(args):
    """Show ML model status"""
    print("\n" + "="*60)
    print("[ML MODEL STATUS]")
    print("="*60)
    
    model = TradingMLModel()
    loaded = model.load()
    
    if not loaded:
        print("\n[!] No trained model found.")
        print("    Run 'python main.py ml train' to train a model.")
        return
    
    print(f"\n[Model Info]")
    print(f"   Version:     {model.MODEL_VERSION}")
    print(f"   Status:      {'Trained' if model.is_trained else 'Not Trained'}")
    print(f"   Horizon:     {model.prediction_horizon} days")
    print(f"   Models:      {', '.join(model.models.keys())}")
    
    if model.training_metrics:
        metrics = model.training_metrics
        print(f"\n[Performance Metrics]")
        print(f"   Accuracy:    {metrics.accuracy*100:.1f}%")
        print(f"   Precision:   {metrics.precision*100:.1f}%")
        print(f"   Recall:      {metrics.recall*100:.1f}%")
        print(f"   F1 Score:    {metrics.f1_score*100:.1f}%")
        print(f"   AUC-ROC:     {metrics.auc_roc:.3f}")
        print(f"   Win Rate:    {metrics.win_rate*100:.1f}%")
        print(f"   Avg Return:  {metrics.avg_return:.2f}%")
        print(f"   Sharpe:      {metrics.sharpe_ratio:.2f}")
    
    if model.feature_importance:
        print(f"\n[Top Features]")
        sorted_features = sorted(model.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            print(f"   {feature}: {importance:.4f}")


def cmd_ml_predict(args):
    """Get ML prediction for a symbol"""
    symbol = args.symbol.upper()
    
    print(f"\n[ML PREDICTION: {symbol}]")
    print("="*60)
    
    model = TradingMLModel()
    if not model.load():
        print("\n[!] No trained model. Run 'python main.py ml train' first.")
        return
    
    from technical_analysis import TechnicalAnalyzer, get_price_data
    
    print(f"Analyzing {symbol}...")
    
    df = get_price_data(symbol, period="6mo")
    if df is None or df.empty:
        print(f"\n[!] Could not get data for {symbol}")
        return
    
    tech_analyzer = TechnicalAnalyzer()
    technical = tech_analyzer.analyze(df)
    
    prediction = model.predict(
        price_data=df,
        technical_signals=technical,
        screener_result=None,
        sentiment_result=None,
        symbol=symbol
    )
    
    if prediction:
        print(format_ml_prediction(prediction))
    else:
        print(f"\n[!] Could not generate prediction for {symbol}")
    
    print_disclaimer()


def cmd_learn(args):
    """Learning system commands"""
    if not LEARNING_AVAILABLE:
        print("\n[!] Learning module not available. Check learning.py imports.")
        return
    
    print_banner()
    
    if args.action == 'cycle':
        # Run a full learning cycle
        print("\n[RUNNING LEARNING CYCLE]")
        print("This will evaluate past predictions and learn from them...")
        
        learner = AutonomousLearner()
        results = learner.run_learning_cycle(min_eval_days=args.days)
        
        if results['stats']['total_predictions'] == 0:
            print("\n[!] No predictions to learn from yet.")
            print("    Run scans with --track flag to start logging predictions:")
            print("    python main.py scan --fast --track")
    
    elif args.action == 'stats':
        # Show learning statistics
        show_learning_stats()
    
    elif args.action == 'weights':
        # Show current learned weights
        from learning import AdaptiveWeightLearner, PredictionTracker
        
        tracker = PredictionTracker()
        learner = AdaptiveWeightLearner(tracker)
        weights = learner.load_learned_weights()
        
        print("\n[LEARNED WEIGHTS]")
        print("="*60)
        
        if weights:
            print(f"\nCurrent scoring weights (learned from past predictions):")
            print(f"    Technical Analysis: {weights.get('technical', 0.45)*100:.1f}%")
            print(f"    Momentum Score:     {weights.get('momentum', 0.35)*100:.1f}%")
            print(f"    Sentiment Score:    {weights.get('sentiment', 0.20)*100:.1f}%")
            
            if 'tech_accuracy' in weights:
                print(f"\nIndicator Accuracy:")
                print(f"    Technical: {weights.get('tech_accuracy', 0.5)*100:.1f}%")
                print(f"    Sentiment: {weights.get('sentiment_accuracy', 0.5)*100:.1f}%")
        else:
            print("\n  No learned weights yet (using defaults):")
            print(f"    Technical: 45%")
            print(f"    Momentum:  35%")
            print(f"    Sentiment: 20%")
            print("\n  Run 'python main.py scan --fast --track' to log predictions,")
            print("  then 'python main.py learn cycle' to learn from them.")
    
    print_disclaimer()


def main():
    parser = argparse.ArgumentParser(
        description="Stock Analysis Tool - Find high-potential trading opportunities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan                    # Full scan of watchlist
  python main.py scan --fast             # Quick scan (no sentiment)
  python main.py scan --fast --track     # Scan and log for learning
  python main.py scan -s "TSLA,NVDA,AMD" # Scan specific stocks
  python main.py analyze TSLA            # Deep dive on one stock
  python main.py top 5                   # Get top 5 picks
  python main.py momentum                # Find momentum leaders
  python main.py breakouts               # Find breakout candidates
  python main.py portfolio               # Build $1000 portfolio

Learning Commands (self-improvement):
  python main.py learn cycle             # Evaluate past predictions & learn
  python main.py learn stats             # Show learning statistics  
  python main.py learn weights           # Show current learned weights

ML Commands:
  python main.py ml train                # Train the ML model
  python main.py ml train --symbols 50   # Train on 50 symbols
  python main.py ml status               # Show model status/metrics
  python main.py ml predict TSLA         # Get ML prediction for a stock
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan watchlist for opportunities')
    scan_parser.add_argument('-s', '--symbols', help='Comma-separated symbols to scan')
    scan_parser.add_argument('--fast', action='store_true', help='Skip sentiment analysis')
    scan_parser.add_argument('--top', type=int, default=5, help='Number of top picks to show')
    scan_parser.add_argument('--track', action='store_true', help='Log predictions for learning')
    scan_parser.set_defaults(func=cmd_scan)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single stock')
    analyze_parser.add_argument('symbol', help='Stock ticker symbol')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Top command
    top_parser = subparsers.add_parser('top', help='Get top N stock picks')
    top_parser.add_argument('n', type=int, default=5, nargs='?', help='Number of picks')
    top_parser.add_argument('--fast', action='store_true', help='Skip sentiment analysis')
    top_parser.set_defaults(func=cmd_top)
    
    # Momentum command
    momentum_parser = subparsers.add_parser('momentum', help='Find momentum leaders')
    momentum_parser.add_argument('--top', type=int, default=10, help='Number to show')
    momentum_parser.set_defaults(func=cmd_momentum)
    
    # Breakouts command
    breakouts_parser = subparsers.add_parser('breakouts', help='Find breakout candidates')
    breakouts_parser.set_defaults(func=cmd_breakouts)
    
    # Trending command
    trending_parser = subparsers.add_parser('trending', help='Show trending tickers')
    trending_parser.add_argument('--analyze', action='store_true', help='Analyze trending stocks')
    trending_parser.set_defaults(func=cmd_trending)
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Build suggested portfolio')
    portfolio_parser.add_argument('--fast', action='store_true', help='Skip sentiment analysis')
    portfolio_parser.set_defaults(func=cmd_portfolio)
    
    # ML commands
    ml_parser = subparsers.add_parser('ml', help='Machine Learning model commands')
    ml_parser.add_argument('action', choices=['train', 'status', 'predict'], 
                          help='ML action: train, status, or predict')
    ml_parser.add_argument('symbol', nargs='?', help='Symbol for prediction (required for predict)')
    ml_parser.add_argument('--symbols', type=int, default=30, 
                          help='Number of symbols to train on (default: 30)')
    ml_parser.set_defaults(func=cmd_ml)
    
    # Learning commands
    learn_parser = subparsers.add_parser('learn', help='Autonomous learning system')
    learn_parser.add_argument('action', choices=['cycle', 'stats', 'weights'], 
                             help='cycle=run learning, stats=show stats, weights=show learned weights')
    learn_parser.add_argument('--days', type=int, default=5, 
                             help='Min days before evaluating predictions (default: 5)')
    learn_parser.set_defaults(func=cmd_learn)
    
    args = parser.parse_args()
    
    if args.command is None:
        # Default to portfolio command for the user's goal
        print_banner()
        print("\n[Quick Start]")
        print("    python main.py portfolio    - Build a $1000 portfolio")
        print("    python main.py scan --fast  - Quick market scan")
        print("    python main.py analyze TSLA - Analyze any stock")
        print("\nRun 'python main.py --help' for all options.")
        print_disclaimer()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
