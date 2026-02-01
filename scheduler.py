#!/usr/bin/env python3
"""
Automated Scheduler for Stock Analysis Tool
Runs the complete daily trading cycle automatically

Features:
- Daily stock scan with predictions
- Automatic paper trading execution
- Learning from past predictions
- New stock discovery
- Comprehensive daily journal
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_WATCHLIST
from daily_journal import DailyJournal, run_daily_journal
from paper_trading import PaperTradingPortfolio, get_portfolio_status
from learning import AutonomousLearner, show_learning_stats
from stock_discovery import run_discovery


# Results storage path
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def run_daily_cycle():
    """
    Run the complete daily trading cycle:
    1. Discover new stocks
    2. Scan stocks and log predictions
    3. Execute paper trades
    4. Run learning cycle
    5. Generate daily journal
    """
    print("\n" + "#"*60)
    print(f"# AUTOMATED DAILY TRADING CYCLE")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)
    
    # Run the full journal cycle (does everything)
    journal = DailyJournal()
    journal_file = journal.run_full_daily_cycle()
    
    # Also save a summary to results folder
    summary = {
        'date': journal.date,
        'timestamp': journal.timestamp,
        'portfolio': journal.capture_portfolio_snapshot(),
        'recommendations': [
            s for s in journal.scan_results 
            if s['signal'] in ['STRONG BUY', 'BUY']
        ][:5],
        'trades_today': journal.trades_made,
        'new_discoveries': len(journal.discoveries),
        'learning': journal.learning_insights
    }
    
    summary_file = RESULTS_DIR / f"daily_{journal.date}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")
    
    return summary


def show_portfolio_status():
    """Display current portfolio status"""
    status = get_portfolio_status()
    
    print("\n" + "="*60)
    print("[PORTFOLIO STATUS]")
    print("="*60)
    print(f"Started: {status['start_date']} with ${status['starting_capital']:,.2f}")
    print(f"\nCurrent Value: ${status['total_value']:,.2f}")
    print(f"  Cash: ${status['cash']:,.2f}")
    print(f"  Positions: ${status['positions_value']:,.2f}")
    print(f"\nTotal Return: ${status['total_return']:+,.2f} ({status['total_return_pct']:+.2f}%)")
    
    if status['positions']:
        print(f"\n[POSITIONS ({status['num_positions']})]")
        print("-"*60)
        for pos in status['positions']:
            print(f"  {pos['symbol']}: {pos['shares']:.2f} shares @ ${pos['entry_price']:.2f}")
            print(f"    Current: ${pos['current_price']:.2f} | P&L: ${pos['pnl']:+.2f} ({pos['pnl_percent']:+.1f}%)")
    else:
        print("\nNo open positions")


def show_recent_journals():
    """Show recent journal entries"""
    print("\n[RECENT JOURNALS]")
    print("="*60)
    
    journal_files = sorted(LOGS_DIR.glob("journal_*.md"), reverse=True)[:7]
    
    if not journal_files:
        print("No journals yet. Run 'python scheduler.py daily' to generate.")
        return
    
    for jf in journal_files:
        date = jf.stem.replace("journal_", "")
        
        # Load corresponding JSON for stats
        json_file = LOGS_DIR / f"journal_{date}.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            
            portfolio = data.get('portfolio', {})
            value = portfolio.get('total_value', 1000)
            ret = portfolio.get('total_return_pct', 0)
            trades = len(data.get('trades', []))
            
            print(f"  {date}: ${value:,.2f} ({ret:+.2f}%) | {trades} trades")
        else:
            print(f"  {date}: (no data)")


def generate_status_badge():
    """Generate a status badge for README"""
    status = get_portfolio_status()
    
    badge = {
        'portfolio_value': round(status['total_value'], 2),
        'total_return_pct': round(status['total_return_pct'], 2),
        'positions': status['num_positions'],
        'last_updated': datetime.now().isoformat()
    }
    
    # Get learning stats
    try:
        from learning import PredictionTracker
        tracker = PredictionTracker()
        learn_stats = tracker.get_stats()
        badge['predictions_tracked'] = learn_stats['total_predictions']
        badge['win_rate'] = round(learn_stats['win_rate'] * 100, 1)
    except:
        pass
    
    badge_file = RESULTS_DIR / "status.json"
    with open(badge_file, 'w') as f:
        json.dump(badge, f, indent=2)
    
    return badge


def main():
    parser = argparse.ArgumentParser(
        description="Automated Stock Trading Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scheduler.py daily      # Run full daily cycle (scan, trade, learn, journal)
  python scheduler.py status     # Show portfolio status
  python scheduler.py journals   # Show recent journal entries
  python scheduler.py learn      # Run learning cycle only
  python scheduler.py discover   # Run stock discovery only
        """
    )
    parser.add_argument(
        'action',
        choices=['daily', 'status', 'journals', 'learn', 'discover'],
        help='Action to perform'
    )
    
    args = parser.parse_args()
    
    if args.action == 'daily':
        run_daily_cycle()
        generate_status_badge()
        
    elif args.action == 'status':
        show_portfolio_status()
        
    elif args.action == 'journals':
        show_recent_journals()
        
    elif args.action == 'learn':
        learner = AutonomousLearner()
        learner.run_learning_cycle(min_eval_days=5)
        generate_status_badge()
        
    elif args.action == 'discover':
        results = run_discovery()
        print(f"\nDiscovered {len(results['new_discoveries'])} new stocks")
        if results['top_recommendations']:
            print("\nTop recommendations from discoveries:")
            for stock in results['top_recommendations'][:5]:
                print(f"  {stock['symbol']}: +{stock.get('change_since_discovery', 0):.1f}% since discovery")


if __name__ == "__main__":
    main()
