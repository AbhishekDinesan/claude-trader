#!/usr/bin/env python3
"""
Automated Scheduler for Stock Analysis Tool
Runs scans and learning cycles automatically

Can be triggered by:
- GitHub Actions (scheduled)
- Cron job
- Task scheduler
- Manual run
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_WATCHLIST
from scanner import UnifiedScanner
from learning import AutonomousLearner, PredictionTracker, show_learning_stats


# Results storage path
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_daily_scan(symbols: list = None, track: bool = True):
    """
    Run daily stock scan and optionally track predictions
    
    Args:
        symbols: List of symbols to scan (uses default watchlist if None)
        track: Whether to log predictions for learning
    """
    print("\n" + "="*60)
    print(f"[AUTOMATED DAILY SCAN]")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if symbols is None:
        # Use a subset for faster automated runs
        symbols = DEFAULT_WATCHLIST[:30]
    
    scanner = UnifiedScanner()
    
    print(f"\nScanning {len(symbols)} stocks...")
    
    opportunities = scanner.scan_watchlist(
        symbols=symbols,
        include_sentiment=False  # Skip sentiment for speed
    )
    
    if not opportunities:
        print("No opportunities found.")
        return None
    
    # Log predictions for learning
    logged_count = 0
    if track:
        print("\nLogging predictions for learning...")
        learner = AutonomousLearner()
        for opp in opportunities:
            if learner.log_prediction_from_opportunity(opp):
                logged_count += 1
        print(f"Logged {logged_count} predictions")
    
    # Get top picks
    buy_signals = [o for o in opportunities if o.signal in ['STRONG BUY', 'BUY']]
    
    # Save results to file
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_scanned': len(opportunities),
        'predictions_logged': logged_count,
        'top_picks': [
            {
                'symbol': o.symbol,
                'signal': o.signal,
                'score': round(o.overall_score, 1),
                'price': round(o.current_price, 2),
                'target_upside': round(o.target_upside, 1),
                'stop_loss': round(o.stop_loss, 1),
                'technical_score': round(o.technical_score, 1),
                'momentum_score': round(o.momentum_score, 1),
                'risk_level': o.risk_level
            }
            for o in buy_signals[:10]
        ],
        'summary': {
            'strong_buy': len([o for o in opportunities if o.signal == 'STRONG BUY']),
            'buy': len([o for o in opportunities if o.signal == 'BUY']),
            'hold': len([o for o in opportunities if o.signal == 'HOLD']),
            'sell': len([o for o in opportunities if 'SELL' in o.signal])
        }
    }
    
    # Save daily results
    date_str = datetime.now().strftime('%Y-%m-%d')
    results_file = RESULTS_DIR / f"scan_{date_str}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("[SCAN SUMMARY]")
    print("="*60)
    print(f"Strong Buy: {results['summary']['strong_buy']}")
    print(f"Buy: {results['summary']['buy']}")
    print(f"Hold: {results['summary']['hold']}")
    print(f"Sell: {results['summary']['sell']}")
    
    if buy_signals:
        print("\n[TOP PICKS]")
        for i, pick in enumerate(results['top_picks'][:5], 1):
            print(f"  {i}. {pick['symbol']}: {pick['signal']} "
                  f"(Score: {pick['score']}, Price: ${pick['price']})")
    
    return results


def run_weekly_learning():
    """
    Run weekly learning cycle to evaluate predictions and adjust weights
    """
    print("\n" + "="*60)
    print(f"[AUTOMATED WEEKLY LEARNING]")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    learner = AutonomousLearner()
    
    # Run learning cycle
    results = learner.run_learning_cycle(min_eval_days=5)
    
    # Save learning results
    date_str = datetime.now().strftime('%Y-%m-%d')
    learning_file = RESULTS_DIR / f"learning_{date_str}.json"
    
    learning_data = {
        'timestamp': datetime.now().isoformat(),
        'evaluated_this_cycle': results['evaluated_this_cycle'],
        'total_predictions': results['stats']['total_predictions'],
        'total_evaluated': results['stats']['evaluated'],
        'win_rate': results['stats']['win_rate']
    }
    
    if 'metrics' in results:
        learning_data['metrics'] = results['metrics']
    
    with open(learning_file, 'w') as f:
        json.dump(learning_data, f, indent=2)
    
    print(f"\nLearning results saved to {learning_file}")
    
    return results


def run_full_cycle():
    """
    Run both scan and learning (for weekly comprehensive run)
    """
    print("\n" + "#"*60)
    print(f"# FULL AUTOMATED CYCLE")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)
    
    # Run scan
    scan_results = run_daily_scan(track=True)
    
    # Run learning
    learning_results = run_weekly_learning()
    
    # Generate summary report
    report = {
        'timestamp': datetime.now().isoformat(),
        'scan': scan_results,
        'learning': {
            'evaluated': learning_results['evaluated_this_cycle'],
            'total_tracked': learning_results['stats']['total_predictions'],
            'win_rate': learning_results['stats']['win_rate']
        }
    }
    
    # Save combined report
    date_str = datetime.now().strftime('%Y-%m-%d')
    report_file = RESULTS_DIR / f"report_{date_str}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "#"*60)
    print("# CYCLE COMPLETE")
    print("#"*60)
    
    return report


def generate_readme_badge():
    """Generate a badge/status for README showing latest stats"""
    tracker = PredictionTracker()
    stats = tracker.get_stats()
    
    badge_data = {
        'predictions': stats['total_predictions'],
        'win_rate': f"{stats['win_rate']*100:.1f}%",
        'last_updated': datetime.now().isoformat()
    }
    
    badge_file = RESULTS_DIR / "status.json"
    with open(badge_file, 'w') as f:
        json.dump(badge_data, f, indent=2)
    
    return badge_data


def main():
    parser = argparse.ArgumentParser(
        description="Automated Stock Analysis Scheduler"
    )
    parser.add_argument(
        'action',
        choices=['scan', 'learn', 'full', 'status'],
        help='Action to perform: scan (daily), learn (weekly), full (both), status (show stats)'
    )
    parser.add_argument(
        '--symbols',
        type=int,
        default=30,
        help='Number of symbols to scan (default: 30)'
    )
    
    args = parser.parse_args()
    
    if args.action == 'scan':
        symbols = DEFAULT_WATCHLIST[:args.symbols]
        run_daily_scan(symbols=symbols, track=True)
        generate_readme_badge()
        
    elif args.action == 'learn':
        run_weekly_learning()
        generate_readme_badge()
        
    elif args.action == 'full':
        run_full_cycle()
        generate_readme_badge()
        
    elif args.action == 'status':
        show_learning_stats()
        badge = generate_readme_badge()
        print(f"\nStatus badge updated: {badge}")


if __name__ == "__main__":
    main()
