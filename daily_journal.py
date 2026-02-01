"""
Daily Trading Journal
Creates a comprehensive daily log of all activity, learnings, and performance

Features:
- Documents daily recommendations and reasoning
- Tracks portfolio performance
- Records what the system learned
- Logs new stock discoveries
- Creates readable markdown reports
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from paper_trading import PaperTradingPortfolio, get_portfolio_status
from learning import PredictionTracker, AdaptiveWeightLearner, AutonomousLearner
from stock_discovery import StockDiscovery, run_discovery
from scanner import UnifiedScanner
from config import DEFAULT_WATCHLIST


# Log directory
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class DailyJournal:
    """Creates and manages daily trading journals"""
    
    def __init__(self):
        self.date = datetime.now().strftime('%Y-%m-%d')
        self.timestamp = datetime.now().isoformat()
        
        # Components
        self.portfolio = PaperTradingPortfolio()
        self.scanner = UnifiedScanner()
        self.learner = AutonomousLearner()
        self.discovery = StockDiscovery()
        
        # Today's data
        self.scan_results = []
        self.trades_made = []
        self.discoveries = []
        self.learning_insights = {}
        self.portfolio_snapshot = None
    
    def run_daily_scan(self, symbols: List[str] = None) -> List[Dict]:
        """Run the daily stock scan"""
        if symbols is None:
            # Use default watchlist + discovered stocks
            symbols = DEFAULT_WATCHLIST[:40]
            discovered = list(self.discovery.discovered.get('stocks', {}).keys())[:10]
            symbols = list(set(symbols + discovered))
        
        print(f"\n[DAILY SCAN] Analyzing {len(symbols)} stocks...")
        
        opportunities = self.scanner.scan_watchlist(
            symbols=symbols,
            include_sentiment=False
        )
        
        # Log predictions for learning
        for opp in opportunities:
            self.learner.log_prediction_from_opportunity(opp)
        
        # Store results
        self.scan_results = [
            {
                'symbol': o.symbol,
                'signal': o.signal,
                'score': round(o.overall_score, 1),
                'technical_score': round(o.technical_score, 1),
                'momentum_score': round(o.momentum_score, 1),
                'price': round(o.current_price, 2),
                'target_upside': round(o.target_upside, 1),
                'stop_loss': round(o.stop_loss, 1),
                'risk_level': o.risk_level,
                'bullish_factors': o.bullish_factors[:3],
                'bearish_factors': o.bearish_factors[:3]
            }
            for o in opportunities
        ]
        
        return self.scan_results
    
    def execute_trades(self) -> List[Dict]:
        """Execute trades based on scan results"""
        print("\n[TRADE EXECUTION]")
        
        # Get current portfolio state
        cash, _, total_value = self.portfolio.get_total_value()
        positions = self.portfolio.get_positions()
        position_symbols = {p.symbol for p in positions}
        
        trades = []
        
        # 1. Check if we should sell any positions
        for position in positions:
            # Find signal for this position
            signal_data = next(
                (s for s in self.scan_results if s['symbol'] == position.symbol),
                None
            )
            
            if signal_data:
                # Sell if signal turned negative or hit stop loss
                if signal_data['signal'] in ['SELL', 'STRONG SELL']:
                    trade = self.portfolio.sell(
                        position.symbol,
                        reason=f"Signal changed to {signal_data['signal']}"
                    )
                    if trade:
                        trades.append({
                            'action': 'SELL',
                            'symbol': position.symbol,
                            'shares': trade.shares,
                            'price': trade.price,
                            'reason': trade.reason
                        })
                
                # Also sell if we've hit our target
                elif position.pnl_percent >= signal_data['target_upside']:
                    trade = self.portfolio.sell(
                        position.symbol,
                        reason=f"Hit target: +{position.pnl_percent:.1f}%"
                    )
                    if trade:
                        trades.append({
                            'action': 'SELL',
                            'symbol': position.symbol,
                            'shares': trade.shares,
                            'price': trade.price,
                            'reason': trade.reason
                        })
        
        # 2. Buy new positions for BUY signals
        buy_signals = [
            s for s in self.scan_results 
            if s['signal'] in ['STRONG BUY', 'BUY'] 
            and s['symbol'] not in position_symbols
            and s['score'] >= 60
        ]
        
        # Sort by score
        buy_signals.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate position size (split cash among top picks)
        num_buys = min(3, len(buy_signals))  # Max 3 new positions per day
        
        if num_buys > 0 and cash > 100:
            position_size = min(cash / num_buys, total_value * 0.20)  # Max 20% per position
            
            for signal in buy_signals[:num_buys]:
                if cash >= 50:  # Minimum trade size
                    reason = f"{signal['signal']} (Score: {signal['score']})"
                    if signal['bullish_factors']:
                        reason += f" - {signal['bullish_factors'][0]}"
                    
                    trade = self.portfolio.buy(
                        signal['symbol'],
                        amount=min(position_size, cash),
                        reason=reason,
                        signal_score=signal['score']
                    )
                    
                    if trade:
                        trades.append({
                            'action': 'BUY',
                            'symbol': signal['symbol'],
                            'shares': trade.shares,
                            'price': trade.price,
                            'reason': trade.reason
                        })
                        
                        # Update cash
                        cash, _, _ = self.portfolio.get_total_value()
        
        self.trades_made = trades
        return trades
    
    def run_discovery(self) -> List[Dict]:
        """Run stock discovery for new opportunities"""
        print("\n[STOCK DISCOVERY]")
        
        discovery_results = run_discovery()
        
        self.discoveries = discovery_results['new_discoveries']
        
        # If we found good stocks, add them to next scan
        for stock in self.discoveries[:3]:  # Top 3
            self.discovery.add_to_watchlist(stock['symbol'])
            print(f"  Added {stock['symbol']} to watchlist")
        
        return self.discoveries
    
    def run_learning(self) -> Dict:
        """Run learning cycle and capture insights"""
        print("\n[LEARNING CYCLE]")
        
        results = self.learner.run_learning_cycle(min_eval_days=5)
        
        # Extract key insights
        insights = {
            'predictions_evaluated': results['evaluated_this_cycle'],
            'total_predictions': results['stats']['total_predictions'],
            'win_rate': results['stats']['win_rate']
        }
        
        if 'metrics' in results:
            metrics = results['metrics']
            insights['signal_performance'] = {
                'strong_buy_win_rate': metrics.get('win_rate_strong_buy', 0),
                'buy_win_rate': metrics.get('win_rate_buy', 0),
                'avg_return_buy': metrics.get('avg_return_buy', 0)
            }
            insights['indicator_accuracy'] = {
                'rsi': metrics.get('rsi_accuracy', 0.5),
                'macd': metrics.get('macd_accuracy', 0.5),
                'trend': metrics.get('trend_accuracy', 0.5),
                'sentiment': metrics.get('sentiment_accuracy', 0.5)
            }
            insights['suggested_weights'] = metrics.get('suggested_weights', {})
        
        self.learning_insights = insights
        return insights
    
    def capture_portfolio_snapshot(self) -> Dict:
        """Capture current portfolio state"""
        self.portfolio_snapshot = self.portfolio.take_daily_snapshot()
        return {
            'date': self.portfolio_snapshot.date,
            'total_value': round(self.portfolio_snapshot.total_value, 2),
            'cash': round(self.portfolio_snapshot.cash, 2),
            'positions_value': round(self.portfolio_snapshot.positions_value, 2),
            'daily_change': round(self.portfolio_snapshot.daily_change, 2),
            'daily_change_pct': round(self.portfolio_snapshot.daily_change_pct, 2),
            'total_return': round(self.portfolio_snapshot.total_return, 2),
            'total_return_pct': round(self.portfolio_snapshot.total_return_pct, 2),
            'vs_spy': round(self.portfolio_snapshot.total_return_pct - self.portfolio_snapshot.spy_return_pct, 2),
            'positions': self.portfolio_snapshot.positions
        }
    
    def generate_markdown_report(self) -> str:
        """Generate a markdown report for today"""
        snapshot = self.capture_portfolio_snapshot()
        
        # Header
        report = f"""# Daily Trading Journal - {self.date}

## Portfolio Performance

| Metric | Value |
|--------|-------|
| **Total Value** | ${snapshot['total_value']:,.2f} |
| **Cash** | ${snapshot['cash']:,.2f} |
| **Positions** | ${snapshot['positions_value']:,.2f} |
| **Daily Change** | ${snapshot['daily_change']:+,.2f} ({snapshot['daily_change_pct']:+.2f}%) |
| **Total Return** | ${snapshot['total_return']:+,.2f} ({snapshot['total_return_pct']:+.2f}%) |
| **vs S&P 500** | {snapshot['vs_spy']:+.2f}% |

"""
        
        # Current Positions
        if snapshot['positions']:
            report += "## Current Positions\n\n"
            report += "| Symbol | Shares | Entry | Current | P&L | P&L % |\n"
            report += "|--------|--------|-------|---------|-----|-------|\n"
            
            for pos in snapshot['positions']:
                report += f"| {pos['symbol']} | {pos['shares']:.2f} | ${pos['entry_price']:.2f} | ${pos['current_price']:.2f} | ${pos['pnl']:+.2f} | {pos['pnl_percent']:+.1f}% |\n"
            report += "\n"
        else:
            report += "## Current Positions\n\n*No open positions*\n\n"
        
        # Today's Trades
        if self.trades_made:
            report += "## Today's Trades\n\n"
            for trade in self.trades_made:
                emoji = "BUY" if trade['action'] == 'BUY' else "SELL"
                report += f"- **{emoji} {trade['symbol']}**: {trade['shares']:.2f} shares @ ${trade['price']:.2f}\n"
                report += f"  - Reason: {trade['reason']}\n"
            report += "\n"
        else:
            report += "## Today's Trades\n\n*No trades executed today*\n\n"
        
        # Top Recommendations
        buy_signals = [s for s in self.scan_results if s['signal'] in ['STRONG BUY', 'BUY']][:5]
        if buy_signals:
            report += "## Top Recommendations\n\n"
            report += "| Symbol | Signal | Score | Price | Target | Why |\n"
            report += "|--------|--------|-------|-------|--------|-----|\n"
            
            for sig in buy_signals:
                why = sig['bullish_factors'][0] if sig['bullish_factors'] else "Multiple factors"
                report += f"| {sig['symbol']} | {sig['signal']} | {sig['score']} | ${sig['price']:.2f} | +{sig['target_upside']:.0f}% | {why[:30]} |\n"
            report += "\n"
        
        # Learning Insights
        if self.learning_insights:
            report += "## What I Learned Today\n\n"
            
            win_rate = self.learning_insights.get('win_rate', 0) * 100
            report += f"- **Overall Win Rate**: {win_rate:.1f}%\n"
            report += f"- **Predictions Tracked**: {self.learning_insights.get('total_predictions', 0)}\n"
            
            if 'signal_performance' in self.learning_insights:
                perf = self.learning_insights['signal_performance']
                report += f"- **BUY Signal Win Rate**: {perf.get('buy_win_rate', 0)*100:.1f}%\n"
                report += f"- **Avg Return on BUY**: {perf.get('avg_return_buy', 0):+.1f}%\n"
            
            if 'indicator_accuracy' in self.learning_insights:
                acc = self.learning_insights['indicator_accuracy']
                report += "\n**Indicator Accuracy:**\n"
                for indicator, accuracy in acc.items():
                    report += f"- {indicator.upper()}: {accuracy*100:.1f}%\n"
            
            if 'suggested_weights' in self.learning_insights:
                weights = self.learning_insights['suggested_weights']
                if weights:
                    report += "\n**Optimized Weights:**\n"
                    report += f"- Technical: {weights.get('technical', 0.45)*100:.0f}%\n"
                    report += f"- Momentum: {weights.get('momentum', 0.35)*100:.0f}%\n"
                    report += f"- Sentiment: {weights.get('sentiment', 0.20)*100:.0f}%\n"
            
            report += "\n"
        
        # New Discoveries
        if self.discoveries:
            report += "## New Stocks Discovered\n\n"
            report += "| Symbol | 1M Change | Price | Sector |\n"
            report += "|--------|-----------|-------|--------|\n"
            
            for stock in self.discoveries[:5]:
                report += f"| {stock['symbol']} | +{stock['change_1m']:.1f}% | ${stock['price']:.2f} | {stock.get('sector', 'N/A')} |\n"
            report += "\n"
        
        # Footer
        report += f"""---
*Generated at {self.timestamp}*
*This is a paper trading simulation - not real money*
"""
        
        return report
    
    def save_journal(self) -> str:
        """Save the journal to a file"""
        # Generate markdown
        md_content = self.generate_markdown_report()
        
        # Save markdown file
        md_file = LOGS_DIR / f"journal_{self.date}.md"
        with open(md_file, 'w') as f:
            f.write(md_content)
        
        # Also save JSON data for programmatic access
        json_data = {
            'date': self.date,
            'timestamp': self.timestamp,
            'portfolio': self.capture_portfolio_snapshot(),
            'scan_results': self.scan_results,
            'trades': self.trades_made,
            'discoveries': self.discoveries,
            'learning': self.learning_insights
        }
        
        json_file = LOGS_DIR / f"journal_{self.date}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Update the main log file (append-only history)
        self._update_history_log(json_data)
        
        print(f"\nJournal saved to {md_file}")
        return str(md_file)
    
    def _update_history_log(self, data: Dict):
        """Update the running history log"""
        history_file = LOGS_DIR / "portfolio_history.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {'entries': [], 'start_date': self.date}
        
        # Add today's summary
        history['entries'].append({
            'date': data['date'],
            'total_value': data['portfolio']['total_value'],
            'return_pct': data['portfolio']['total_return_pct'],
            'trades': len(data['trades']),
            'positions': len(data['portfolio']['positions'])
        })
        
        history['last_updated'] = self.timestamp
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def run_full_daily_cycle(self) -> str:
        """Run the complete daily cycle"""
        print("\n" + "="*60)
        print(f"DAILY TRADING CYCLE - {self.date}")
        print("="*60)
        
        # 1. Run discovery for new stocks
        self.run_discovery()
        
        # 2. Run stock scan
        self.run_daily_scan()
        
        # 3. Execute trades
        self.execute_trades()
        
        # 4. Run learning
        self.run_learning()
        
        # 5. Save journal
        journal_file = self.save_journal()
        
        # Print summary
        snapshot = self.capture_portfolio_snapshot()
        print("\n" + "="*60)
        print("DAILY SUMMARY")
        print("="*60)
        print(f"Portfolio Value: ${snapshot['total_value']:,.2f}")
        print(f"Daily Change: ${snapshot['daily_change']:+,.2f} ({snapshot['daily_change_pct']:+.2f}%)")
        print(f"Total Return: ${snapshot['total_return']:+,.2f} ({snapshot['total_return_pct']:+.2f}%)")
        print(f"vs S&P 500: {snapshot['vs_spy']:+.2f}%")
        print(f"Trades Today: {len(self.trades_made)}")
        print(f"New Stocks Found: {len(self.discoveries)}")
        print(f"\nFull journal: {journal_file}")
        
        return journal_file


def run_daily_journal():
    """Run the daily journal - main entry point"""
    journal = DailyJournal()
    return journal.run_full_daily_cycle()


if __name__ == "__main__":
    run_daily_journal()
