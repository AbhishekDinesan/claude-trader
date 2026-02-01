"""
Paper Trading Simulator
Simulates a $1000 portfolio based on system recommendations

Features:
- Tracks virtual portfolio with real prices
- Executes trades based on signals
- Calculates daily P&L and total return
- Maintains trade history
- Compares against S&P 500 benchmark
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import yfinance as yf

# Database path
PORTFOLIO_DB = os.path.join(os.path.dirname(__file__), 'portfolio.db')
PORTFOLIO_LOG = os.path.join(os.path.dirname(__file__), 'logs', 'portfolio_history.json')


@dataclass
class Position:
    """A single stock position"""
    symbol: str
    shares: float
    entry_price: float
    entry_date: str
    entry_reason: str
    current_price: float = 0.0
    current_value: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0


@dataclass
class Trade:
    """Record of a trade"""
    id: str
    timestamp: str
    symbol: str
    action: str  # BUY or SELL
    shares: float
    price: float
    total: float
    reason: str
    signal_score: float


@dataclass
class PortfolioSnapshot:
    """Daily portfolio snapshot"""
    date: str
    cash: float
    positions_value: float
    total_value: float
    daily_change: float
    daily_change_pct: float
    total_return: float
    total_return_pct: float
    spy_return_pct: float  # Benchmark comparison
    positions: List[Dict]
    trades_today: List[Dict]


class PaperTradingPortfolio:
    """Manages the paper trading portfolio"""
    
    STARTING_CAPITAL = 1000.0
    MAX_POSITION_SIZE = 0.25  # Max 25% in one stock
    MIN_TRADE_SIZE = 50.0  # Minimum $50 per trade
    
    def __init__(self, db_path: str = PORTFOLIO_DB):
        self.db_path = db_path
        self._init_database()
        self._load_state()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolio state
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY,
                cash REAL,
                starting_capital REAL,
                start_date TEXT,
                last_updated TEXT
            )
        ''')
        
        # Current positions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                shares REAL,
                entry_price REAL,
                entry_date TEXT,
                entry_reason TEXT
            )
        ''')
        
        # Trade history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                shares REAL,
                price REAL,
                total REAL,
                reason TEXT,
                signal_score REAL
            )
        ''')
        
        # Daily snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS snapshots (
                date TEXT PRIMARY KEY,
                cash REAL,
                positions_value REAL,
                total_value REAL,
                daily_change REAL,
                daily_change_pct REAL,
                total_return REAL,
                total_return_pct REAL,
                spy_return_pct REAL,
                positions_json TEXT,
                trades_json TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_state(self):
        """Load portfolio state from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM portfolio_state WHERE id = 1')
        row = cursor.fetchone()
        
        if row:
            self.cash = row[1]
            self.starting_capital = row[2]
            self.start_date = row[3]
        else:
            # Initialize new portfolio
            self.cash = self.STARTING_CAPITAL
            self.starting_capital = self.STARTING_CAPITAL
            self.start_date = datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute('''
                INSERT INTO portfolio_state (id, cash, starting_capital, start_date, last_updated)
                VALUES (1, ?, ?, ?, ?)
            ''', (self.cash, self.starting_capital, self.start_date, datetime.now().isoformat()))
            conn.commit()
        
        conn.close()
    
    def _save_state(self):
        """Save portfolio state to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE portfolio_state 
            SET cash = ?, last_updated = ?
            WHERE id = 1
        ''', (self.cash, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_positions(self) -> List[Position]:
        """Get all current positions with updated prices"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM positions')
        rows = cursor.fetchall()
        conn.close()
        
        positions = []
        for row in rows:
            pos = Position(
                symbol=row[0],
                shares=row[1],
                entry_price=row[2],
                entry_date=row[3],
                entry_reason=row[4]
            )
            
            # Get current price
            try:
                ticker = yf.Ticker(pos.symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    pos.current_price = hist['Close'].iloc[-1]
                    pos.current_value = pos.shares * pos.current_price
                    pos.pnl = pos.current_value - (pos.shares * pos.entry_price)
                    pos.pnl_percent = (pos.current_price / pos.entry_price - 1) * 100
            except:
                pos.current_price = pos.entry_price
                pos.current_value = pos.shares * pos.entry_price
            
            positions.append(pos)
        
        return positions
    
    def get_total_value(self) -> Tuple[float, float, float]:
        """Get total portfolio value (cash, positions, total)"""
        positions = self.get_positions()
        positions_value = sum(p.current_value for p in positions)
        total = self.cash + positions_value
        return self.cash, positions_value, total
    
    def buy(self, symbol: str, amount: float, reason: str, signal_score: float) -> Optional[Trade]:
        """
        Buy a stock
        
        Args:
            symbol: Stock ticker
            amount: Dollar amount to invest
            reason: Why buying this stock
            signal_score: The signal score that triggered this
        
        Returns:
            Trade record if successful
        """
        # Validate
        if amount > self.cash:
            print(f"  [!] Not enough cash. Have ${self.cash:.2f}, need ${amount:.2f}")
            return None
        
        if amount < self.MIN_TRADE_SIZE:
            print(f"  [!] Trade size ${amount:.2f} below minimum ${self.MIN_TRADE_SIZE}")
            return None
        
        # Check position size limit
        _, _, total_value = self.get_total_value()
        if amount > total_value * self.MAX_POSITION_SIZE:
            amount = total_value * self.MAX_POSITION_SIZE
            print(f"  [!] Capping position size to ${amount:.2f} (25% max)")
        
        # Get current price
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            if hist.empty:
                print(f"  [!] Could not get price for {symbol}")
                return None
            price = hist['Close'].iloc[-1]
        except Exception as e:
            print(f"  [!] Error getting price for {symbol}: {e}")
            return None
        
        # Calculate shares (allow fractional)
        shares = amount / price
        total = shares * price
        
        # Execute trade
        self.cash -= total
        
        # Update or create position
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if we already have this position
        cursor.execute('SELECT shares, entry_price FROM positions WHERE symbol = ?', (symbol,))
        existing = cursor.fetchone()
        
        if existing:
            # Average into existing position
            old_shares, old_price = existing
            new_shares = old_shares + shares
            new_avg_price = ((old_shares * old_price) + (shares * price)) / new_shares
            
            cursor.execute('''
                UPDATE positions SET shares = ?, entry_price = ?
                WHERE symbol = ?
            ''', (new_shares, new_avg_price, symbol))
        else:
            # New position
            cursor.execute('''
                INSERT INTO positions (symbol, shares, entry_price, entry_date, entry_reason)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, shares, price, datetime.now().strftime('%Y-%m-%d'), reason))
        
        # Record trade
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trade = Trade(
            id=trade_id,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action='BUY',
            shares=shares,
            price=price,
            total=total,
            reason=reason,
            signal_score=signal_score
        )
        
        cursor.execute('''
            INSERT INTO trades (id, timestamp, symbol, action, shares, price, total, reason, signal_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trade.id, trade.timestamp, trade.symbol, trade.action, 
              trade.shares, trade.price, trade.total, trade.reason, trade.signal_score))
        
        conn.commit()
        conn.close()
        
        self._save_state()
        
        print(f"  [BUY] {shares:.4f} shares of {symbol} @ ${price:.2f} = ${total:.2f}")
        return trade
    
    def sell(self, symbol: str, shares: float = None, reason: str = "Signal change") -> Optional[Trade]:
        """
        Sell a stock (partial or full)
        
        Args:
            symbol: Stock ticker
            shares: Number of shares to sell (None = sell all)
            reason: Why selling
        
        Returns:
            Trade record if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT shares, entry_price FROM positions WHERE symbol = ?', (symbol,))
        position = cursor.fetchone()
        
        if not position:
            print(f"  [!] No position in {symbol}")
            conn.close()
            return None
        
        current_shares, entry_price = position
        
        if shares is None:
            shares = current_shares
        
        if shares > current_shares:
            shares = current_shares
        
        # Get current price
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            price = hist['Close'].iloc[-1]
        except:
            price = entry_price
        
        total = shares * price
        self.cash += total
        
        # Update position
        remaining = current_shares - shares
        if remaining < 0.0001:
            cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
        else:
            cursor.execute('UPDATE positions SET shares = ? WHERE symbol = ?', (remaining, symbol))
        
        # Record trade
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_sell"
        trade = Trade(
            id=trade_id,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action='SELL',
            shares=shares,
            price=price,
            total=total,
            reason=reason,
            signal_score=0
        )
        
        cursor.execute('''
            INSERT INTO trades (id, timestamp, symbol, action, shares, price, total, reason, signal_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trade.id, trade.timestamp, trade.symbol, trade.action,
              trade.shares, trade.price, trade.total, trade.reason, trade.signal_score))
        
        conn.commit()
        conn.close()
        
        self._save_state()
        
        pnl = (price - entry_price) * shares
        pnl_pct = (price / entry_price - 1) * 100
        print(f"  [SELL] {shares:.4f} shares of {symbol} @ ${price:.2f} = ${total:.2f} (P&L: ${pnl:+.2f}, {pnl_pct:+.1f}%)")
        
        return trade
    
    def take_daily_snapshot(self) -> PortfolioSnapshot:
        """Record daily portfolio snapshot"""
        positions = self.get_positions()
        cash, positions_value, total_value = self.get_total_value()
        
        # Get yesterday's snapshot for daily change
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT total_value FROM snapshots 
            ORDER BY date DESC LIMIT 1
        ''')
        last = cursor.fetchone()
        
        if last:
            prev_value = last[0]
            daily_change = total_value - prev_value
            daily_change_pct = (daily_change / prev_value) * 100 if prev_value > 0 else 0
        else:
            daily_change = 0
            daily_change_pct = 0
        
        # Calculate total return
        total_return = total_value - self.starting_capital
        total_return_pct = (total_return / self.starting_capital) * 100
        
        # Get SPY return for comparison
        try:
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(start=self.start_date)
            if len(spy_hist) > 1:
                spy_return_pct = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100
            else:
                spy_return_pct = 0
        except:
            spy_return_pct = 0
        
        # Get today's trades
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT * FROM trades WHERE timestamp LIKE ?
        ''', (f"{today}%",))
        trades_today = cursor.fetchall()
        
        snapshot = PortfolioSnapshot(
            date=today,
            cash=cash,
            positions_value=positions_value,
            total_value=total_value,
            daily_change=daily_change,
            daily_change_pct=daily_change_pct,
            total_return=total_return,
            total_return_pct=total_return_pct,
            spy_return_pct=spy_return_pct,
            positions=[asdict(p) for p in positions],
            trades_today=[{
                'symbol': t[2], 'action': t[3], 'shares': t[4],
                'price': t[5], 'total': t[6], 'reason': t[7]
            } for t in trades_today]
        )
        
        # Save snapshot
        cursor.execute('''
            INSERT OR REPLACE INTO snapshots 
            (date, cash, positions_value, total_value, daily_change, daily_change_pct,
             total_return, total_return_pct, spy_return_pct, positions_json, trades_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.date, snapshot.cash, snapshot.positions_value, snapshot.total_value,
            snapshot.daily_change, snapshot.daily_change_pct, snapshot.total_return,
            snapshot.total_return_pct, snapshot.spy_return_pct,
            json.dumps(snapshot.positions), json.dumps(snapshot.trades_today)
        ))
        
        conn.commit()
        conn.close()
        
        return snapshot
    
    def get_performance_history(self, days: int = 30) -> List[Dict]:
        """Get portfolio performance history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, total_value, total_return_pct, spy_return_pct
            FROM snapshots
            ORDER BY date DESC
            LIMIT ?
        ''', (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {'date': r[0], 'value': r[1], 'return_pct': r[2], 'spy_pct': r[3]}
            for r in reversed(rows)
        ]
    
    def get_trade_history(self, days: int = 30) -> List[Trade]:
        """Get recent trade history"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trades WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', (cutoff,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Trade(id=r[0], timestamp=r[1], symbol=r[2], action=r[3],
                  shares=r[4], price=r[5], total=r[6], reason=r[7], signal_score=r[8])
            for r in rows
        ]


def get_portfolio_status() -> Dict:
    """Get current portfolio status for display"""
    portfolio = PaperTradingPortfolio()
    cash, positions_value, total_value = portfolio.get_total_value()
    positions = portfolio.get_positions()
    
    total_return = total_value - portfolio.starting_capital
    total_return_pct = (total_return / portfolio.starting_capital) * 100
    
    return {
        'starting_capital': portfolio.starting_capital,
        'cash': cash,
        'positions_value': positions_value,
        'total_value': total_value,
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'start_date': portfolio.start_date,
        'positions': [asdict(p) for p in positions],
        'num_positions': len(positions)
    }
