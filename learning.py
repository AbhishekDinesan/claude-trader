"""
Autonomous Learning System
Tracks predictions, evaluates outcomes, and learns from mistakes

Features:
- Logs all predictions with timestamps
- Periodically checks actual price movements vs predictions
- Calculates accuracy metrics by signal type, stock, and indicator
- Adjusts scoring weights based on what's actually working
- Runs autonomously on schedule
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

import yfinance as yf
from config import SCORING_CONFIG


# Database path for persistence
DB_PATH = os.path.join(os.path.dirname(__file__), 'learning_data.db')
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'learned_weights.json')


@dataclass
class PredictionRecord:
    """A single prediction to track"""
    id: str
    symbol: str
    timestamp: str
    
    # Prediction details
    signal: str  # STRONG BUY, BUY, HOLD, SELL, STRONG SELL
    overall_score: float
    technical_score: float
    momentum_score: float
    sentiment_score: float
    
    # Price at prediction
    entry_price: float
    target_upside: float
    stop_loss: float
    
    # Technical indicators at prediction time
    rsi: float
    macd_signal: str  # bullish, bearish, none
    trend: str  # bullish, bearish, neutral
    bb_signal: str  # oversold, overbought, neutral
    volume_ratio: float
    
    # Outcome (filled in later)
    outcome_checked: bool = False
    check_date: str = ""
    actual_return_5d: float = 0.0
    actual_return_10d: float = 0.0
    actual_return_20d: float = 0.0
    max_gain: float = 0.0
    max_drawdown: float = 0.0
    hit_target: bool = False
    hit_stop: bool = False
    outcome_label: str = ""  # WIN, LOSS, NEUTRAL


@dataclass 
class LearningMetrics:
    """Aggregated learning metrics"""
    total_predictions: int
    evaluated_predictions: int
    
    # Win rates by signal
    win_rate_strong_buy: float
    win_rate_buy: float
    win_rate_hold: float
    win_rate_sell: float
    
    # Average returns by signal
    avg_return_strong_buy: float
    avg_return_buy: float
    avg_return_hold: float
    avg_return_sell: float
    
    # Indicator effectiveness (correlation with positive outcomes)
    rsi_accuracy: float
    macd_accuracy: float
    trend_accuracy: float
    volume_accuracy: float
    sentiment_accuracy: float
    
    # Suggested weight adjustments
    suggested_weights: Dict[str, float]
    
    # Last updated
    last_updated: str


class PredictionTracker:
    """Tracks and stores predictions for learning"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                timestamp TEXT,
                signal TEXT,
                overall_score REAL,
                technical_score REAL,
                momentum_score REAL,
                sentiment_score REAL,
                entry_price REAL,
                target_upside REAL,
                stop_loss REAL,
                rsi REAL,
                macd_signal TEXT,
                trend TEXT,
                bb_signal TEXT,
                volume_ratio REAL,
                outcome_checked INTEGER DEFAULT 0,
                check_date TEXT,
                actual_return_5d REAL,
                actual_return_10d REAL,
                actual_return_20d REAL,
                max_gain REAL,
                max_drawdown REAL,
                hit_target INTEGER DEFAULT 0,
                hit_stop INTEGER DEFAULT 0,
                outcome_label TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metrics_json TEXT,
                weights_json TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, prediction: PredictionRecord) -> bool:
        """Log a new prediction to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO predictions 
                (id, symbol, timestamp, signal, overall_score, technical_score,
                 momentum_score, sentiment_score, entry_price, target_upside,
                 stop_loss, rsi, macd_signal, trend, bb_signal, volume_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.id, prediction.symbol, prediction.timestamp,
                prediction.signal, prediction.overall_score, prediction.technical_score,
                prediction.momentum_score, prediction.sentiment_score,
                prediction.entry_price, prediction.target_upside, prediction.stop_loss,
                prediction.rsi, prediction.macd_signal, prediction.trend,
                prediction.bb_signal, prediction.volume_ratio
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging prediction: {e}")
            return False
    
    def get_pending_evaluations(self, min_days: int = 5) -> List[PredictionRecord]:
        """Get predictions that haven't been evaluated yet and are old enough"""
        cutoff_date = (datetime.now() - timedelta(days=min_days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE outcome_checked = 0 AND timestamp < ?
        ''', (cutoff_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append(PredictionRecord(
                id=row[0], symbol=row[1], timestamp=row[2], signal=row[3],
                overall_score=row[4], technical_score=row[5], momentum_score=row[6],
                sentiment_score=row[7], entry_price=row[8], target_upside=row[9],
                stop_loss=row[10], rsi=row[11], macd_signal=row[12], trend=row[13],
                bb_signal=row[14], volume_ratio=row[15], outcome_checked=bool(row[16]),
                check_date=row[17] or "", actual_return_5d=row[18] or 0,
                actual_return_10d=row[19] or 0, actual_return_20d=row[20] or 0,
                max_gain=row[21] or 0, max_drawdown=row[22] or 0,
                hit_target=bool(row[23]), hit_stop=bool(row[24]),
                outcome_label=row[25] or ""
            ))
        
        return predictions
    
    def update_prediction_outcome(self, prediction: PredictionRecord):
        """Update a prediction with its outcome"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions SET
                outcome_checked = 1,
                check_date = ?,
                actual_return_5d = ?,
                actual_return_10d = ?,
                actual_return_20d = ?,
                max_gain = ?,
                max_drawdown = ?,
                hit_target = ?,
                hit_stop = ?,
                outcome_label = ?
            WHERE id = ?
        ''', (
            prediction.check_date, prediction.actual_return_5d,
            prediction.actual_return_10d, prediction.actual_return_20d,
            prediction.max_gain, prediction.max_drawdown,
            int(prediction.hit_target), int(prediction.hit_stop),
            prediction.outcome_label, prediction.id
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_evaluated(self) -> List[PredictionRecord]:
        """Get all predictions that have been evaluated"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM predictions WHERE outcome_checked = 1')
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append(PredictionRecord(
                id=row[0], symbol=row[1], timestamp=row[2], signal=row[3],
                overall_score=row[4], technical_score=row[5], momentum_score=row[6],
                sentiment_score=row[7], entry_price=row[8], target_upside=row[9],
                stop_loss=row[10], rsi=row[11], macd_signal=row[12], trend=row[13],
                bb_signal=row[14], volume_ratio=row[15], outcome_checked=True,
                check_date=row[17] or "", actual_return_5d=row[18] or 0,
                actual_return_10d=row[19] or 0, actual_return_20d=row[20] or 0,
                max_gain=row[21] or 0, max_drawdown=row[22] or 0,
                hit_target=bool(row[23]), hit_stop=bool(row[24]),
                outcome_label=row[25] or ""
            ))
        
        return predictions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quick statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE outcome_checked = 1')
        evaluated = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE outcome_label = "WIN"')
        wins = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_predictions': total,
            'evaluated': evaluated,
            'pending': total - evaluated,
            'wins': wins,
            'win_rate': wins / evaluated if evaluated > 0 else 0
        }


class OutcomeEvaluator:
    """Evaluates prediction outcomes by checking actual price movements"""
    
    def __init__(self, tracker: PredictionTracker):
        self.tracker = tracker
    
    def evaluate_prediction(self, prediction: PredictionRecord) -> PredictionRecord:
        """
        Evaluate a single prediction by checking what actually happened
        
        Looks at:
        - 5-day, 10-day, 20-day returns
        - Max gain and drawdown during period
        - Whether target or stop was hit
        """
        try:
            # Get price data from prediction date to now
            pred_date = datetime.fromisoformat(prediction.timestamp)
            
            ticker = yf.Ticker(prediction.symbol)
            hist = ticker.history(start=pred_date, end=datetime.now())
            
            if hist.empty or len(hist) < 5:
                return prediction
            
            entry_price = prediction.entry_price
            
            # Calculate returns at different horizons
            if len(hist) >= 5:
                prediction.actual_return_5d = (hist['Close'].iloc[4] / entry_price - 1) * 100
            if len(hist) >= 10:
                prediction.actual_return_10d = (hist['Close'].iloc[9] / entry_price - 1) * 100
            if len(hist) >= 20:
                prediction.actual_return_20d = (hist['Close'].iloc[19] / entry_price - 1) * 100
            else:
                prediction.actual_return_20d = (hist['Close'].iloc[-1] / entry_price - 1) * 100
            
            # Max gain and drawdown
            highs = hist['High'] / entry_price - 1
            lows = hist['Low'] / entry_price - 1
            
            prediction.max_gain = highs.max() * 100
            prediction.max_drawdown = lows.min() * 100
            
            # Check if target or stop was hit
            prediction.hit_target = prediction.max_gain >= prediction.target_upside
            prediction.hit_stop = prediction.max_drawdown <= -prediction.stop_loss
            
            # Determine outcome label
            # Use 10-day return as primary metric
            primary_return = prediction.actual_return_10d if len(hist) >= 10 else prediction.actual_return_5d
            
            if prediction.signal in ['STRONG BUY', 'BUY']:
                # For buy signals, we want positive returns
                if primary_return > 3:
                    prediction.outcome_label = 'WIN'
                elif primary_return < -3:
                    prediction.outcome_label = 'LOSS'
                else:
                    prediction.outcome_label = 'NEUTRAL'
            elif prediction.signal in ['SELL', 'STRONG SELL']:
                # For sell signals, we want negative returns (avoiding loss)
                if primary_return < -3:
                    prediction.outcome_label = 'WIN'
                elif primary_return > 3:
                    prediction.outcome_label = 'LOSS'
                else:
                    prediction.outcome_label = 'NEUTRAL'
            else:
                # HOLD signals
                if abs(primary_return) < 5:
                    prediction.outcome_label = 'WIN'
                else:
                    prediction.outcome_label = 'NEUTRAL'
            
            prediction.outcome_checked = True
            prediction.check_date = datetime.now().isoformat()
            
        except Exception as e:
            print(f"Error evaluating {prediction.symbol}: {e}")
        
        return prediction
    
    def evaluate_all_pending(self, min_days: int = 5) -> int:
        """Evaluate all pending predictions"""
        pending = self.tracker.get_pending_evaluations(min_days)
        
        evaluated_count = 0
        for prediction in pending:
            print(f"  Evaluating {prediction.symbol} from {prediction.timestamp[:10]}...")
            updated = self.evaluate_prediction(prediction)
            
            if updated.outcome_checked:
                self.tracker.update_prediction_outcome(updated)
                evaluated_count += 1
                print(f"    -> {updated.outcome_label}: {updated.actual_return_10d:+.1f}% (10d)")
        
        return evaluated_count


class AdaptiveWeightLearner:
    """Learns optimal weights from prediction outcomes"""
    
    def __init__(self, tracker: PredictionTracker):
        self.tracker = tracker
        self.weights_path = WEIGHTS_PATH
    
    def analyze_performance(self) -> LearningMetrics:
        """Analyze all evaluated predictions to learn what works"""
        evaluated = self.tracker.get_all_evaluated()
        
        if len(evaluated) < 10:
            print("Not enough data to learn from (need at least 10 evaluated predictions)")
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(p) for p in evaluated])
        
        # Win rates by signal
        signal_groups = df.groupby('signal')
        
        def calc_win_rate(group):
            return (group['outcome_label'] == 'WIN').sum() / len(group) if len(group) > 0 else 0
        
        win_rates = signal_groups.apply(calc_win_rate)
        avg_returns = signal_groups['actual_return_10d'].mean()
        
        # Indicator effectiveness analysis
        # Check if technical indicators correlate with positive outcomes
        
        # RSI accuracy (oversold -> positive returns for buys)
        rsi_accuracy = self._calc_indicator_accuracy(df, 'rsi', 'actual_return_10d')
        
        # MACD accuracy
        macd_accuracy = self._calc_macd_accuracy(df)
        
        # Trend accuracy
        trend_accuracy = self._calc_trend_accuracy(df)
        
        # Volume accuracy
        volume_accuracy = self._calc_volume_accuracy(df)
        
        # Sentiment accuracy (based on sentiment_score)
        sentiment_accuracy = self._calc_sentiment_accuracy(df)
        
        # Calculate suggested weights based on accuracy
        suggested_weights = self._calculate_optimal_weights(
            rsi_accuracy, macd_accuracy, trend_accuracy, 
            volume_accuracy, sentiment_accuracy
        )
        
        metrics = LearningMetrics(
            total_predictions=len(df) + len(self.tracker.get_pending_evaluations(0)),
            evaluated_predictions=len(df),
            win_rate_strong_buy=win_rates.get('STRONG BUY', 0),
            win_rate_buy=win_rates.get('BUY', 0),
            win_rate_hold=win_rates.get('HOLD', 0),
            win_rate_sell=win_rates.get('SELL', 0) if 'SELL' in win_rates else 0,
            avg_return_strong_buy=avg_returns.get('STRONG BUY', 0),
            avg_return_buy=avg_returns.get('BUY', 0),
            avg_return_hold=avg_returns.get('HOLD', 0),
            avg_return_sell=avg_returns.get('SELL', 0) if 'SELL' in avg_returns else 0,
            rsi_accuracy=rsi_accuracy,
            macd_accuracy=macd_accuracy,
            trend_accuracy=trend_accuracy,
            volume_accuracy=volume_accuracy,
            sentiment_accuracy=sentiment_accuracy,
            suggested_weights=suggested_weights,
            last_updated=datetime.now().isoformat()
        )
        
        return metrics
    
    def _calc_indicator_accuracy(self, df: pd.DataFrame, indicator: str, return_col: str) -> float:
        """Calculate correlation between indicator and returns"""
        try:
            valid = df[[indicator, return_col]].dropna()
            if len(valid) < 5:
                return 0.5
            
            correlation = valid[indicator].corr(valid[return_col])
            # Convert correlation (-1 to 1) to accuracy (0 to 1)
            return (correlation + 1) / 2
        except:
            return 0.5
    
    def _calc_macd_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate MACD signal accuracy"""
        try:
            # Bullish MACD should lead to positive returns on buys
            bullish = df[df['macd_signal'] == 'bullish']
            bearish = df[df['macd_signal'] == 'bearish']
            
            bullish_correct = (bullish['actual_return_10d'] > 0).sum() / len(bullish) if len(bullish) > 0 else 0.5
            bearish_correct = (bearish['actual_return_10d'] < 0).sum() / len(bearish) if len(bearish) > 0 else 0.5
            
            return (bullish_correct + bearish_correct) / 2
        except:
            return 0.5
    
    def _calc_trend_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate trend indicator accuracy"""
        try:
            bullish = df[df['trend'] == 'bullish']
            bearish = df[df['trend'] == 'bearish']
            
            bullish_correct = (bullish['actual_return_10d'] > 0).sum() / len(bullish) if len(bullish) > 0 else 0.5
            bearish_correct = (bearish['actual_return_10d'] < 0).sum() / len(bearish) if len(bearish) > 0 else 0.5
            
            return (bullish_correct + bearish_correct) / 2
        except:
            return 0.5
    
    def _calc_volume_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate volume indicator accuracy"""
        try:
            # High volume should confirm moves
            high_vol = df[df['volume_ratio'] > 1.5]
            if len(high_vol) < 3:
                return 0.5
            
            # Check if high volume predictions had stronger moves
            high_vol_abs_return = high_vol['actual_return_10d'].abs().mean()
            all_abs_return = df['actual_return_10d'].abs().mean()
            
            return min(1.0, high_vol_abs_return / all_abs_return) if all_abs_return > 0 else 0.5
        except:
            return 0.5
    
    def _calc_sentiment_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate sentiment score accuracy"""
        try:
            # Higher sentiment scores should correlate with positive returns
            return self._calc_indicator_accuracy(df, 'sentiment_score', 'actual_return_10d')
        except:
            return 0.5
    
    def _calculate_optimal_weights(self, rsi_acc: float, macd_acc: float, 
                                    trend_acc: float, vol_acc: float, 
                                    sent_acc: float) -> Dict[str, float]:
        """Calculate optimal weights based on indicator accuracy"""
        # Base weights
        base_weights = {
            'technical': 0.45,
            'momentum': 0.35,
            'sentiment': 0.20
        }
        
        # Adjust based on accuracy
        # Technical weight is influenced by RSI, MACD, trend, volume
        tech_accuracy = (rsi_acc + macd_acc + trend_acc + vol_acc) / 4
        
        # Scale weights based on accuracy (0.5 = no change)
        tech_multiplier = 0.5 + tech_accuracy
        sent_multiplier = 0.5 + sent_acc
        
        # Calculate new weights
        raw_tech = base_weights['technical'] * tech_multiplier
        raw_sent = base_weights['sentiment'] * sent_multiplier
        raw_mom = base_weights['momentum']  # Momentum stays similar
        
        # Normalize to sum to 1
        total = raw_tech + raw_mom + raw_sent
        
        return {
            'technical': round(raw_tech / total, 3),
            'momentum': round(raw_mom / total, 3),
            'sentiment': round(raw_sent / total, 3),
            'tech_accuracy': round(tech_accuracy, 3),
            'sentiment_accuracy': round(sent_acc, 3)
        }
    
    def save_learned_weights(self, metrics: LearningMetrics):
        """Save learned weights to file"""
        data = {
            'weights': metrics.suggested_weights,
            'metrics': {
                'win_rate_buy': metrics.win_rate_buy,
                'win_rate_strong_buy': metrics.win_rate_strong_buy,
                'avg_return_buy': metrics.avg_return_buy,
                'evaluated_predictions': metrics.evaluated_predictions
            },
            'last_updated': metrics.last_updated
        }
        
        with open(self.weights_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved learned weights to {self.weights_path}")
    
    def load_learned_weights(self) -> Optional[Dict[str, float]]:
        """Load previously learned weights"""
        if os.path.exists(self.weights_path):
            with open(self.weights_path, 'r') as f:
                data = json.load(f)
            return data.get('weights')
        return None


class AutonomousLearner:
    """Main class for autonomous learning - ties everything together"""
    
    def __init__(self):
        self.tracker = PredictionTracker()
        self.evaluator = OutcomeEvaluator(self.tracker)
        self.weight_learner = AdaptiveWeightLearner(self.tracker)
    
    def log_prediction_from_opportunity(self, opp) -> bool:
        """
        Log a prediction from a StockOpportunity object
        Call this after every scan to track predictions
        """
        pred_id = f"{opp.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract technical indicators
        rsi = opp.technical.rsi if opp.technical else 50
        macd_signal = opp.technical.macd_crossover if opp.technical else 'none'
        trend = opp.technical.trend_direction if opp.technical else 'neutral'
        bb_signal = opp.technical.bb_signal if opp.technical else 'neutral'
        volume_ratio = opp.technical.volume_ratio if opp.technical else 1.0
        
        prediction = PredictionRecord(
            id=pred_id,
            symbol=opp.symbol,
            timestamp=datetime.now().isoformat(),
            signal=opp.signal,
            overall_score=opp.overall_score,
            technical_score=opp.technical_score,
            momentum_score=opp.momentum_score,
            sentiment_score=opp.sentiment_score,
            entry_price=opp.current_price,
            target_upside=opp.target_upside,
            stop_loss=opp.stop_loss,
            rsi=rsi,
            macd_signal=macd_signal,
            trend=trend,
            bb_signal=bb_signal,
            volume_ratio=volume_ratio
        )
        
        return self.tracker.log_prediction(prediction)
    
    def run_learning_cycle(self, min_eval_days: int = 5) -> Dict[str, Any]:
        """
        Run a complete learning cycle:
        1. Evaluate pending predictions
        2. Analyze performance
        3. Calculate new weights
        4. Save results
        """
        print("\n" + "="*60)
        print("[AUTONOMOUS LEARNING CYCLE]")
        print("="*60)
        
        # Step 1: Evaluate pending predictions
        print("\n[Step 1] Evaluating pending predictions...")
        evaluated = self.evaluator.evaluate_all_pending(min_eval_days)
        print(f"  Evaluated {evaluated} predictions")
        
        # Step 2: Get current stats
        stats = self.tracker.get_stats()
        print(f"\n[Step 2] Current Statistics:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Evaluated: {stats['evaluated']}")
        print(f"  Pending: {stats['pending']}")
        print(f"  Overall Win Rate: {stats['win_rate']*100:.1f}%")
        
        # Step 3: Analyze and learn
        print("\n[Step 3] Analyzing performance and learning...")
        metrics = self.weight_learner.analyze_performance()
        
        result = {
            'evaluated_this_cycle': evaluated,
            'stats': stats
        }
        
        if metrics:
            print(f"\n  Performance by Signal Type:")
            print(f"    STRONG BUY: {metrics.win_rate_strong_buy*100:.1f}% win rate, {metrics.avg_return_strong_buy:+.1f}% avg return")
            print(f"    BUY: {metrics.win_rate_buy*100:.1f}% win rate, {metrics.avg_return_buy:+.1f}% avg return")
            print(f"    HOLD: {metrics.win_rate_hold*100:.1f}% win rate, {metrics.avg_return_hold:+.1f}% avg return")
            
            print(f"\n  Indicator Accuracy:")
            print(f"    RSI: {metrics.rsi_accuracy*100:.1f}%")
            print(f"    MACD: {metrics.macd_accuracy*100:.1f}%")
            print(f"    Trend: {metrics.trend_accuracy*100:.1f}%")
            print(f"    Volume: {metrics.volume_accuracy*100:.1f}%")
            print(f"    Sentiment: {metrics.sentiment_accuracy*100:.1f}%")
            
            print(f"\n  Suggested Weight Adjustments:")
            print(f"    Technical: {metrics.suggested_weights['technical']*100:.1f}%")
            print(f"    Momentum: {metrics.suggested_weights['momentum']*100:.1f}%")
            print(f"    Sentiment: {metrics.suggested_weights['sentiment']*100:.1f}%")
            
            # Step 4: Save learned weights
            self.weight_learner.save_learned_weights(metrics)
            
            result['metrics'] = asdict(metrics)
        
        print("\n" + "="*60)
        print("[LEARNING CYCLE COMPLETE]")
        print("="*60)
        
        return result
    
    def get_learned_weights(self) -> Optional[Dict[str, float]]:
        """Get the currently learned weights"""
        return self.weight_learner.load_learned_weights()


def create_prediction_from_scan(opportunity) -> Optional[PredictionRecord]:
    """Helper to create a prediction record from a scan result"""
    learner = AutonomousLearner()
    if learner.log_prediction_from_opportunity(opportunity):
        return True
    return False


# CLI functions for manual use
def run_learning_cycle():
    """Run learning cycle from command line"""
    learner = AutonomousLearner()
    return learner.run_learning_cycle()


def show_learning_stats():
    """Show current learning statistics"""
    tracker = PredictionTracker()
    stats = tracker.get_stats()
    
    print("\n[LEARNING STATISTICS]")
    print(f"  Total predictions logged: {stats['total_predictions']}")
    print(f"  Evaluated: {stats['evaluated']}")
    print(f"  Pending evaluation: {stats['pending']}")
    print(f"  Overall Win Rate: {stats['win_rate']*100:.1f}%")
    
    # Load and show current weights
    learner = AdaptiveWeightLearner(tracker)
    weights = learner.load_learned_weights()
    
    if weights:
        print(f"\n[CURRENT LEARNED WEIGHTS]")
        print(f"  Technical: {weights.get('technical', 0.45)*100:.1f}%")
        print(f"  Momentum: {weights.get('momentum', 0.35)*100:.1f}%")
        print(f"  Sentiment: {weights.get('sentiment', 0.20)*100:.1f}%")
    else:
        print("\n  No learned weights yet (using defaults)")
    
    return stats
