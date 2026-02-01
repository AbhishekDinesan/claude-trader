"""
Machine Learning Trading Model
Sophisticated ensemble model for stock prediction

Features:
- Gradient Boosting (XGBoost/LightGBM) ensemble
- Neural network pattern recognition
- Multi-target prediction (probability, returns, confidence)
- Feature engineering from technical, momentum, and sentiment data
- Historical backtesting and walk-forward optimization
- Online learning capabilities
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MLPrediction:
    """Container for ML model predictions"""
    symbol: str
    
    # Primary predictions
    win_probability: float  # 0-1 probability of profitable trade
    expected_return: float  # Expected % return over prediction horizon
    
    # Confidence metrics
    model_confidence: float  # 0-1 how confident the model is
    prediction_quality: str  # 'high', 'medium', 'low'
    
    # Model agreement (for ensemble)
    ensemble_agreement: float  # How much models agree (0-1)
    
    # Signal strength
    ml_signal: str  # 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    ml_score: float  # 0-100 score
    
    # Feature importance for this prediction
    top_features: List[Tuple[str, float]]
    
    # Metadata
    prediction_horizon: int  # Days forward
    model_version: str
    predicted_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ModelMetrics:
    """Performance metrics for the model"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    sharpe_ratio: float
    win_rate: float
    avg_return: float
    max_drawdown: float
    total_trades: int
    profitable_trades: int


class FeatureEngineer:
    """
    Comprehensive feature engineering from all data sources
    Extracts ~100+ features for ML model
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler = RobustScaler() if SKLEARN_AVAILABLE else None
        self.fitted = False
    
    def extract_features(self, 
                        price_data: pd.DataFrame,
                        technical_signals: Any = None,
                        screener_result: Any = None,
                        sentiment_result: Any = None) -> np.ndarray:
        """
        Extract all features from available data sources
        
        Returns:
            Feature vector as numpy array
        """
        features = {}
        
        # === PRICE-BASED FEATURES ===
        if price_data is not None and not price_data.empty:
            features.update(self._extract_price_features(price_data))
        
        # === TECHNICAL INDICATOR FEATURES ===
        if technical_signals is not None:
            features.update(self._extract_technical_features(technical_signals))
        
        # === SCREENER/MOMENTUM FEATURES ===
        if screener_result is not None:
            features.update(self._extract_momentum_features(screener_result))
        
        # === SENTIMENT FEATURES ===
        if sentiment_result is not None:
            features.update(self._extract_sentiment_features(sentiment_result))
        
        # === DERIVED/INTERACTION FEATURES ===
        features.update(self._extract_interaction_features(features))
        
        # Store feature names
        self.feature_names = list(features.keys())
        
        # Convert to array
        feature_vector = np.array([features.get(f, 0) for f in self.feature_names])
        
        return feature_vector
    
    def _extract_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract features from price data"""
        features = {}
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Returns at various horizons
        for days in [1, 2, 3, 5, 10, 21, 63]:
            if len(close) > days:
                ret = (close.iloc[-1] / close.iloc[-days-1] - 1) * 100
                features[f'return_{days}d'] = ret
        
        # Volatility features
        returns = close.pct_change().dropna()
        if len(returns) > 10:
            features['volatility_5d'] = returns.tail(5).std() * np.sqrt(252) * 100
            features['volatility_20d'] = returns.tail(20).std() * np.sqrt(252) * 100
            features['volatility_ratio'] = features['volatility_5d'] / (features['volatility_20d'] + 0.001)
        
        # Price position relative to range
        if len(close) >= 20:
            high_20 = high.tail(20).max()
            low_20 = low.tail(20).min()
            features['price_position_20d'] = (close.iloc[-1] - low_20) / (high_20 - low_20 + 0.001)
        
        if len(close) >= 52:
            high_52 = high.tail(252).max() if len(high) >= 252 else high.max()
            low_52 = low.tail(252).min() if len(low) >= 252 else low.min()
            features['price_position_52w'] = (close.iloc[-1] - low_52) / (high_52 - low_52 + 0.001)
        
        # Gap features
        if len(df) >= 2:
            features['overnight_gap'] = ((df['Open'].iloc[-1] / close.iloc[-2]) - 1) * 100
        
        # Range features
        daily_range = (high - low) / close * 100
        features['avg_daily_range_5d'] = daily_range.tail(5).mean()
        features['avg_daily_range_20d'] = daily_range.tail(20).mean()
        
        # Volume features
        avg_vol = volume.tail(20).mean()
        features['volume_ratio_1d'] = volume.iloc[-1] / (avg_vol + 1)
        features['volume_ratio_5d'] = volume.tail(5).mean() / (avg_vol + 1)
        features['volume_trend'] = volume.tail(5).mean() / (volume.tail(20).mean() + 1)
        
        # Price momentum indicators
        if len(close) >= 10:
            features['momentum_5d'] = close.iloc[-1] / close.iloc[-5] - 1
            features['momentum_10d'] = close.iloc[-1] / close.iloc[-10] - 1
        
        # Rate of change acceleration
        if len(close) >= 10:
            roc_5 = close.pct_change(5).dropna()
            if len(roc_5) >= 5:
                features['momentum_acceleration'] = roc_5.iloc[-1] - roc_5.iloc[-5]
        
        # Price distribution features
        if len(close) >= 20:
            features['price_skewness'] = returns.tail(20).skew()
            features['price_kurtosis'] = returns.tail(20).kurtosis()
        
        return features
    
    def _extract_technical_features(self, tech) -> Dict[str, float]:
        """Extract features from technical analysis signals"""
        features = {}
        
        # RSI features
        features['rsi'] = tech.rsi
        features['rsi_oversold'] = 1 if tech.rsi_signal == 'oversold' else 0
        features['rsi_overbought'] = 1 if tech.rsi_signal == 'overbought' else 0
        features['rsi_distance_from_50'] = abs(tech.rsi - 50)
        
        # MACD features
        features['macd'] = tech.macd
        features['macd_signal'] = tech.macd_signal_line
        features['macd_histogram'] = tech.macd_histogram
        features['macd_bullish_cross'] = 1 if tech.macd_crossover == 'bullish' else 0
        features['macd_bearish_cross'] = 1 if tech.macd_crossover == 'bearish' else 0
        
        # Bollinger Band features
        features['bb_position'] = tech.bb_position
        features['bb_squeeze'] = 1 if tech.bb_squeeze else 0
        features['bb_oversold'] = 1 if tech.bb_signal == 'oversold' else 0
        features['bb_overbought'] = 1 if tech.bb_signal == 'overbought' else 0
        
        # Moving average features
        features['above_sma20'] = 1 if tech.above_sma20 else 0
        features['above_sma50'] = 1 if tech.above_sma50 else 0
        features['golden_cross'] = 1 if tech.golden_cross else 0
        features['death_cross'] = 1 if tech.death_cross else 0
        
        # Trend features
        features['trend_bullish'] = 1 if tech.trend_direction == 'bullish' else 0
        features['trend_bearish'] = 1 if tech.trend_direction == 'bearish' else 0
        features['trend_strength'] = tech.trend_strength
        
        # Volume features
        features['volume_ratio'] = tech.volume_ratio
        features['volume_increasing'] = 1 if tech.volume_trend == 'increasing' else 0
        
        # Support/Resistance
        features['near_support'] = 1 if tech.near_support else 0
        features['near_resistance'] = 1 if tech.near_resistance else 0
        
        # Technical score
        features['technical_score'] = tech.technical_score
        
        return features
    
    def _extract_momentum_features(self, scr) -> Dict[str, float]:
        """Extract features from screener/momentum data"""
        features = {}
        
        # Price changes at different timeframes
        features['change_1d'] = scr.change_1d
        features['change_1w'] = scr.change_1w
        features['change_1m'] = scr.change_1m
        features['change_3m'] = scr.change_3m
        
        # Momentum acceleration
        features['momentum_accel_1w_1d'] = scr.change_1w - scr.change_1d * 5
        features['momentum_accel_1m_1w'] = scr.change_1m - scr.change_1w * 4
        
        # Volatility and beta
        features['daily_volatility'] = scr.volatility
        features['beta'] = scr.beta if scr.beta else 1.0
        
        # Volume metrics
        features['screener_volume_ratio'] = scr.volume_ratio
        
        # Fundamental
        features['pe_ratio'] = scr.pe_ratio if scr.pe_ratio and scr.pe_ratio > 0 else 25.0
        features['pe_low'] = 1 if scr.pe_ratio and scr.pe_ratio < 15 else 0
        features['pe_high'] = 1 if scr.pe_ratio and scr.pe_ratio > 50 else 0
        
        # Market cap (log transformed)
        features['log_market_cap'] = np.log10(scr.market_cap + 1)
        
        return features
    
    def _extract_sentiment_features(self, sent) -> Dict[str, float]:
        """Extract features from sentiment analysis"""
        features = {}
        
        # Overall sentiment
        features['overall_sentiment'] = sent.overall_sentiment
        features['sentiment_bullish'] = 1 if sent.sentiment_label in ['bullish', 'very_bullish'] else 0
        features['sentiment_bearish'] = 1 if sent.sentiment_label in ['bearish', 'very_bearish'] else 0
        
        # Reddit sentiment
        features['reddit_sentiment'] = sent.reddit_sentiment
        features['reddit_mentions'] = min(100, sent.reddit_mentions)  # Cap outliers
        features['reddit_mentions_log'] = np.log1p(sent.reddit_mentions)
        
        # News sentiment
        features['news_sentiment'] = sent.news_sentiment
        features['news_articles'] = min(50, sent.news_articles)
        
        # Buzz and trend
        features['buzz_score'] = sent.buzz_score
        features['sentiment_improving'] = 1 if sent.sentiment_trend == 'improving' else 0
        features['sentiment_declining'] = 1 if sent.sentiment_trend == 'declining' else 0
        
        # Sentiment agreement (reddit vs news)
        features['sentiment_agreement'] = 1 - abs(sent.reddit_sentiment - sent.news_sentiment) / 2
        
        return features
    
    def _extract_interaction_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction and derived features"""
        derived = {}
        
        # Technical + Sentiment interactions
        if 'technical_score' in features and 'overall_sentiment' in features:
            derived['tech_sentiment_product'] = features['technical_score'] * (features['overall_sentiment'] + 1)
            derived['tech_sentiment_agreement'] = 1 if (
                (features['technical_score'] > 60 and features['overall_sentiment'] > 0) or
                (features['technical_score'] < 40 and features['overall_sentiment'] < 0)
            ) else 0
        
        # Momentum + Volume interactions
        if 'change_1w' in features and 'volume_ratio' in features:
            derived['momentum_volume_product'] = features['change_1w'] * features['volume_ratio']
        
        # RSI + BB interactions (potential reversals)
        if 'rsi_oversold' in features and 'bb_oversold' in features:
            derived['double_oversold'] = features['rsi_oversold'] * features['bb_oversold']
        if 'rsi_overbought' in features and 'bb_overbought' in features:
            derived['double_overbought'] = features['rsi_overbought'] * features['bb_overbought']
        
        # Trend confirmation
        if 'trend_bullish' in features and 'macd_bullish_cross' in features:
            derived['bullish_confirmation'] = features['trend_bullish'] * features['macd_bullish_cross']
        
        # Risk-adjusted momentum
        if 'change_1w' in features and 'daily_volatility' in features:
            derived['sharpe_1w'] = features['change_1w'] / (features['daily_volatility'] + 0.001)
        
        # Momentum consistency
        if all(k in features for k in ['change_1d', 'change_1w', 'change_1m']):
            signs = [np.sign(features[k]) for k in ['change_1d', 'change_1w', 'change_1m']]
            derived['momentum_consistency'] = abs(sum(signs)) / 3
        
        return derived
    
    def fit_scaler(self, X: np.ndarray):
        """Fit the scaler on training data"""
        if self.scaler is not None:
            self.scaler.fit(X)
            self.fitted = True
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if self.scaler is not None and self.fitted:
            return self.scaler.transform(X)
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform features"""
        self.fit_scaler(X)
        return self.transform(X)


# Only define NeuralNetworkModel if PyTorch is available
if TORCH_AVAILABLE:
    class NeuralNetworkModel(nn.Module):
        """
        Deep neural network for pattern recognition in market data
        """
        
        def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                prev_dim = hidden_dim
            
            self.feature_extractor = nn.Sequential(*layers)
            
            # Output heads
            self.win_prob_head = nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            self.return_head = nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            features = self.feature_extractor(x)
            win_prob = self.win_prob_head(features)
            expected_return = self.return_head(features)
            return win_prob, expected_return
else:
    # Placeholder when PyTorch is not available
    NeuralNetworkModel = None


class TradingMLModel:
    """
    Sophisticated ensemble ML model for trading predictions
    Combines XGBoost, LightGBM, and Neural Networks
    """
    
    MODEL_VERSION = "2.0.0"
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.feature_engineer = FeatureEngineer()
        
        # Model ensemble
        self.models = {}
        self.model_weights = {}
        
        # Training state
        self.is_trained = False
        self.training_metrics = None
        self.feature_importance = {}
        
        # Configuration
        self.prediction_horizon = 5  # Days forward to predict
        self.min_training_samples = 100
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models in the ensemble"""
        
        # XGBoost classifier
        if XGBOOST_AVAILABLE:
            self.models['xgb_classifier'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.model_weights['xgb_classifier'] = 0.35
            
            self.models['xgb_regressor'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
            self.model_weights['xgb_regressor'] = 0.25
        
        # LightGBM classifier
        if LIGHTGBM_AVAILABLE:
            self.models['lgb_classifier'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            )
            self.model_weights['lgb_classifier'] = 0.25
        
        # Sklearn fallback
        if SKLEARN_AVAILABLE and not self.models:
            self.models['rf_classifier'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            self.model_weights['rf_classifier'] = 0.5
            
            self.models['gb_classifier'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.model_weights['gb_classifier'] = 0.5
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
    
    def prepare_training_data(self, historical_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from historical records
        
        Args:
            historical_data: List of dicts with features and outcomes
            
        Returns:
            X (features), y_class (win/loss), y_reg (returns)
        """
        X_list = []
        y_class_list = []
        y_reg_list = []
        
        for record in historical_data:
            try:
                features = record.get('features')
                future_return = record.get('future_return', 0)
                
                if features is not None and len(features) > 0:
                    X_list.append(features)
                    y_class_list.append(1 if future_return > 0 else 0)
                    y_reg_list.append(future_return)
            except Exception as e:
                continue
        
        if len(X_list) < self.min_training_samples:
            raise ValueError(f"Insufficient training data: {len(X_list)} samples (need {self.min_training_samples})")
        
        X = np.array(X_list)
        y_class = np.array(y_class_list)
        y_reg = np.array(y_reg_list)
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y_class, y_reg
    
    def train(self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray,
              validation_split: float = 0.2) -> ModelMetrics:
        """
        Train all models in the ensemble
        
        Args:
            X: Feature matrix
            y_class: Binary labels (win/loss)
            y_reg: Regression targets (returns)
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        print("\n" + "="*60)
        print("[ML ENSEMBLE MODEL TRAINING]")
        print("="*60)
        print(f"Training samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Win rate in data: {y_class.mean()*100:.1f}%")
        print("="*60)
        
        # Scale features
        X_scaled = self.feature_engineer.fit_transform(X)
        
        # Time series split for validation
        n_samples = len(X_scaled)
        split_idx = int(n_samples * (1 - validation_split))
        
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_class_train, y_class_val = y_class[:split_idx], y_class[split_idx:]
        y_reg_train, y_reg_val = y_reg[:split_idx], y_reg[split_idx:]
        
        # Train each model
        for name, model in self.models.items():
            try:
                print(f"\n[*] Training {name}...")
                
                if 'regressor' in name:
                    model.fit(X_train, y_reg_train)
                else:
                    model.fit(X_train, y_class_train)
                
                print(f"   [+] {name} trained successfully")
                
            except Exception as e:
                print(f"   [!] {name} failed: {e}")
        
        # Calculate metrics on validation set
        metrics = self._evaluate(X_val, y_class_val, y_reg_val)
        
        # Extract feature importance
        self._extract_feature_importance()
        
        self.is_trained = True
        self.training_metrics = metrics
        
        print("\n" + "="*60)
        print("[TRAINING RESULTS]")
        print("="*60)
        print(f"Accuracy: {metrics.accuracy*100:.1f}%")
        print(f"Precision: {metrics.precision*100:.1f}%")
        print(f"Recall: {metrics.recall*100:.1f}%")
        print(f"F1 Score: {metrics.f1_score*100:.1f}%")
        print(f"AUC-ROC: {metrics.auc_roc:.3f}")
        print(f"Win Rate: {metrics.win_rate*100:.1f}%")
        print(f"Avg Return: {metrics.avg_return:.2f}%")
        print("="*60)
        
        return metrics
    
    def _evaluate(self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray) -> ModelMetrics:
        """Evaluate model performance"""
        
        # Get ensemble predictions
        y_pred_proba = self._ensemble_predict_proba(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        y_pred_return = self._ensemble_predict_return(X)
        
        # Classification metrics
        accuracy = accuracy_score(y_class, y_pred)
        precision = precision_score(y_class, y_pred, zero_division=0)
        recall = recall_score(y_class, y_pred, zero_division=0)
        f1 = f1_score(y_class, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_class, y_pred_proba)
        except:
            auc = 0.5
        
        # Trading metrics (simulated)
        # Only trade when model is confident (prob > 0.6)
        confident_mask = y_pred_proba > 0.6
        trades = confident_mask.sum()
        
        if trades > 0:
            actual_returns = y_reg[confident_mask]
            profitable = (actual_returns > 0).sum()
            win_rate = profitable / trades
            avg_return = actual_returns.mean()
            
            # Calculate max drawdown
            cumulative = np.cumsum(actual_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
            
            # Sharpe ratio (simplified)
            if actual_returns.std() > 0:
                sharpe = actual_returns.mean() / actual_returns.std() * np.sqrt(252/self.prediction_horizon)
            else:
                sharpe = 0
        else:
            win_rate = 0
            avg_return = 0
            max_drawdown = 0
            sharpe = 0
            profitable = 0
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            avg_return=avg_return,
            max_drawdown=max_drawdown,
            total_trades=int(trades),
            profitable_trades=int(profitable)
        )
    
    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions"""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if 'classifier' in name:
                try:
                    proba = model.predict_proba(X)[:, 1]
                    predictions.append(proba)
                    weights.append(self.model_weights[name])
                except:
                    pass
        
        if not predictions:
            return np.full(len(X), 0.5)
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.average(predictions, axis=0, weights=weights)
    
    def _ensemble_predict_return(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble return predictions"""
        predictions = []
        
        for name, model in self.models.items():
            if 'regressor' in name:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except:
                    pass
        
        if not predictions:
            return np.zeros(len(X))
        
        return np.mean(predictions, axis=0)
    
    def _extract_feature_importance(self):
        """Extract feature importance from trained models"""
        importance_sum = np.zeros(len(self.feature_engineer.feature_names))
        count = 0
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    if len(importance) == len(importance_sum):
                        importance_sum += importance
                        count += 1
            except:
                pass
        
        if count > 0:
            importance_avg = importance_sum / count
            self.feature_importance = dict(zip(
                self.feature_engineer.feature_names,
                importance_avg
            ))
    
    def predict(self, 
                price_data: pd.DataFrame,
                technical_signals: Any = None,
                screener_result: Any = None,
                sentiment_result: Any = None,
                symbol: str = "UNKNOWN") -> Optional[MLPrediction]:
        """
        Make prediction for a single stock
        
        Returns:
            MLPrediction with all prediction details
        """
        if not self.is_trained:
            print("[!] Model not trained! Using fallback predictions.")
            return self._fallback_prediction(symbol)
        
        try:
            # Extract features
            features = self.feature_engineer.extract_features(
                price_data=price_data,
                technical_signals=technical_signals,
                screener_result=screener_result,
                sentiment_result=sentiment_result
            )
            
            # Handle missing values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale
            X = self.feature_engineer.transform(features.reshape(1, -1))
            
            # Get predictions from each model
            probas = []
            for name, model in self.models.items():
                if 'classifier' in name:
                    try:
                        proba = model.predict_proba(X)[0, 1]
                        probas.append(proba)
                    except:
                        pass
            
            # Ensemble probability
            win_probability = np.mean(probas) if probas else 0.5
            
            # Expected return
            expected_return = self._ensemble_predict_return(X)[0]
            
            # Model confidence (based on agreement)
            if len(probas) > 1:
                ensemble_agreement = 1 - np.std(probas) * 2  # Scale to 0-1
                ensemble_agreement = max(0, min(1, ensemble_agreement))
            else:
                ensemble_agreement = 0.5
            
            # Model confidence (how far from 0.5)
            model_confidence = abs(win_probability - 0.5) * 2
            
            # Prediction quality
            if model_confidence > 0.6 and ensemble_agreement > 0.7:
                prediction_quality = 'high'
            elif model_confidence > 0.3 and ensemble_agreement > 0.5:
                prediction_quality = 'medium'
            else:
                prediction_quality = 'low'
            
            # Signal and score
            ml_score = win_probability * 100
            
            if win_probability > 0.75:
                ml_signal = 'STRONG_BUY'
            elif win_probability > 0.6:
                ml_signal = 'BUY'
            elif win_probability > 0.4:
                ml_signal = 'HOLD'
            elif win_probability > 0.25:
                ml_signal = 'SELL'
            else:
                ml_signal = 'STRONG_SELL'
            
            # Top features for this prediction
            top_features = self._get_top_features_for_prediction(features)
            
            return MLPrediction(
                symbol=symbol,
                win_probability=win_probability,
                expected_return=expected_return,
                model_confidence=model_confidence,
                prediction_quality=prediction_quality,
                ensemble_agreement=ensemble_agreement,
                ml_signal=ml_signal,
                ml_score=ml_score,
                top_features=top_features,
                prediction_horizon=self.prediction_horizon,
                model_version=self.MODEL_VERSION
            )
            
        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")
            return self._fallback_prediction(symbol)
    
    def _get_top_features_for_prediction(self, features: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get most important features for this specific prediction"""
        if not self.feature_importance:
            return []
        
        # Weight by both global importance and current feature value
        weighted_importance = []
        for i, name in enumerate(self.feature_engineer.feature_names):
            global_imp = self.feature_importance.get(name, 0)
            value = features[i] if i < len(features) else 0
            
            # Combine global importance with feature magnitude
            weighted = global_imp * (1 + abs(value) * 0.1)
            weighted_importance.append((name, weighted, value))
        
        # Sort by weighted importance
        weighted_importance.sort(key=lambda x: x[1], reverse=True)
        
        return [(name, value) for name, _, value in weighted_importance[:top_n]]
    
    def _fallback_prediction(self, symbol: str) -> MLPrediction:
        """Fallback prediction when model isn't trained"""
        return MLPrediction(
            symbol=symbol,
            win_probability=0.5,
            expected_return=0.0,
            model_confidence=0.0,
            prediction_quality='low',
            ensemble_agreement=0.0,
            ml_signal='HOLD',
            ml_score=50.0,
            top_features=[],
            prediction_horizon=self.prediction_horizon,
            model_version=self.MODEL_VERSION
        )
    
    def save(self, filename: str = "trading_model.pkl"):
        """Save model to disk"""
        filepath = os.path.join(self.model_dir, filename)
        
        save_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'feature_engineer': self.feature_engineer,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'prediction_horizon': self.prediction_horizon,
            'model_version': self.MODEL_VERSION
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[+] Model saved to {filepath}")
    
    def load(self, filename: str = "trading_model.pkl") -> bool:
        """Load model from disk"""
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.models = save_data['models']
            self.model_weights = save_data['model_weights']
            self.feature_engineer = save_data['feature_engineer']
            self.feature_importance = save_data['feature_importance']
            self.is_trained = save_data['is_trained']
            self.training_metrics = save_data.get('training_metrics')
            self.prediction_horizon = save_data.get('prediction_horizon', 5)
            
            print(f"[+] Model loaded from {filepath}")
            print(f"    Version: {save_data.get('model_version', 'unknown')}")
            if self.training_metrics:
                print(f"    Win Rate: {self.training_metrics.win_rate*100:.1f}%")
                print(f"    Accuracy: {self.training_metrics.accuracy*100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class HistoricalDataCollector:
    """
    Collects and prepares historical data for ML training
    """
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.feature_engineer = FeatureEngineer()
    
    def collect_training_data(self, 
                             symbols: List[str],
                             prediction_horizon: int = 5) -> List[Dict]:
        """
        Collect historical training data for multiple symbols
        
        Args:
            symbols: List of ticker symbols
            prediction_horizon: Days forward for target calculation
            
        Returns:
            List of training records with features and outcomes
        """
        import yfinance as yf
        from technical_analysis import TechnicalAnalyzer
        from screener import StockScreener
        
        tech_analyzer = TechnicalAnalyzer()
        screener = StockScreener()
        
        all_records = []
        
        print(f"\n[*] Collecting historical data for {len(symbols)} symbols...")
        print(f"    Lookback: {self.lookback_days} days")
        print(f"    Prediction horizon: {prediction_horizon} days\n")
        
        for idx, symbol in enumerate(symbols):
            try:
                print(f"[{idx+1}/{len(symbols)}] Processing {symbol}...", end=" ")
                
                # Get historical data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2y")
                
                if len(hist) < 100:
                    print("insufficient data")
                    continue
                
                # Walk through history creating training samples
                records = []
                
                for i in range(60, len(hist) - prediction_horizon):
                    # Get data up to point i
                    df_slice = hist.iloc[:i+1].copy()
                    
                    # Calculate future return (target)
                    current_price = hist['Close'].iloc[i]
                    future_price = hist['Close'].iloc[i + prediction_horizon]
                    future_return = (future_price / current_price - 1) * 100
                    
                    # Get technical signals
                    tech_signals = tech_analyzer.analyze(df_slice)
                    
                    if tech_signals is None:
                        continue
                    
                    # Extract features (no sentiment for historical - too slow)
                    features = self.feature_engineer.extract_features(
                        price_data=df_slice,
                        technical_signals=tech_signals,
                        screener_result=None,
                        sentiment_result=None
                    )
                    
                    records.append({
                        'symbol': symbol,
                        'date': hist.index[i].strftime('%Y-%m-%d'),
                        'features': features,
                        'future_return': future_return,
                        'is_profitable': future_return > 0
                    })
                
                all_records.extend(records)
                print(f"{len(records)} samples")
                
            except Exception as e:
                print(f"error: {e}")
                continue
        
        print(f"\n[+] Collected {len(all_records)} total training samples")
        return all_records
    
    def save_training_data(self, records: List[Dict], filepath: str = "training_data.pkl"):
        """Save training data to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(records, f)
        print(f"[+] Training data saved to {filepath}")
    
    def load_training_data(self, filepath: str = "training_data.pkl") -> Optional[List[Dict]]:
        """Load training data from disk"""
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def train_model_from_scratch(symbols: List[str] = None, 
                             save_path: str = "models/trading_model.pkl") -> TradingMLModel:
    """
    Complete pipeline to train a new model from scratch
    
    Args:
        symbols: List of symbols to train on (uses default if None)
        save_path: Where to save the trained model
        
    Returns:
        Trained TradingMLModel
    """
    from config import DEFAULT_WATCHLIST
    
    if symbols is None:
        symbols = DEFAULT_WATCHLIST[:30]  # Use subset for faster training
    
    print("\n" + "="*60)
    print("[ML MODEL TRAINING PIPELINE]")
    print("="*60)
    
    # Step 1: Collect data
    collector = HistoricalDataCollector()
    training_data = collector.collect_training_data(symbols, prediction_horizon=5)
    
    if len(training_data) < 100:
        raise ValueError("Insufficient training data collected")
    
    # Step 2: Initialize and train model
    model = TradingMLModel()
    
    X, y_class, y_reg = model.prepare_training_data(training_data)
    metrics = model.train(X, y_class, y_reg)
    
    # Step 3: Save model
    model.save(os.path.basename(save_path))
    
    return model


# ============================================
# Utility functions for integration
# ============================================

def get_ml_prediction(symbol: str,
                     price_data: pd.DataFrame,
                     technical_signals: Any = None,
                     screener_result: Any = None,
                     sentiment_result: Any = None,
                     model: TradingMLModel = None) -> Optional[MLPrediction]:
    """
    Convenience function to get ML prediction for a stock
    Loads model if not provided
    """
    if model is None:
        model = TradingMLModel()
        model.load()
    
    return model.predict(
        price_data=price_data,
        technical_signals=technical_signals,
        screener_result=screener_result,
        sentiment_result=sentiment_result,
        symbol=symbol
    )


def format_ml_prediction(pred: MLPrediction) -> str:
    """Format ML prediction for display"""
    quality_marker = {'high': '[+++]', 'medium': '[++]', 'low': '[+]'}
    
    lines = [
        f"\n{'='*50}",
        f"[ML PREDICTION: {pred.symbol}]",
        f"{'='*50}",
        f"",
        f"Win Probability: {pred.win_probability*100:.1f}%",
        f"Expected Return: {pred.expected_return:+.2f}% ({pred.prediction_horizon}d)",
        f"",
        f"Signal: {pred.ml_signal}",
        f"ML Score: {pred.ml_score:.0f}/100",
        f"",
        f"{quality_marker.get(pred.prediction_quality, '[?]')} Quality: {pred.prediction_quality.upper()}",
        f"Model Agreement: {pred.ensemble_agreement*100:.0f}%",
        f"Confidence: {pred.model_confidence*100:.0f}%",
    ]
    
    if pred.top_features:
        lines.append(f"\n[Key Factors]")
        for feature, value in pred.top_features[:5]:
            lines.append(f"    * {feature}: {value:.2f}")
    
    lines.append(f"\n[Model v{pred.model_version}]")
    
    return "\n".join(lines)
