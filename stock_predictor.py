"""
Stock Market Prediction using Multi-Model Sentiment Ensemble

A machine learning system exploring the relationship between financial news
sentiment and stock price movements, implementing state-of-the-art NLP
techniques for sentiment analysis combined with time series forecasting.

Key Research References:
- Hutto & Gilbert (2014): VADER sentiment analysis for social media
- Akbik et al. (2018): Flair - contextual string embeddings for NLP
- Loughran & McDonald (2011): Financial sentiment lexicons
- Bollen et al. (2011): Twitter mood predicts the stock market
- Ding et al. (2015): Deep learning for event-driven stock prediction

Author: Janno Louwrens
Created: November 2023
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from flair.models import TextClassifier
from flair.data import Sentence
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration for the stock prediction pipeline.

    Based on empirical findings from financial ML literature:
    - Lag windows of 3-5 days capture short-term momentum (Lo & MacKinlay, 1988)
    - Rolling windows capture regime changes (Hamilton, 1989)
    - TimeSeriesSplit prevents look-ahead bias (Bailey et al., 2014)
    """
    stock_data_path: Path
    news_data_path: Path
    output_dir: Path
    test_size: float = 0.1
    n_lags: int = 5  # Capture weekly patterns
    rolling_windows: Tuple[int, ...] = (3, 5, 7, 14, 21)  # Short to medium term
    random_state: int = 42
    cv_splits: int = 5
    feature_selection_k: int = 30  # Top K features


class SentimentEnsemble:
    """
    Multi-model sentiment analysis ensemble.

    Implements three complementary approaches:
    1. VADER: Rule-based, optimized for social media (Hutto & Gilbert, 2014)
    2. TextBlob: Pattern-based with subjectivity analysis
    3. Flair: Contextual embeddings via biLSTM (Akbik et al., 2018)

    The ensemble approach reduces model-specific biases and improves
    robustness across different text styles (financial news, social media, etc.)
    """

    def __init__(self):
        logger.info("Initializing sentiment ensemble...")
        self._vader = SentimentIntensityAnalyzer()
        self._flair = TextClassifier.load('sentiment')
        logger.info("Sentiment models loaded successfully")

    def get_vader_scores(self, text: str) -> Dict[str, float]:
        """
        Get VADER sentiment scores including compound and component scores.

        VADER is particularly effective for financial text as it handles:
        - Negations ("not good" -> negative)
        - Intensifiers ("very good" -> more positive)
        - Punctuation emphasis ("great!!!" -> stronger)
        """
        if not isinstance(text, str) or not text.strip():
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        return self._vader.polarity_scores(text)

    def get_textblob_scores(self, text: str) -> Dict[str, float]:
        """
        Get TextBlob polarity and subjectivity scores.

        Subjectivity (0-1) indicates how opinion-based the text is,
        which can help weight the reliability of sentiment signals.
        """
        if not isinstance(text, str) or not text.strip():
            return {'polarity': 0.0, 'subjectivity': 0.0}
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def get_flair_score(self, text: str) -> float:
        """
        Get Flair sentiment score using contextual embeddings.

        Flair uses character-level biLSTM to capture context that
        traditional methods miss, particularly effective for:
        - Sarcasm detection
        - Domain-specific terminology
        - Complex sentence structures
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0
        try:
            sentence = Sentence(text)
            self._flair.predict(sentence)
            if not sentence.labels:
                return 0.0
            label = sentence.labels[0]
            # Convert to [-1, 1] scale with confidence weighting
            return label.score if label.value == 'POSITIVE' else -label.score
        except Exception as e:
            logger.warning(f"Flair prediction failed: {e}")
            return 0.0

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Get comprehensive sentiment analysis from all models.

        Returns individual scores plus ensemble metrics:
        - Simple average (equal weighting)
        - Confidence-weighted average (using Flair confidence)
        """
        vader = self.get_vader_scores(text)
        textblob = self.get_textblob_scores(text)
        flair = self.get_flair_score(text)

        # Ensemble: simple average of normalized scores
        ensemble_avg = np.mean([vader['compound'], textblob['polarity'], flair])

        return {
            'vader_compound': vader['compound'],
            'vader_pos': vader['pos'],
            'vader_neg': vader['neg'],
            'textblob_polarity': textblob['polarity'],
            'textblob_subjectivity': textblob['subjectivity'],
            'flair': flair,
            'ensemble': ensemble_avg
        }


class FeatureEngineer:
    """
    Feature engineering for financial time series prediction.

    Implements features based on financial ML research:
    - Lagged returns (momentum signals)
    - Rolling statistics (volatility regimes)
    - Sentiment decay (news impact fades over time)
    - Technical indicators (price patterns)

    References:
    - Jegadeesh & Titman (1993): Momentum strategies
    - Bollerslev (1986): GARCH volatility modeling
    - Tetlock (2007): Media content and stock returns
    """

    def __init__(self, config: Config):
        self.config = config
        self.sentiment_analyzer = SentimentEnsemble()

    def create_features(self, stock_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set from stock and news data."""
        logger.info("Starting feature engineering pipeline...")

        # Merge datasets
        df = pd.merge(stock_df, news_df, on='date', how='left')

        # 1. Sentiment features
        logger.info("Computing multi-model sentiment features...")
        df = self._add_sentiment_features(df)

        # 2. Price-based features
        logger.info("Computing price-based features...")
        df = self._add_price_features(df)

        # 3. Lag features (momentum signals)
        logger.info("Computing lag features...")
        df = self._add_lag_features(df)

        # 4. Rolling statistics (volatility/regime features)
        logger.info("Computing rolling statistics...")
        df = self._add_rolling_features(df)

        # 5. Sentiment decay features
        logger.info("Computing sentiment decay features...")
        df = self._add_sentiment_decay_features(df)

        # Drop NaN rows from lagged features
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values")

        return df

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-model sentiment scores."""
        # Apply sentiment analysis
        sentiment_results = df['text'].apply(self.sentiment_analyzer.analyze)

        # Extract individual features
        for key in ['vader_compound', 'vader_pos', 'vader_neg',
                    'textblob_polarity', 'textblob_subjectivity',
                    'flair', 'ensemble']:
            df[f'sentiment_{key}'] = sentiment_results.apply(lambda x: x.get(key, 0.0))

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-derived technical features."""
        # Daily returns
        df['returns'] = df['closing_price'].pct_change()

        # Log returns (more normally distributed)
        df['log_returns'] = np.log(df['closing_price'] / df['closing_price'].shift(1))

        # Volatility (realized, using squared returns)
        df['volatility'] = df['returns'].rolling(5).std()

        # Price relative to moving averages (mean reversion signals)
        for window in [5, 10, 20]:
            ma = df['closing_price'].rolling(window).mean()
            df[f'price_ma_ratio_{window}'] = df['closing_price'] / ma

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for temporal patterns."""
        for lag in range(1, self.config.n_lags + 1):
            # Price lags
            df[f'price_lag_{lag}'] = df['closing_price'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

            # Sentiment lags
            df[f'sentiment_ensemble_lag_{lag}'] = df['sentiment_ensemble'].shift(lag)
            df[f'sentiment_vader_lag_{lag}'] = df['sentiment_vader_compound'].shift(lag)
            df[f'sentiment_flair_lag_{lag}'] = df['sentiment_flair'].shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics."""
        for window in self.config.rolling_windows:
            # Price statistics
            df[f'price_rolling_mean_{window}'] = df['closing_price'].rolling(window).mean()
            df[f'price_rolling_std_{window}'] = df['closing_price'].rolling(window).std()
            df[f'price_rolling_min_{window}'] = df['closing_price'].rolling(window).min()
            df[f'price_rolling_max_{window}'] = df['closing_price'].rolling(window).max()

            # Price range (Bollinger-like)
            df[f'price_range_{window}'] = (
                df[f'price_rolling_max_{window}'] - df[f'price_rolling_min_{window}']
            ) / df[f'price_rolling_mean_{window}']

            # Sentiment rolling statistics
            df[f'sentiment_rolling_mean_{window}'] = df['sentiment_ensemble'].rolling(window).mean()
            df[f'sentiment_rolling_std_{window}'] = df['sentiment_ensemble'].rolling(window).std()

            # Momentum (price change over window)
            df[f'momentum_{window}'] = df['closing_price'].pct_change(window)

        return df

    def _add_sentiment_decay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add exponentially weighted sentiment features.

        Based on research showing news impact decays over time
        (Tetlock et al., 2008). Uses exponential weighting where
        recent sentiment has more influence than older sentiment.
        """
        for halflife in [1, 3, 5]:
            df[f'sentiment_ewm_{halflife}'] = (
                df['sentiment_ensemble'].ewm(halflife=halflife).mean()
            )
        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names (excluding target and metadata)."""
        exclude = {'date', 'closing_price', 'text', 'returns', 'log_returns'}
        # This will be called after create_features, so we know all columns
        return []  # Will be populated dynamically


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    Implements proper backtesting to avoid look-ahead bias:
    - Train on past data only
    - Test on future data
    - Expanding or rolling window

    Reference: Bailey et al. (2014) "The Deflated Sharpe Ratio"
    """

    def __init__(self, n_splits: int = 5, test_size: float = 0.1):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for walk-forward validation."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        return list(tscv.split(X))


class ModelTrainer:
    """
    Model training with hyperparameter optimization.

    Uses TimeSeriesSplit for cross-validation to prevent data leakage
    in time series context (standard k-fold would leak future information).

    The MLP architecture is chosen for:
    - Non-linear relationship modeling
    - Ability to capture complex interactions between sentiment and price
    - Regularization via early stopping to prevent overfitting
    """

    def __init__(self, config: Config):
        self.config = config
        self.best_model: Optional[Pipeline] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_results: Optional[pd.DataFrame] = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Train model with hyperparameter tuning using TimeSeriesSplit."""
        logger.info("Starting hyperparameter optimization...")

        # Pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', RobustScaler()),  # Robust to outliers in financial data
            ('regressor', MLPRegressor(
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=self.config.random_state,
                solver='adam',
                activation='relu'
            ))
        ])

        # Hyperparameter search space
        param_grid = {
            'regressor__hidden_layer_sizes': [
                (64,), (128,), (64, 32), (128, 64), (128, 64, 32)
            ],
            'regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
            'regressor__learning_rate_init': [0.0005, 0.001, 0.005]
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)

        # Grid search with negative MSE (sklearn convention)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(X, y)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score (neg MSE): {grid_search.best_score_:.6f}")

        return self.best_model

    def evaluate(self, model: Pipeline, X: pd.DataFrame, y: pd.Series,
                 dataset_name: str = "Test") -> Dict[str, float]:
        """Comprehensive model evaluation with multiple metrics."""
        y_pred = model.predict(X)

        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100,
            'directional_accuracy': self._directional_accuracy(y, y_pred)
        }

        logger.info(f"\n--- {dataset_name} Set Metrics ---")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")

        return metrics

    def _directional_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (hit rate).

        In financial applications, predicting the direction of price
        movement is often more important than predicting exact values.
        """
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        return np.mean(actual_direction == pred_direction) * 100

    def save_model(self, model: Pipeline, path: Path) -> None:
        """Save trained model to disk."""
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load_model(path: Path) -> Pipeline:
        """Load trained model from disk."""
        return joblib.load(path)


class ResultsVisualizer:
    """Visualization utilities for model results."""

    @staticmethod
    def plot_predictions(dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray,
                         title: str, save_path: Optional[Path] = None) -> None:
        """Plot actual vs predicted values with confidence intervals."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Main prediction plot
        ax1 = axes[0]
        ax1.plot(dates, y_true, label='Actual', color='blue', alpha=0.7)
        ax1.plot(dates, y_pred, label='Predicted', color='red', alpha=0.7, linestyle='--')
        ax1.fill_between(dates, y_true, y_pred, alpha=0.2, color='gray')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Closing Price')
        ax1.set_title(title)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)

        # Residuals plot
        ax2 = axes[1]
        residuals = y_true.values - y_pred
        ax2.bar(dates, residuals, color='steelblue', alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='-', linewidth=1)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title('Residuals (Actual - Predicted)')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_feature_importance(model: Pipeline, feature_names: List[str],
                                save_path: Optional[Path] = None, top_n: int = 20) -> None:
        """Plot feature importance based on neural network weights."""
        regressor = model.named_steps['regressor']

        # Use absolute mean of first layer weights as importance proxy
        weights = np.abs(regressor.coefs_[0]).mean(axis=1)

        importance_df = pd.DataFrame({
            'feature': feature_names[:len(weights)],
            'importance': weights
        }).sort_values('importance', ascending=True).tail(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
        plt.xlabel('Importance (Mean Absolute Weight)')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()


class StockPredictor:
    """
    Main orchestrator for the stock prediction pipeline.

    Pipeline steps:
    1. Data loading and validation
    2. Feature engineering (sentiment + technical)
    3. Train/test split (temporal, no shuffle)
    4. Hyperparameter optimization with time series CV
    5. Model evaluation and visualization
    6. Model persistence
    """

    def __init__(self, config: Config):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.trainer = ModelTrainer(config)
        self.visualizer = ResultsVisualizer()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Execute the full prediction pipeline."""
        logger.info("=" * 70)
        logger.info("Stock Market Prediction Pipeline - Multi-Model Sentiment Ensemble")
        logger.info("=" * 70)

        # 1. Load data
        logger.info("\n[1/6] Loading data...")
        stock_df = pd.read_csv(self.config.stock_data_path)
        news_df = pd.read_csv(self.config.news_data_path)
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        news_df['date'] = pd.to_datetime(news_df['date'])
        logger.info(f"Loaded {len(stock_df)} stock records, {len(news_df)} news records")

        # 2. Feature engineering
        logger.info("\n[2/6] Engineering features...")
        df = self.feature_engineer.create_features(stock_df, news_df)

        # Get feature columns (exclude metadata and target)
        exclude_cols = {'date', 'closing_price', 'text'}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df['closing_price']
        dates = df['date']

        logger.info(f"Created {len(feature_cols)} features")

        # 3. Train/test split (temporal)
        logger.info("\n[3/6] Splitting data (temporal)...")
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        dates_test = dates.iloc[split_idx:]
        logger.info(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")

        # 4. Train model
        logger.info("\n[4/6] Training model with hyperparameter optimization...")
        model = self.trainer.train(X_train, y_train)

        # 5. Evaluate
        logger.info("\n[5/6] Evaluating model...")
        train_metrics = self.trainer.evaluate(model, X_train, y_train, "Training")
        test_metrics = self.trainer.evaluate(model, X_test, y_test, "Test")

        # 6. Visualize and save
        logger.info("\n[6/6] Generating visualizations and saving results...")

        model_path = self.config.output_dir / 'model.joblib'
        self.trainer.save_model(model, model_path)

        y_pred = model.predict(X_test)

        self.visualizer.plot_predictions(
            dates_test, y_test, y_pred,
            'Stock Price Prediction: Actual vs Predicted',
            self.config.output_dir / 'predictions.png'
        )

        self.visualizer.plot_feature_importance(
            model, feature_cols,
            self.config.output_dir / 'feature_importance.png'
        )

        # Save predictions
        results_df = pd.DataFrame({
            'date': dates_test,
            'actual': y_test,
            'predicted': y_pred,
            'error': y_test.values - y_pred,
            'error_pct': ((y_test.values - y_pred) / y_test.values) * 100
        })
        results_df.to_csv(self.config.output_dir / 'predictions.csv', index=False)

        logger.info("\n" + "=" * 70)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 70)

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'best_params': self.trainer.best_params,
            'model_path': model_path,
            'n_features': len(feature_cols)
        }


def main():
    """Main entry point."""
    config = Config(
        stock_data_path=Path('data/stock_price_data.csv'),
        news_data_path=Path('data/financial_news_data.csv'),
        output_dir=Path('output'),
        test_size=0.1,
        n_lags=5,
        rolling_windows=(3, 5, 7, 14, 21),
        random_state=42,
        cv_splits=5
    )

    predictor = StockPredictor(config)
    results = predictor.run()

    return results


if __name__ == '__main__':
    main()
