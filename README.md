# Stock Market Prediction using Multi-Model Sentiment Ensemble

A machine learning system exploring the relationship between financial news sentiment and stock price movements, implementing state-of-the-art NLP techniques combined with time series forecasting.

**Author:** Janno Louwrens
**Created:** November 2023

## Research Foundation

This project implements techniques from the following academic literature:

| Paper | Year | Technique Used |
|-------|------|----------------|
| Hutto & Gilbert, "VADER: A Parsimonious Rule-based Model for Sentiment Analysis" | 2014 | VADER sentiment scoring |
| Akbik et al., "Contextual String Embeddings for Sequence Labeling" | 2018 | Flair NLP embeddings |
| Loughran & McDonald, "When is a Liability not a Liability?" | 2011 | Financial sentiment lexicons |
| Bollen et al., "Twitter mood predicts the stock market" | 2011 | Social sentiment → price correlation |
| Ding et al., "Deep Learning for Event-Driven Stock Prediction" | 2015 | Event-driven features |
| Jegadeesh & Titman, "Returns to Buying Winners and Selling Losers" | 1993 | Momentum features |
| Tetlock, "Giving Content to Investor Sentiment" | 2007 | Media content → returns |
| Bailey et al., "The Deflated Sharpe Ratio" | 2014 | Walk-forward validation |

## Design Rationale

I approached this as a research exploration rather than a trading bot. Early experiments showed single sentiment models were noisy, so I built an ensemble (VADER, TextBlob, Flair) and paired it with rigorous time-series validation to avoid look-ahead bias.

The feature pipeline intentionally mixes news sentiment with momentum and rolling statistics so I could measure which signals actually matter. The final result documents both the marginal predictive lift and the practical limits of forecasting, which was the most valuable outcome of the project.

## Key Features

### Multi-Model Sentiment Ensemble
Three complementary NLP approaches reduce model-specific biases:

1. **VADER** - Rule-based sentiment optimized for social media text
   - Handles negations, intensifiers, punctuation emphasis
   - Returns compound + component scores (positive, negative, neutral)

2. **TextBlob** - Pattern-based NLP with subjectivity analysis
   - Polarity score [-1, 1]
   - Subjectivity score [0, 1] - weights sentiment reliability

3. **Flair** - Deep learning contextual embeddings (biLSTM)
   - Character-level representations capture context
   - Effective for sarcasm, domain-specific terminology

### Feature Engineering Pipeline
- **Lag features** - Momentum signals (Jegadeesh & Titman, 1993)
- **Rolling statistics** - Volatility regimes, Bollinger-like indicators
- **Sentiment decay** - Exponentially weighted averages (Tetlock et al., 2008)
- **Technical indicators** - Moving average ratios, price momentum

### Proper Time Series Methodology
- **TimeSeriesSplit** cross-validation prevents look-ahead bias
- **Temporal train/test split** - no shuffling
- **Walk-forward validation** for realistic backtesting
- **Directional accuracy** metric for trading relevance

## Project Structure

```
stock-market-predictions/
├── stock_predictor.py      # Main ML pipeline
├── data/
│   ├── stock_price_data.csv
│   ├── financial_news_data.csv
│   └── Tesla/              # Tesla-specific datasets
├── archive/                # Original experimental scripts
├── output/                 # Generated predictions and plots
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/stock-market-predictions.git
cd stock-market-predictions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage

### Quick Start

```bash
python stock_predictor.py
```

### Programmatic Usage

```python
from pathlib import Path
from stock_predictor import Config, StockPredictor

config = Config(
    stock_data_path=Path('data/stock_price_data.csv'),
    news_data_path=Path('data/financial_news_data.csv'),
    output_dir=Path('output'),
    test_size=0.1,
    n_lags=5,
    rolling_windows=(3, 5, 7, 14, 21),
    cv_splits=5
)

predictor = StockPredictor(config)
results = predictor.run()

print(f"Test R²: {results['test_metrics']['r2']:.4f}")
print(f"Directional Accuracy: {results['test_metrics']['directional_accuracy']:.1f}%")
```

## Architecture

```
                        ┌─────────────────────┐
                        │   Financial News    │
                        └──────────┬──────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │    VADER      │     │   TextBlob    │     │    Flair      │
    │  (Rule-based) │     │  (Pattern)    │     │  (biLSTM)     │
    └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
            │                     │                     │
            └──────────────┬──────┴──────┬──────────────┘
                           │             │
                    ┌──────▼─────┐ ┌─────▼──────┐
                    │  Ensemble  │ │ Subjectivity│
                    │   Score    │ │  Weighting  │
                    └──────┬─────┘ └─────┬───────┘
                           │             │
            ┌──────────────┼─────────────┼──────────────┐
            │              │             │              │
    ┌───────▼──────┐ ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
    │ Lag Features │ │ Rolling   │ │ Sentiment │ │ Technical │
    │ (Momentum)   │ │ Stats     │ │ Decay     │ │ Indicators│
    └───────┬──────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
            │              │             │              │
            └──────────────┴──────┬──────┴──────────────┘
                                  │
                           ┌──────▼──────┐
                           │ RobustScaler │
                           └──────┬──────┘
                                  │
                           ┌──────▼──────┐
                           │ MLPRegressor │
                           │ (GridSearchCV│
                           │ TimeSeriesSplit)
                           └──────┬──────┘
                                  │
                           ┌──────▼──────┐
                           │ Predictions │
                           └─────────────┘
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |
| MAPE | Mean Absolute Percentage Error |
| Directional Accuracy | % correct direction predictions |

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_size` | 0.1 | Fraction held out for testing |
| `n_lags` | 5 | Number of lagged features |
| `rolling_windows` | (3,5,7,14,21) | Rolling statistic windows |
| `cv_splits` | 5 | TimeSeriesSplit folds |
| `feature_selection_k` | 30 | Top K features to select |

## Key Findings

After extensive experimentation, this project demonstrates:

1. **Sentiment-Price Correlation Exists** - But is weak and noisy
2. **Ensemble Outperforms Single Models** - Reduces bias across text styles
3. **Temporal Features Dominate** - Price momentum > sentiment signals
4. **Market Efficiency Limits Prediction** - News is priced in quickly
5. **Directional Accuracy ~55%** - Slight edge over random

### Conclusion

While sentiment analysis provides marginal predictive value, the efficient market hypothesis holds: consistent prediction is extremely difficult. The primary value is in understanding the relationship between media content and market behavior, not in developing a trading system.

## References

```bibtex
@inproceedings{hutto2014vader,
  title={VADER: A Parsimonious Rule-based Model for Sentiment Analysis},
  author={Hutto, Clayton J and Gilbert, Eric},
  booktitle={ICWSM},
  year={2014}
}

@inproceedings{akbik2018flair,
  title={Contextual String Embeddings for Sequence Labeling},
  author={Akbik, Alan and Blythe, Duncan and Vollgraf, Roland},
  booktitle={COLING},
  year={2018}
}

@article{bollen2011twitter,
  title={Twitter mood predicts the stock market},
  author={Bollen, Johan and Mao, Huina and Zeng, Xiaojun},
  journal={Journal of Computational Science},
  year={2011}
}

@article{tetlock2007giving,
  title={Giving Content to Investor Sentiment},
  author={Tetlock, Paul C},
  journal={The Journal of Finance},
  year={2007}
}
```

## License

MIT License
