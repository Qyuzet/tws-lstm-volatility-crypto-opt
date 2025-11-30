# AI-Based Volatility-Driven Portfolio Optimization for Cryptocurrency Markets

This repository contains the complete implementation of an LSTM-based volatility forecasting system for cryptocurrency portfolio optimization, as described in the accompanying academic paper.

## Project Structure

```
suli-lstm-volatility-crypto-opt/
├── crypto_portfolio_strategy.py    # Main implementation script
├── paper.md                         # Academic paper
├── IMPLEMENTATION_SUMMARY.md        # Detailed implementation documentation
├── README.md                        # This file
├── dataset/                         # Dataset folder (generated)
│   ├── prices.csv                   # Raw price data
│   ├── volumes.csv                  # Raw volume data
│   ├── features.csv                 # Engineered features
│   ├── garch_forecasts.csv          # GARCH volatility predictions
│   └── lstm_forecasts.csv           # LSTM volatility predictions
└── results/                         # Results folder (generated)
    ├── metrics/                     # Performance metrics
    │   ├── portfolio_performance.csv
    │   ├── lstm_weights.csv
    │   ├── mvp_weights.csv
    │   ├── rp_weights.csv
    │   └── performance_metrics.csv
    └── figures/                     # Visualization charts
        ├── figure1_equity_curve.png
        ├── figure2_rolling_volatility.png
        ├── figure3_drawdown.png
        └── figure_bonus_weights.png
```

## Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn yfinance arch tensorflow
```

### Run the Analysis

```bash
python crypto_portfolio_strategy.py
```

The script will:

1. Fetch cryptocurrency data (BTC, ETH, SOL) from Yahoo Finance
2. Engineer features (log returns, volatility, RSI, ATR)
3. Train GARCH(1,1) baseline model
4. Train LSTM volatility forecasting model
5. Backtest all portfolio strategies (Equal-Weight, MVP, Risk Parity, LSTM-Guided)
6. Calculate performance metrics for all strategies
7. Generate visualization charts comparing all strategies

All results will be saved to `dataset/` and `results/` folders.

## Key Features

- **Data Collection**: Automated fetching from Yahoo Finance
- **Feature Engineering**: Log returns, rolling volatility, RSI, ATR
- **GARCH Baseline**: GARCH(1,1) volatility forecasting
- **LSTM Model**: 2-layer LSTM (64→32 units) with dropout
- **Portfolio Strategies**:
  - Equal-Weight Baseline (33.3% each)
  - Minimum Variance Portfolio (MVP)
  - Risk Parity Portfolio (RP)
  - LSTM-Guided Dynamic Allocation (volatility-driven with constraints, transaction costs, and risk budget)
- **Performance Metrics**: Sharpe, Sortino, MDD, CVaR, annualized returns
- **Visualizations**: Equity curves, volatility comparison, drawdown analysis, portfolio weights

## Results

The implementation generates comprehensive results including:

- Performance comparison table
- Equity curve charts
- Rolling volatility analysis
- Drawdown comparison
- Dynamic portfolio weights over time

See `IMPLEMENTATION_SUMMARY.md` for detailed results and analysis.

## Files Description

### Main Files

- `crypto_portfolio_strategy.py` - Complete implementation (875 lines)
- `paper.md` - Academic paper describing the methodology
- `IMPLEMENTATION_SUMMARY.md` - Detailed implementation documentation

### Data Files (in `dataset/`)

- `prices.csv` - Historical price data for BTC, ETH, SOL
- `volumes.csv` - Historical volume data
- `features.csv` - Engineered features (15 columns, 1004 rows)
- `garch_forecasts.csv` - GARCH volatility predictions
- `lstm_forecasts.csv` - LSTM volatility predictions

### Results Files (in `results/`)

- `metrics/portfolio_performance.csv` - Daily equity curves and returns for all strategies
- `metrics/lstm_weights.csv` - LSTM dynamic portfolio weights over time
- `metrics/mvp_weights.csv` - Minimum Variance Portfolio weights over time
- `metrics/rp_weights.csv` - Risk Parity Portfolio weights over time
- `metrics/performance_metrics.csv` - Summary performance table (all 4 strategies)
- `figures/figure1_equity_curve.png` - Equity curve comparison (all 4 strategies)
- `figures/figure2_rolling_volatility.png` - Volatility comparison (all 4 strategies)
- `figures/figure3_drawdown.png` - Drawdown analysis (all 4 strategies)
- `figures/figure_bonus_weights.png` - LSTM portfolio weights visualization

## Configuration

Key parameters in `crypto_portfolio_strategy.py`:

- **Assets**: BTC-USD, ETH-USD, SOL-USD
- **Period**: January 2023 - October 2025
- **LSTM Architecture**: 64→32 units, dropout 0.2
- **Lookback Window**: 30 days
- **Forecast Horizon**: 7 days
- **Training**: 50 epochs, batch size 32
- **Train/Test Split**: 80/20

## Citation

If you use this code in your research, please cite the accompanying paper.

## License

This project is for academic and research purposes.

## Contributing

This is an academic research project. For questions or suggestions, please open an issue.

## Disclaimer

This implementation is for research and educational purposes only. It is not financial advice. Cryptocurrency investments carry significant risk.
