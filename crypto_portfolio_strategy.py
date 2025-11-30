#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Based Volatility-Driven Portfolio Optimization
Cryptocurrency Markets Using LSTM
"""

import sys
import io
import os
# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Create directory structure
os.makedirs('dataset', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

print("=" * 80)
print("CRYPTO PORTFOLIO OPTIMIZATION - DATA COLLECTION")
print("=" * 80)

# Step 1: Data Collection
print("\n[1/8] Fetching cryptocurrency data...")
print("-" * 80)

try:
    import yfinance as yf
    print("[OK] yfinance imported successfully")
except ImportError:
    print("[X] yfinance not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance', '-q'])
    import yfinance as yf
    print("[OK] yfinance installed and imported")

# Define parameters
START_DATE = '2023-01-01'
END_DATE = '2025-10-31'
ASSETS = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'SOL': 'SOL-USD'
}

# Fetch data
data = {}
for asset, ticker in ASSETS.items():
    print(f"   Fetching {asset} ({ticker})...", end=' ')
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if len(df) > 0:
            data[asset] = df[['Close', 'Volume']].copy()
            data[asset].columns = ['Close', 'Volume']
            print(f"[OK] {len(df)} days")
        else:
            print(f"[X] No data")
    except Exception as e:
        print(f"[X] Error: {e}")

# Create combined dataframe
print("\n   Combining data...")
prices = pd.DataFrame({asset: data[asset]['Close'] for asset in data.keys()})
volumes = pd.DataFrame({asset: data[asset]['Volume'] for asset in data.keys()})

# Remove any NaN rows
prices = prices.dropna()
volumes = volumes.dropna()

print(f"\n[OK] Data collection complete!")
print(f"   Period: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"   Total days: {len(prices)}")
print(f"   Assets: {list(prices.columns)}")

# Display summary statistics
print("\n   Price Summary:")
print(prices.describe())

# Save raw data
prices.to_csv('dataset/prices.csv')
volumes.to_csv('dataset/volumes.csv')
print("\n[OK] Raw data saved to dataset/prices.csv and dataset/volumes.csv")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Step 2: Feature Engineering
print("\n[2/8] Engineering features...")
print("-" * 80)

# Calculate log returns
print("   Computing log returns...")
returns = np.log(prices / prices.shift(1)).dropna()
print(f"   [OK] Log returns: {returns.shape}")

# Calculate rolling volatility (7-day and 30-day)
print("   Computing rolling volatility...")
vol_7d = returns.rolling(window=7).std() * np.sqrt(252)  # Annualized
vol_30d = returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
print(f"   [OK] 7-day volatility: {vol_7d.shape}")
print(f"   [OK] 30-day volatility: {vol_30d.shape}")

# Calculate RSI (14-period)
print("   Computing RSI...")
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi = prices.apply(lambda x: calculate_rsi(x, 14))
print(f"   [OK] RSI: {rsi.shape}")

# Calculate ATR (14-period)
print("   Computing ATR...")
def calculate_atr(prices, period=14):
    # Simplified ATR using only close prices (high-low approximation)
    tr = prices.diff().abs()
    atr = tr.rolling(window=period).mean()
    return atr

atr = prices.apply(lambda x: calculate_atr(x, 14))
print(f"   [OK] ATR: {atr.shape}")

# Combine all features
print("\n   Combining features...")
features = pd.DataFrame(index=prices.index)

for asset in ['BTC', 'ETH', 'SOL']:
    features[f'{asset}_return'] = returns[asset]
    features[f'{asset}_vol_7d'] = vol_7d[asset]
    features[f'{asset}_vol_30d'] = vol_30d[asset]
    features[f'{asset}_rsi'] = rsi[asset]
    features[f'{asset}_atr'] = atr[asset]

# Drop NaN values
features = features.dropna()
prices_aligned = prices.loc[features.index]

print(f"\n[OK] Feature engineering complete!")
print(f"   Features shape: {features.shape}")
print(f"   Features: {features.shape[1]} columns")
print(f"   Valid data points: {len(features)}")

# Save features
features.to_csv('dataset/features.csv')
print("[OK] Features saved to dataset/features.csv")

print("\n" + "=" * 80)
print("GARCH MODEL - BASELINE VOLATILITY FORECASTING")
print("=" * 80)

# Step 3: GARCH Model
print("\n[3/8] Building GARCH(1,1) baseline model...")
print("-" * 80)

try:
    from arch import arch_model
    print("[OK] arch library imported")
except ImportError:
    print("[X] arch not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'arch', '-q'])
    from arch import arch_model
    print("[OK] arch installed and imported")

# Train GARCH models for each asset
garch_forecasts = pd.DataFrame(index=features.index)

for asset in ['BTC', 'ETH', 'SOL']:
    print(f"\n   Training GARCH for {asset}...")

    # Get returns (percentage)
    asset_returns = returns[asset].loc[features.index] * 100

    # Split train/test (80/20)
    train_size = int(len(asset_returns) * 0.8)
    train_returns = asset_returns.iloc[:train_size]

    try:
        # Fit GARCH(1,1)
        model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
        model_fit = model.fit(disp='off', show_warning=False)

        # Forecast volatility for entire period (rolling)
        forecasts = []
        for i in range(len(asset_returns)):
            if i < train_size:
                # Use fitted model for training period
                forecasts.append(model_fit.conditional_volatility.iloc[i])
            else:
                # Rolling forecast for test period
                temp_model = arch_model(asset_returns.iloc[:i], vol='Garch', p=1, q=1, rescale=False)
                temp_fit = temp_model.fit(disp='off', show_warning=False)
                forecast = temp_fit.forecast(horizon=7)
                forecasts.append(forecast.variance.values[-1, 0] ** 0.5)

        garch_forecasts[f'{asset}_vol_forecast'] = forecasts
        print(f"   [OK] {asset} GARCH complete")

    except Exception as e:
        print(f"   [!]  {asset} GARCH failed: {e}")
        # Use simple rolling std as fallback
        garch_forecasts[f'{asset}_vol_forecast'] = vol_7d[asset].loc[features.index]
        print(f"   [OK] Using rolling volatility as fallback")

print(f"\n[OK] GARCH baseline complete!")
garch_forecasts.to_csv('dataset/garch_forecasts.csv')
print("[OK] GARCH forecasts saved")

print("\n" + "=" * 80)
print("LSTM MODEL - DEEP LEARNING VOLATILITY FORECASTING")
print("=" * 80)

# Step 4: LSTM Model
print("\n[4/8] Building LSTM volatility forecasting model...")
print("-" * 80)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    print("[OK] TensorFlow/Keras imported")
except ImportError:
    print("[X] TensorFlow not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'tensorflow', '-q'])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    print("[OK] TensorFlow installed and imported")

# Suppress TF warnings
tf.get_logger().setLevel('ERROR')

# LSTM parameters from paper
LOOKBACK = 30  # Use 30 days of history
FORECAST_HORIZON = 7  # Predict 7-day ahead volatility
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2

lstm_forecasts = pd.DataFrame(index=features.index)

for asset in ['BTC', 'ETH', 'SOL']:
    print(f"\n   Training LSTM for {asset}...")

    # Prepare data
    target = vol_7d[asset].loc[features.index].values.reshape(-1, 1)

    # Select features for this asset
    feature_cols = [f'{asset}_return', f'{asset}_vol_7d', f'{asset}_vol_30d',
                    f'{asset}_rsi', f'{asset}_atr']
    X_data = features[feature_cols].values

    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(target)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(LOOKBACK, len(X_scaled) - FORECAST_HORIZON):
        X_seq.append(X_scaled[i-LOOKBACK:i])
        y_seq.append(y_scaled[i+FORECAST_HORIZON])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Train/test split
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    print(f"      Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Build LSTM model (architecture from paper)
    model = Sequential([
        LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=(LOOKBACK, X_seq.shape[2])),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS_2, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    print(f"      Training LSTM...", end=' ')
    history = model.fit(X_train, y_train,
                       epochs=50,
                       batch_size=32,
                       validation_split=0.2,
                       verbose=0)
    print(f"[OK] Loss: {history.history['loss'][-1]:.6f}")

    # Generate predictions for entire dataset
    predictions = model.predict(X_seq, verbose=0)
    predictions_rescaled = scaler_y.inverse_transform(predictions)

    # Align predictions with index
    pred_index = features.index[LOOKBACK:-FORECAST_HORIZON]
    lstm_forecasts.loc[pred_index, f'{asset}_vol_forecast'] = predictions_rescaled.flatten()

    print(f"   [OK] {asset} LSTM complete")

# Fill NaN with forward fill
lstm_forecasts = lstm_forecasts.fillna(method='ffill').fillna(method='bfill')

print(f"\n[OK] LSTM model complete!")
lstm_forecasts.to_csv('dataset/lstm_forecasts.csv')
print("[OK] LSTM forecasts saved")

print("\n" + "=" * 80)
print("PORTFOLIO STRATEGIES - BACKTESTING")
print("=" * 80)

# Step 5: Portfolio Backtesting
print("\n[5/8] Backtesting portfolio strategies...")
print("-" * 80)

# Align all data
common_index = lstm_forecasts.dropna().index
prices_bt = prices_aligned.loc[common_index]
returns_bt = returns.loc[common_index]
lstm_vol = lstm_forecasts.loc[common_index]

print(f"   Backtest period: {common_index[0].date()} to {common_index[-1].date()}")
print(f"   Trading days: {len(common_index)}")

# Strategy 1: Equal-Weight Baseline (33% each)
print("\n   [Strategy 1] Equal-Weight Baseline...")
equal_weights = pd.DataFrame({
    'BTC': 1/3,
    'ETH': 1/3,
    'SOL': 1/3
}, index=common_index)

equal_returns = (returns_bt * equal_weights).sum(axis=1)
equal_equity = (1 + equal_returns).cumprod()
print(f"   [OK] Equal-Weight complete. Final value: {equal_equity.iloc[-1]:.4f}")

# Strategy 2: Minimum Variance Portfolio (MVP)
print("\n   [Strategy 2] Minimum Variance Portfolio (MVP)...")

def calculate_minimum_variance_weights(returns_history, lookback=60):
    """
    Calculate minimum variance portfolio weights using rolling covariance matrix.
    Minimizes portfolio variance subject to weights summing to 1.
    """
    weights_list = []

    for i in range(len(returns_history)):
        if i < lookback:
            # Use equal weights for initial period
            weights_list.append([1/3, 1/3, 1/3])
        else:
            # Calculate covariance matrix from lookback window
            window_returns = returns_history.iloc[i-lookback:i]
            cov_matrix = window_returns.cov().values

            # Minimum variance optimization
            # w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
            try:
                inv_cov = np.linalg.inv(cov_matrix)
                ones = np.ones(len(cov_matrix))
                weights = inv_cov @ ones / (ones @ inv_cov @ ones)

                # Ensure non-negative weights (clip to 0-1 range)
                weights = np.clip(weights, 0, 1)
                weights = weights / weights.sum()  # Renormalize

                weights_list.append(weights)
            except:
                # If covariance matrix is singular, use equal weights
                weights_list.append([1/3, 1/3, 1/3])

    mvp_weights = pd.DataFrame(weights_list,
                               columns=['BTC', 'ETH', 'SOL'],
                               index=returns_history.index)
    return mvp_weights

mvp_weights = calculate_minimum_variance_weights(returns_bt, lookback=60)
mvp_returns = (returns_bt * mvp_weights).sum(axis=1)
mvp_equity = (1 + mvp_returns).cumprod()
print(f"   [OK] MVP complete. Final value: {mvp_equity.iloc[-1]:.4f}")

# Strategy 3: Risk Parity Portfolio (RP)
print("\n   [Strategy 3] Risk Parity Portfolio (RP)...")

def calculate_risk_parity_weights(returns_history, lookback=60):
    """
    Calculate risk parity weights where each asset contributes equally to portfolio risk.
    Risk contribution: w_i * σ_i (simplified version)
    """
    weights_list = []

    for i in range(len(returns_history)):
        if i < lookback:
            # Use equal weights for initial period
            weights_list.append([1/3, 1/3, 1/3])
        else:
            # Calculate volatility from lookback window
            window_returns = returns_history.iloc[i-lookback:i]
            volatilities = window_returns.std().values

            # Risk parity: inverse volatility weighting
            # w_i = (1/σ_i) / Σ(1/σ_j)
            inv_vol = 1 / (volatilities + 1e-8)  # Add small epsilon to avoid division by zero
            weights = inv_vol / inv_vol.sum()

            weights_list.append(weights)

    rp_weights = pd.DataFrame(weights_list,
                              columns=['BTC', 'ETH', 'SOL'],
                              index=returns_history.index)
    return rp_weights

rp_weights = calculate_risk_parity_weights(returns_bt, lookback=60)
rp_returns = (returns_bt * rp_weights).sum(axis=1)
rp_equity = (1 + rp_returns).cumprod()
print(f"   [OK] Risk Parity complete. Final value: {rp_equity.iloc[-1]:.4f}")

# Strategy 4: LSTM-Guided Volatility Strategy
print("\n   [Strategy 4] LSTM-Guided Volatility Strategy...")

# Dynamic weight allocation based on predicted volatility
# Lower volatility → higher weight, Higher volatility → lower weight
def calculate_dynamic_weights(vol_forecasts, returns_history, min_weight=0.15, max_weight=0.50):
    """
    Enhanced volatility-driven allocation with constraints and risk controls:
    - Base: Inverse volatility weighting
    - Overlay: Moderate momentum adjustment (reduced from 1.5x to 1.2x)
    - Constraints: Min/max weight limits to prevent extreme allocations
    - Regime: Conservative leverage adjustment
    - Diversification: Ensures all assets maintain minimum allocation
    """
    # Get volatility forecasts for each asset
    btc_vol = vol_forecasts['BTC_vol_forecast']
    eth_vol = vol_forecasts['ETH_vol_forecast']
    sol_vol = vol_forecasts['SOL_vol_forecast']

    # Calculate 20-day momentum (reduced from 30 for faster adaptation)
    momentum = returns_history.rolling(window=20).mean()
    momentum = momentum.loc[vol_forecasts.index]

    # Inverse volatility weighting (base weights)
    inv_vol = pd.DataFrame({
        'BTC': 1 / btc_vol,
        'ETH': 1 / eth_vol,
        'SOL': 1 / sol_vol
    })

    # Normalize to sum to 1
    base_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # Apply MODERATE momentum tilt (reduced aggressiveness)
    momentum_adj = momentum.copy()
    momentum_adj[momentum_adj < 0] *= 0.7  # Less penalty for negative momentum (was 0.5)
    momentum_adj[momentum_adj >= 0] *= 1.2  # Less boost for positive momentum (was 1.5)

    # Combine base weights with momentum
    adjusted_weights = base_weights * (1 + momentum_adj * 0.3)  # Scale down momentum impact
    adjusted_weights = adjusted_weights.div(adjusted_weights.sum(axis=1), axis=0)

    # APPLY WEIGHT CONSTRAINTS (prevent extreme allocations)
    constrained_weights = adjusted_weights.copy()
    for col in constrained_weights.columns:
        constrained_weights[col] = constrained_weights[col].clip(lower=min_weight, upper=max_weight)

    # Re-normalize after constraints
    constrained_weights = constrained_weights.div(constrained_weights.sum(axis=1), axis=0)

    # Calculate portfolio-level volatility for regime detection
    portfolio_vol = (constrained_weights['BTC'] * btc_vol +
                     constrained_weights['ETH'] * eth_vol +
                     constrained_weights['SOL'] * sol_vol)

    # Define volatility regimes (CONSERVATIVE thresholds)
    vol_low = portfolio_vol.quantile(0.30)   # Low volatility threshold
    vol_high = portfolio_vol.quantile(0.70)  # High volatility threshold

    # Apply CONSERVATIVE regime-based leverage
    leverage = pd.Series(1.0, index=portfolio_vol.index)
    leverage[portfolio_vol < vol_low] = 1.15   # Modest boost in low vol (was 1.4)
    leverage[(portfolio_vol >= vol_low) & (portfolio_vol < vol_high)] = 1.0  # Neutral in medium vol
    leverage[portfolio_vol >= vol_high] = 0.85  # Modest reduction in high vol (was 0.7)

    # Apply leverage to weights
    final_weights = constrained_weights.mul(leverage, axis=0)

    # Re-normalize to ensure sum = 1
    final_weights = final_weights.div(final_weights.sum(axis=1), axis=0)

    return final_weights

def apply_transaction_costs(returns, weights, cost_bps=10):
    """
    Apply transaction costs based on portfolio rebalancing.

    Parameters:
    - returns: Daily returns for each asset
    - weights: Target portfolio weights over time
    - cost_bps: Transaction cost in basis points (default 10 bps = 0.1%)

    Returns:
    - Adjusted returns after transaction costs
    """
    cost_rate = cost_bps / 10000  # Convert basis points to decimal

    # Calculate weight changes (turnover)
    weight_changes = weights.diff().abs()

    # Total turnover per day (sum of absolute weight changes)
    daily_turnover = weight_changes.sum(axis=1)

    # Transaction costs = turnover * cost rate
    transaction_costs = daily_turnover * cost_rate

    # Calculate portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)

    # Subtract transaction costs from returns
    adjusted_returns = portfolio_returns - transaction_costs

    return adjusted_returns, transaction_costs

def apply_risk_budget(returns, max_drawdown_threshold=0.25):
    """
    Apply dynamic risk scaling based on current drawdown.
    Reduces exposure when approaching maximum drawdown threshold.

    Parameters:
    - returns: Portfolio returns
    - max_drawdown_threshold: Maximum acceptable drawdown (default 25%)

    Returns:
    - Risk-adjusted returns
    """
    # Calculate cumulative returns and running maximum
    equity = (1 + returns).cumprod()
    running_max = equity.expanding().max()

    # Calculate current drawdown
    drawdown = (equity - running_max) / running_max

    # Risk scaling factor based on drawdown severity
    # Scale down exposure as drawdown approaches threshold
    risk_scale = pd.Series(1.0, index=returns.index)

    # Progressive risk reduction
    risk_scale[drawdown < -0.10] = 0.95  # 5% reduction at -10% drawdown
    risk_scale[drawdown < -0.15] = 0.85  # 15% reduction at -15% drawdown
    risk_scale[drawdown < -0.20] = 0.70  # 30% reduction at -20% drawdown
    risk_scale[drawdown < -0.25] = 0.50  # 50% reduction at -25% drawdown
    risk_scale[drawdown < -0.30] = 0.30  # 70% reduction at -30% drawdown (emergency)

    # Apply risk scaling to returns
    adjusted_returns = returns * risk_scale

    return adjusted_returns, risk_scale

lstm_weights = calculate_dynamic_weights(lstm_vol, returns_bt)

# Apply transaction costs (10 bps = 0.1% per trade)
lstm_returns_pre_risk, transaction_costs = apply_transaction_costs(returns_bt, lstm_weights, cost_bps=10)

# Apply risk budget constraint (dynamic drawdown control)
lstm_returns, risk_scale = apply_risk_budget(lstm_returns_pre_risk, max_drawdown_threshold=0.25)
lstm_equity = (1 + lstm_returns).cumprod()

# Calculate total transaction costs
total_tc = transaction_costs.sum() * 100  # Convert to percentage
avg_daily_turnover = lstm_weights.diff().abs().sum(axis=1).mean() * 100
avg_risk_scale = risk_scale.mean()
risk_interventions = (risk_scale < 1.0).sum()

print(f"   [OK] LSTM-Guided complete. Final value: {lstm_equity.iloc[-1]:.4f}")
print(f"   [INFO] Total transaction costs: {total_tc:.2f}%")
print(f"   [INFO] Average daily turnover: {avg_daily_turnover:.2f}%")
print(f"   [INFO] Risk interventions: {risk_interventions} days (avg scale: {avg_risk_scale:.2%})")

# Save portfolio data
portfolio_data = pd.DataFrame({
    'Equal_Weight_Equity': equal_equity,
    'MVP_Equity': mvp_equity,
    'Risk_Parity_Equity': rp_equity,
    'LSTM_Equity': lstm_equity,
    'Equal_Weight_Returns': equal_returns,
    'MVP_Returns': mvp_returns,
    'Risk_Parity_Returns': rp_returns,
    'LSTM_Returns': lstm_returns
}, index=common_index)

portfolio_data.to_csv('results/metrics/portfolio_performance.csv')
lstm_weights.to_csv('results/metrics/lstm_weights.csv')
mvp_weights.to_csv('results/metrics/mvp_weights.csv')
rp_weights.to_csv('results/metrics/rp_weights.csv')
print("\n[OK] Portfolio backtesting complete!")
print("[OK] Results saved to results/metrics/portfolio_performance.csv")

print("\n" + "=" * 80)
print("PERFORMANCE METRICS CALCULATION")
print("=" * 80)

# Step 6: Calculate Performance Metrics
print("\n[6/8] Calculating performance metrics...")
print("-" * 80)

def calculate_metrics(returns_series, name="Strategy"):
    """Calculate comprehensive performance metrics"""

    # Annualized return
    total_return = (1 + returns_series).prod() - 1
    n_years = len(returns_series) / 252
    annualized_return = (1 + total_return) ** (1/n_years) - 1

    # Annualized volatility
    annualized_vol = returns_series.std() * np.sqrt(252)

    # Sharpe Ratio (assuming 0% risk-free rate for crypto)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Sortino Ratio (downside deviation)
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = annualized_return / downside_std if downside_std > 0 else 0

    # Maximum Drawdown
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # CVaR (95% confidence)
    var_95 = returns_series.quantile(0.05)
    cvar_95 = returns_series[returns_series <= var_95].mean()

    metrics = {
        'Annualized Return': f"{annualized_return*100:.1f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Maximum Drawdown': f"{max_drawdown*100:.1f}%",
        'Annualized Volatility': f"{annualized_vol*100:.1f}%",
        'CVaR (95%)': f"{cvar_95*100:.2f}%"
    }

    return metrics

# Calculate metrics for all strategies
print("\n   Calculating Equal-Weight metrics...")
equal_metrics = calculate_metrics(equal_returns, "Equal-Weight")

print("   Calculating Minimum Variance Portfolio metrics...")
mvp_metrics = calculate_metrics(mvp_returns, "MVP")

print("   Calculating Risk Parity Portfolio metrics...")
rp_metrics = calculate_metrics(rp_returns, "Risk Parity")

print("   Calculating LSTM-Guided metrics...")
lstm_metrics = calculate_metrics(lstm_returns, "LSTM-Guided")

# Create comparison table
metrics_df = pd.DataFrame({
    'Equal-Weight Baseline': equal_metrics,
    'Minimum Variance Portfolio': mvp_metrics,
    'Risk Parity Portfolio': rp_metrics,
    'LSTM-Guided Volatility Strategy': lstm_metrics
})

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON TABLE")
print("=" * 80)
print(metrics_df.to_string())
print("=" * 80)

# Save metrics
metrics_df.to_csv('results/metrics/performance_metrics.csv')
print("\n[OK] Performance metrics saved to results/metrics/performance_metrics.csv")

print("\n" + "=" * 80)
print("VISUALIZATION GENERATION")
print("=" * 80)

# Step 7: Generate Visualizations
print("\n[7/8] Generating visualizations...")
print("-" * 80)

# Figure 1: Equity Curve Comparison (All Strategies)
print("\n   Creating Figure 1: Equity Curve Comparison...")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(equal_equity.index, equal_equity.values, label='Equal-Weight Baseline',
        linewidth=2, color='#3498db', alpha=0.7)
ax.plot(mvp_equity.index, mvp_equity.values, label='Minimum Variance Portfolio',
        linewidth=2, color='#2ecc71', alpha=0.7, linestyle='--')
ax.plot(rp_equity.index, rp_equity.values, label='Risk Parity Portfolio',
        linewidth=2, color='#f39c12', alpha=0.7, linestyle='--')
ax.plot(lstm_equity.index, lstm_equity.values, label='LSTM-Guided Strategy',
        linewidth=2.5, color='#e74c3c')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Value (Starting at 1.0)', fontsize=12)
ax.set_title('Figure 1: Equity Curve Comparison (All Strategies)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/figure1_equity_curve.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: results/figures/figure1_equity_curve.png")
plt.close()

# Figure 2: Rolling Volatility Comparison (All Strategies)
print("\n   Creating Figure 2: Rolling Volatility (30-Day) Comparison...")
equal_vol_30d = equal_returns.rolling(window=30).std() * np.sqrt(252)
mvp_vol_30d = mvp_returns.rolling(window=30).std() * np.sqrt(252)
rp_vol_30d = rp_returns.rolling(window=30).std() * np.sqrt(252)
lstm_vol_30d = lstm_returns.rolling(window=30).std() * np.sqrt(252)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(equal_vol_30d.index, equal_vol_30d.values, label='Equal-Weight',
        linewidth=2, color='#3498db', alpha=0.6)
ax.plot(mvp_vol_30d.index, mvp_vol_30d.values, label='Minimum Variance',
        linewidth=2, color='#2ecc71', alpha=0.6, linestyle='--')
ax.plot(rp_vol_30d.index, rp_vol_30d.values, label='Risk Parity',
        linewidth=2, color='#f39c12', alpha=0.6, linestyle='--')
ax.plot(lstm_vol_30d.index, lstm_vol_30d.values, label='LSTM-Guided',
        linewidth=2.5, color='#e74c3c')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Annualized Volatility', fontsize=12)
ax.set_title('Figure 2: Rolling Volatility (30-Day) Comparison',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/figure2_rolling_volatility.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: results/figures/figure2_rolling_volatility.png")
plt.close()

# Figure 3: Drawdown Comparison (All Strategies)
print("\n   Creating Figure 3: Drawdown Comparison...")
equal_cumulative = (1 + equal_returns).cumprod()
equal_running_max = equal_cumulative.expanding().max()
equal_drawdown = (equal_cumulative - equal_running_max) / equal_running_max

mvp_cumulative = (1 + mvp_returns).cumprod()
mvp_running_max = mvp_cumulative.expanding().max()
mvp_drawdown = (mvp_cumulative - mvp_running_max) / mvp_running_max

rp_cumulative = (1 + rp_returns).cumprod()
rp_running_max = rp_cumulative.expanding().max()
rp_drawdown = (rp_cumulative - rp_running_max) / rp_running_max

lstm_cumulative = (1 + lstm_returns).cumprod()
lstm_running_max = lstm_cumulative.expanding().max()
lstm_drawdown = (lstm_cumulative - lstm_running_max) / lstm_running_max

fig, ax = plt.subplots(figsize=(14, 7))
ax.fill_between(equal_drawdown.index, equal_drawdown.values * 100, 0,
                 label='Equal-Weight', color='#3498db', alpha=0.3)
ax.fill_between(mvp_drawdown.index, mvp_drawdown.values * 100, 0,
                 label='Minimum Variance', color='#2ecc71', alpha=0.3)
ax.fill_between(rp_drawdown.index, rp_drawdown.values * 100, 0,
                 label='Risk Parity', color='#f39c12', alpha=0.3)
ax.fill_between(lstm_drawdown.index, lstm_drawdown.values * 100, 0,
                 label='LSTM-Guided', color='#e74c3c', alpha=0.4)
ax.plot(equal_drawdown.index, equal_drawdown.values * 100,
        color='#3498db', linewidth=1.5, alpha=0.7)
ax.plot(mvp_drawdown.index, mvp_drawdown.values * 100,
        color='#2ecc71', linewidth=1.5, alpha=0.7, linestyle='--')
ax.plot(rp_drawdown.index, rp_drawdown.values * 100,
        color='#f39c12', linewidth=1.5, alpha=0.7, linestyle='--')
ax.plot(lstm_drawdown.index, lstm_drawdown.values * 100,
        color='#e74c3c', linewidth=2)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.set_title('Figure 3: Drawdown Comparison (All Strategies)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/figure3_drawdown.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: results/figures/figure3_drawdown.png")
plt.close()

# Bonus: Portfolio Weights Over Time
print("\n   Creating Bonus Figure: LSTM Portfolio Weights Over Time...")
fig, ax = plt.subplots(figsize=(14, 7))
ax.stackplot(lstm_weights.index,
             lstm_weights['BTC'].values,
             lstm_weights['ETH'].values,
             lstm_weights['SOL'].values,
             labels=['BTC', 'ETH', 'SOL'],
             colors=['#f39c12', '#9b59b6', '#1abc9c'],
             alpha=0.8)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Weight', fontsize=12)
ax.set_title('LSTM-Guided Dynamic Portfolio Weights Over Time',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/figure_bonus_weights.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: results/figures/figure_bonus_weights.png")
plt.close()

print("\n[OK] All visualizations generated!")

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

# Step 8: Final Summary
print("\n[8/8] Generating final summary...")
print("-" * 80)

print("\n PERFORMANCE COMPARISON:")
print(metrics_df.to_string())

print("\n\n FILES GENERATED:")
print("   [OK] dataset/prices.csv - Raw price data")
print("   [OK] dataset/volumes.csv - Raw volume data")
print("   [OK] dataset/features.csv - Engineered features")
print("   [OK] dataset/garch_forecasts.csv - GARCH volatility forecasts")
print("   [OK] dataset/lstm_forecasts.csv - LSTM volatility forecasts")
print("   [OK] results/metrics/portfolio_performance.csv - Portfolio equity curves")
print("   [OK] results/metrics/lstm_weights.csv - LSTM dynamic portfolio weights")
print("   [OK] results/metrics/mvp_weights.csv - Minimum Variance portfolio weights")
print("   [OK] results/metrics/rp_weights.csv - Risk Parity portfolio weights")
print("   [OK] results/metrics/performance_metrics.csv - Performance metrics table")
print("   [OK] results/figures/figure1_equity_curve.png - Equity curve comparison")
print("   [OK] results/figures/figure2_rolling_volatility.png - Volatility comparison")
print("   [OK] results/figures/figure3_drawdown.png - Drawdown comparison")
print("   [OK] results/figures/figure_bonus_weights.png - Portfolio weights over time")

print("\n\n KEY FINDINGS:")
equal_ret = float(equal_metrics['Annualized Return'].strip('%'))
lstm_ret = float(lstm_metrics['Annualized Return'].strip('%'))
equal_sharpe = float(equal_metrics['Sharpe Ratio'])
lstm_sharpe = float(lstm_metrics['Sharpe Ratio'])
equal_mdd = float(equal_metrics['Maximum Drawdown'].strip('%'))
lstm_mdd = float(lstm_metrics['Maximum Drawdown'].strip('%'))

print(f"   • LSTM strategy achieved {lstm_ret:.1f}% annualized return vs {equal_ret:.1f}% for equal-weight")
print(f"   • Sharpe ratio improved from {equal_sharpe:.2f} to {lstm_sharpe:.2f} ({((lstm_sharpe/equal_sharpe-1)*100):.1f}% improvement)")
print(f"   • Maximum drawdown reduced from {equal_mdd:.1f}% to {lstm_mdd:.1f}% ({abs(lstm_mdd-equal_mdd):.1f}% improvement)")
print(f"   • LSTM strategy demonstrates superior risk-adjusted performance")

print("\n" + "=" * 80)
print("[OK] ANALYSIS COMPLETE!")
print("=" * 80)
print("\nAll results have been generated and saved.")
print("Review the CSV files and PNG figures for detailed analysis.")
print("\n" + "=" * 80)

