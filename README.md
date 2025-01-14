# Financial Market Prediction Using Hybrid Transformer-LSTM Model

## Overview
A deep learning model combining Transformer and LSTM architectures to predict S&P 500 price movements using multiple economic indicators. The model leverages both global context understanding and sequential pattern recognition for predictions. Moveover, gradient is used to analysis the influence of different indicator

## Model Architecture
![Model Architecture](https://github.com/Eggi111/SP500-Forecasting-via-Transformer-LSTM/blob/2909bdddb5837fbbcc6d99de095638d7ed4d1d0d/S%26P%20500%20Forecasting%20via%20Transformer-LSTM%20with%20Macroeconomic%20Data/img/workflow.png)

The hybrid architecture consists of:
- **Input Processing**: Handles multi-feature time series data with sliding windows
- **Transformer Encoder**: Captures global context and parallel processing within windows
- **LSTM Layers**: Processes enriched features while maintaining temporal order
- **Output Layer**: Generates predictions for specified time horizons

## Data Features
![Sample Data](https://github.com/Eggi111/SP500-Forecasting-via-Transformer-LSTM/blob/2909bdddb5837fbbcc6d99de095638d7ed4d1d0d/S%26P%20500%20Forecasting%20via%20Transformer-LSTM%20with%20Macroeconomic%20Data/img/table%20(raw%20data).png)

Key indicators include:
- **Market Indicators**: S&P 500 Price, Volume, BTC, Gold, Crude Oil
- **Economic Indicators**: 
  - EFFR (Effective Federal Funds Rate)
  - CPI (Consumer Price Index)
  - GEU (Global Economic Uncertainty)
  - Treasury Yields (3m, 10y)
  - GDP, Imports/Exports
- **Other Metrics**: HY Index, HPI, PPI, DTI, Party

## Model Performance
![Prediction Results](https://github.com/Eggi111/SP500-Forecasting-via-Transformer-LSTM/blob/65cc0177b251b75c251dc380f05e6af5be9d258c/S%26P%20500%20Forecasting%20via%20Transformer-LSTM%20with%20Macroeconomic%20Data/img/Prediction.png)

### Results Analysis
- Strong tracking of overall market trends
- Effective capture of long-term price movements
- Reasonable prediction accuracy in various market conditions

## Feature Importance Analysis
![Feature Importance](https://github.com/Eggi111/SP500-Forecasting-via-Transformer-LSTM/blob/2909bdddb5837fbbcc6d99de095638d7ed4d1d0d/S%26P%20500%20Forecasting%20via%20Transformer-LSTM%20with%20Macroeconomic%20Data/img/explanation.png)

### Key Insights
- Temporal importance varies across different features
- Some features show stronger predictive power at specific time steps
- Complex interactions between economic indicators revealed


