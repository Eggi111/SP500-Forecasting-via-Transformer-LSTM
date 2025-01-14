# Financial Market Prediction Using Hybrid Transformer-LSTM Model

## Overview
A deep learning model combining Transformer and LSTM architectures to predict S&P 500 price movements using multiple economic indicators. The model leverages both global context understanding and sequential pattern recognition for accurate market predictions.

## Model Architecture
![Model Architecture](img/workflow.png)

The hybrid architecture consists of:
- **Input Processing**: Handles multi-feature time series data with sliding windows
- **Transformer Encoder**: Captures global context and parallel processing within windows
- **LSTM Layers**: Processes enriched features while maintaining temporal order
- **Output Layer**: Generates predictions for specified time horizons

## Data Features
![Sample Data](path/to/table_image.png)

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
![Feature Importance](path/to/heatmap_image.png)

### Key Insights
- Temporal importance varies across different features
- Some features show stronger predictive power at specific time steps
- Complex interactions between economic indicators revealed

## Technical Details

### Model Parameters
- Window Size: 8 time steps
- Feature Dimension: 18
- Transformer Dimension: 32
- LSTM Hidden Dimension: 512
- Number of LSTM Layers: 2

### Implementation
