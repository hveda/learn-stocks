# Stock Price Forecasting Model Comparison

## Overview

This report compares the performance of ARIMA, Prophet, and Ensemble forecasting models on Indonesian stock price data.

## Metrics Explanation

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of the average squared differences
- **MAPE (Mean Absolute Percentage Error)**: Average percentage difference between predicted and actual values
- **R² (R-squared)**: Statistical measure of how close the data are to the fitted regression line
- **Coverage**: Percentage of actual values falling within the prediction intervals

## Results

### BBCA.JK

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|----|
| Prophet | 467.2273 | 550.3784 | 5.45% | -0.5006 |
| Ensemble | 272.3245 | 304.7926 | 2.87% | -5.3782 |

## Conclusion

Based on the metrics above, we can observe the following patterns:

- For BBCA.JK, the Ensemble model performs best in terms of MAPE.

## Recommendations

1. For trading decisions, consider using the model with lower error metrics.
2. Prophet generally provides better uncertainty estimates through prediction intervals.
3. ARIMA may be more suitable for stocks with well-defined trends and seasonality.
4. Consider ensemble approaches combining models for potentially improved results.
