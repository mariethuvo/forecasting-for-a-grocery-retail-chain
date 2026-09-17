# Demand Forecasting for a Grocery Retail Chain
 
Regression-based forecasting model built for a case study on FreshMart, a 45-store regional grocery chain. The goal was to replace a flat 3-week rolling average with a model that accounts for store-specific and seasonal differences in demand.
 
## Business Problem
 
FreshMart's stores don't behave the same way. A tourist-heavy Orlando location, a college town like Gainesville, and a retirement-community store in Naples all have different demand patterns, but the company was forecasting demand with one global rule. The result: an 8.3% stockout rate and a 4.2% spoilage rate on perishables — the company was simultaneously running out of product and throwing product away, often in the same week.
 
The task: predict weekly units sold per product category per store  (a continuous target, so this is a regression problem) using store characteristics and seasonal signals.
 
## Data
 
Weekly sales data across 45 stores over two years (2022–2023), with store-level attributes (type, square footage, years open, area median income, nearby competitors, college-town/tourist-area flags) and product category.
 
## Approach
 
**EDA first.** Before modeling, I checked whether the "one rule for all stores" assumption actually held up - it didn't. Sales volume varied by roughly 4x between the highest and lowest-volume stores, and demand volatility (measured with coefficient of variation) differed sharply by product category. Tourist-area and college-town stores also showed distinct seasonal curves compared to standard stores, which confirmed the store-specific approach was worth building.
 
**Feature engineering**, based on what the EDA surfaced:
- `is_holiday_week` — flags high-traffic weeks (Thanksgiving, Christmas/New Year, Super Bowl)
- `seasonal_sin` — a sine-wave transform of month, to model cyclic seasonality continuously rather than as 12 separate categories
- `store_volume_segment` — stores bucketed into small/medium/large by square footage
- `college_break` — college-town flag combined with summer/winter break months
- `tourist_peak` — tourist-area flag combined with spring break/summer/holiday months
**Model:** Linear Regression, trained on 2022 data and tested on 2023 — a time-based split, since this is a forecasting problem and a random split would leak future information into training.
 
## Results
 
| Metric | Value |
|---|---|
| Test RMSE | reported in notebook |
| Test MAE | reported in notebook |
| Test R² | reported in notebook |
| Improvement over baseline (category-average) | ~47% reduction in MAPE |
 
The baseline (predicting each category's historical average) was used as the comparison point, since that's effectively what the existing rolling-average method was doing. The model's train and test error were close to each other, which was a good sign against overfitting given the time-based split.
 
**Sanity checks:** No negative predictions, and residuals were checked for pattern (a well-fit model shouldn't have residuals that trend with the prediction). One caveat worth flagging honestly: the residual distribution was right-skewed with a long tail out to about +3,000 units, meaning when the model missed, it tended to under-predict rather than over-predict — worth knowing before using this to set inventory levels, since under-prediction risk maps to stockouts.
 
## What I'd do differently
 
Linear regression was the right starting point for interpretability (the business needed to see which factors moved the forecast, not just get a number), but the skewed residuals suggest a model that can capture non-linear interactions (gradient boosting, for instance) would likely close some of the remaining error, at the cost of losing the direct coefficient interpretation.
 
## Tech Stack
 
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
 
## Files
 
- `VoMarie_FreshMartGrocery.ipynb` 
 
