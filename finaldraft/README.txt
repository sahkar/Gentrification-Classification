FINAL PROJECT - HOUSING PRICE REGRESSION
========================================

WHAT TO RUN:
------------
1. Open: notebooks/housing_price_regression.ipynb
2. Run all cells

FILES:
------
- time_series_preparation.py  - Prepares time series data
- regression_model.py         - Trains and evaluates models
- notebooks/housing_price_regression.ipynb - Main notebook

DATA:
-----
- raw_data/ - Input data (ZORI rent, Yelp businesses)
- data/ - Processed data (generated)
- outputs/ - Results and visualizations (generated)
- models/ - Saved models (generated)

WHAT IT DOES:
-------------
Predicts future average rent per zip code using:
- Historical rent trends (time series features)
- Gentrification indicators from Yelp
- Uses zip code holdout for realistic validation

