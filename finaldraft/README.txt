GENTRIFICATION CLASSIFICATION MODEL
====================================

RESEARCH QUESTION:
Can we use Yelp business ecosystem data to identify gentrified neighborhoods
(future high rent growth areas)?

SETUP:
1. Install dependencies: pip install -r requirements.txt
2. Set YELP_API_KEY in .env file (for fetch_yelp_data.py)

RUN:
1. Main pipeline: python main.py
2. Fetch Yelp data (if needed): python fetch_yelp_data.py

FILES:
- main.py                      - Main classification pipeline
- classification_preparation.py - Data preparation (rent growth labels + Yelp features)
- classification_model.py       - Classification models (Logistic, RF, GB, XGBoost)
- fetch_yelp_data.py            - Fetches Yelp data via API
- presentation.ipynb           - Presentation notebook with visualizations

DATA:
- raw_data/ - Input data (ZORI rent, Yelp businesses)
- data/ - Processed classification data (generated)
- outputs/ - Results and visualizations (generated)
- models/ - Saved models (generated)

MODEL:
Binary classification predicting high rent growth zip codes (top 25%) using:
- Yelp business ecosystem features only (no historical rent data)
- Gentrification indicators (scores, rates, business quality)
- Business diversity and engagement metrics

OUTPUTS:
- classification_results.csv - Model performance metrics
- classification_feature_importance.csv - Feature importance rankings
- classification_model_comparison.png - Model comparison charts
- roc_curves.png - ROC curves for all models
- confusion_matrix.png - Confusion matrix for best model
- classification_feature_importance.png - Top 15 features visualization
- precision_recall_curves.png - Precision-recall curves
- best_classifier.pkl - Saved best model

