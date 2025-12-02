"""
Time Series Data Preparation for Housing Price Regression
==========================================================

This script creates an expanded dataset that:
1. Uses full time series ZORI data (monthly rent from 2015-2025)
2. Expands gentrification features from Yelp data
3. Creates a panel dataset (zip code × time)
4. Extracts time series features (trends, seasonality, momentum)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataPreparation:
    def __init__(self, data_dir="raw_data", eda_dir="../eda"):
        self.data_dir = Path(data_dir)
        self.eda_dir = Path(eda_dir)
        self.processed_data = None
        
    def load_zori_timeseries(self):
        """Load and process ZORI time series data"""
        print("Loading ZORI time series data...")
        
        zori_path = self.data_dir / "Zip_zori_uc_sfrcondomfr_sm_month.csv"
        zori_df = pd.read_csv(zori_path)
        
        # Extract zip codes from RegionName
        zori_df['zip'] = zori_df['RegionName'].astype(str).str.zfill(5)
        
        # Filter to Bay Area metros
        bay_area_metros = [
            "San Francisco-Oakland-Berkeley, CA",
            "San Jose-Sunnyvale-Santa Clara, CA"
        ]
        
        print(f"  Total rows in ZORI data: {len(zori_df):,}")
        print(f"  Filtering by metro areas: {bay_area_metros}")
        
        zori_df = zori_df[zori_df['Metro'].isin(bay_area_metros)]
        
        if len(zori_df) == 0:
            raise ValueError(f"No data found for Bay Area metros: {bay_area_metros}")
        
        unique_zips = zori_df['zip'].unique()
        print(f"  Found {len(unique_zips)} unique zip codes in Bay Area metros")
        print(f"  Sample zip codes: {sorted(unique_zips)[:10]}")
        
        # Get date columns
        date_cols = [col for col in zori_df.columns if '-' in col and col.count('-') == 2]
        
        # Melt to long format: zip × date
        id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 
                   'State', 'City', 'Metro', 'CountyName', 'zip']
        zori_long = pd.melt(zori_df, id_vars=id_vars, value_vars=date_cols,
                           var_name='date', value_name='rent_price')
        
        # Convert date to datetime
        zori_long['date'] = pd.to_datetime(zori_long['date'])
        
        # Remove rows with missing rent prices
        zori_long = zori_long.dropna(subset=['rent_price'])
        
        # Sort by zip and date
        zori_long = zori_long.sort_values(['zip', 'date']).reset_index(drop=True)
        
        print(f"✓ Loaded ZORI time series: {len(zori_long)} observations")
        print(f"  Zip codes: {zori_long['zip'].nunique()}")
        print(f"  Date range: {zori_long['date'].min()} to {zori_long['date'].max()}")
        
        return zori_long
    
    def create_timeseries_features(self, df):
        """Create time series features from rent data"""
        print("Creating time series features...")
        
        df = df.copy()
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Calculate features per zip code
        features_list = []
        
        for zip_code in df['zip'].unique():
            zip_data = df[df['zip'] == zip_code].sort_values('date').copy()
            
            # Rolling statistics
            zip_data['rent_ma_3m'] = zip_data['rent_price'].rolling(window=3, min_periods=1).mean()
            zip_data['rent_ma_6m'] = zip_data['rent_price'].rolling(window=6, min_periods=1).mean()
            zip_data['rent_ma_12m'] = zip_data['rent_price'].rolling(window=12, min_periods=1).mean()
            
            # Rolling standard deviation (volatility)
            zip_data['rent_std_3m'] = zip_data['rent_price'].rolling(window=3, min_periods=1).std()
            zip_data['rent_std_12m'] = zip_data['rent_price'].rolling(window=12, min_periods=1).std()
            
            # Lagged values
            zip_data['rent_lag_1m'] = zip_data['rent_price'].shift(1)
            zip_data['rent_lag_3m'] = zip_data['rent_price'].shift(3)
            zip_data['rent_lag_6m'] = zip_data['rent_price'].shift(6)
            zip_data['rent_lag_12m'] = zip_data['rent_price'].shift(12)
            
            # Percentage changes
            zip_data['rent_pct_change_1m'] = zip_data['rent_price'].pct_change(1)
            zip_data['rent_pct_change_3m'] = zip_data['rent_price'].pct_change(3)
            zip_data['rent_pct_change_6m'] = zip_data['rent_price'].pct_change(6)
            zip_data['rent_pct_change_12m'] = zip_data['rent_price'].pct_change(12)
            
            # Momentum indicators
            zip_data['rent_momentum_3m'] = zip_data['rent_price'] - zip_data['rent_lag_3m']
            zip_data['rent_momentum_6m'] = zip_data['rent_price'] - zip_data['rent_lag_6m']
            zip_data['rent_momentum_12m'] = zip_data['rent_price'] - zip_data['rent_lag_12m']
            
            # Acceleration (change in momentum) - KEEP (shows trend direction)
            zip_data['rent_acceleration'] = zip_data['rent_pct_change_1m'] - zip_data['rent_pct_change_1m'].shift(1)
            
            # Year-over-year change - KEEP (shows long-term trend)
            zip_data['rent_yoy'] = zip_data['rent_price'].pct_change(12)
            
            # Long-term trend indicator (slope over 12 months) - KEEP
            if len(zip_data) >= 12:
                zip_data['rent_trend_slope'] = zip_data['rent_price'].rolling(12).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 12 else np.nan, raw=False
                )
            
            # Relative position in year (seasonality)
            zip_data['month_sin'] = np.sin(2 * np.pi * zip_data['month'] / 12)
            zip_data['month_cos'] = np.cos(2 * np.pi * zip_data['month'] / 12)
            
            # Trend features (linear trend over time)
            zip_data['time_index'] = range(len(zip_data))
            
            # Price level features (relative to historical)
            if len(zip_data) >= 12:
                zip_data['rent_vs_12m_avg'] = zip_data['rent_price'] / zip_data['rent_ma_12m']
                zip_data['rent_vs_12m_min'] = zip_data['rent_price'] / zip_data['rent_price'].rolling(12).min()
                zip_data['rent_vs_12m_max'] = zip_data['rent_price'] / zip_data['rent_price'].rolling(12).max()
            
            features_list.append(zip_data)
        
        result = pd.concat(features_list, ignore_index=True)
        
        print(f"✓ Created time series features")
        print(f"  Total features added: {len([c for c in result.columns if c not in df.columns])}")
        
        return result
    
    def expand_gentrification_features(self, yelp_df):
        """Create expanded gentrification features from Yelp data"""
        print("Expanding gentrification features...")
        
        yelp_df = yelp_df.copy()
        
        # Convert zip to string
        yelp_df['zip'] = yelp_df['zip'].astype(str).str.zfill(5)
        
        # Define gentrification indicators
        HIP_KEYWORDS = [
            'artisanal', 'craft', 'fusion', 'minimalist', 'matcha', 
            'third wave', 'boutique', 'curated', 'specialty', 'gourmet',
            'farm-to-table', 'organic', 'sustainable', 'locally sourced',
            'small batch', 'handcrafted', 'bespoke', 'elevated'
        ]
        
        TRENDY_CATEGORIES = [
            'New American', 'Vegan', 'Coffee & Tea', 'Cocktail Bars',
            'Matcha', 'Bubble Tea', 'Poke', 'Pilates', 'Yoga',
            'Wine Bars', 'Tapas', 'Izakaya', 'Gastropubs',
            'Artisanal', 'Specialty', 'Third Wave Coffee'
        ]
        
        HIGH_END_PRICE = ['$$$', '$$$$']
        
        # Enhanced gentrification scoring
        def calculate_gentrification_score(row):
            score = 0
            indicators = []
            
            # Price tier (weighted)
            if pd.notna(row.get('price')):
                if row['price'] == '$$$$':
                    score += 4
                    indicators.append('very_high_price')
                elif row['price'] == '$$$':
                    score += 3
                    indicators.append('high_price')
                elif row['price'] == '$$':
                    score += 1
                    indicators.append('moderate_price')
            
            # Categories
            categories_str = str(row.get('categories', '')).lower()
            for cat in TRENDY_CATEGORIES:
                if cat.lower() in categories_str:
                    score += 2
                    indicators.append(f'category_{cat.lower().replace(" ", "_")}')
                    break
            
            # Rating and reviews (high engagement = gentrified)
            if pd.notna(row.get('rating')):
                if row['rating'] >= 4.5:
                    score += 2
                    indicators.append('high_rating')
                elif row['rating'] >= 4.0:
                    score += 1
                    indicators.append('good_rating')
            
            if pd.notna(row.get('review_count')):
                if row['review_count'] > 500:
                    score += 2
                    indicators.append('high_reviews')
                elif row['review_count'] > 200:
                    score += 1
                    indicators.append('moderate_reviews')
            
            # Keywords in name/description
            name_str = str(row.get('name', '')).lower()
            for keyword in HIP_KEYWORDS:
                if keyword in name_str:
                    score += 1
                    indicators.append(f'keyword_{keyword}')
                    break
            
            return pd.Series({
                'gentrification_score': score,
                'is_gentrified': score >= 5,  # Threshold for gentrification
                'indicator_count': len(indicators),
                'has_high_price': row.get('price') in HIGH_END_PRICE if pd.notna(row.get('price')) else False,
                'has_trendy_category': any(cat.lower() in categories_str for cat in TRENDY_CATEGORIES),
                'high_engagement': (row.get('rating', 0) >= 4.0) and (row.get('review_count', 0) > 200)
            })
        
        # Apply scoring
        gentrification_features = yelp_df.apply(calculate_gentrification_score, axis=1)
        yelp_df = pd.concat([yelp_df, gentrification_features], axis=1)
        
        # Aggregate by zip code
        zip_gentrification = yelp_df.groupby('zip').agg({
            'gentrification_score': ['mean', 'std', 'sum', 'count'],
            'is_gentrified': ['sum', 'mean'],
            'indicator_count': 'mean',
            'has_high_price': 'sum',
            'has_trendy_category': 'sum',
            'high_engagement': 'sum',
            'rating': ['mean', 'std', 'median'],
            'review_count': ['mean', 'sum', 'median'],
            'price': lambda x: (x.isin(HIGH_END_PRICE).sum() / len(x) if len(x) > 0 else 0)
        }).reset_index()
        
        # Flatten column names
        zip_gentrification.columns = [
            'zip',
            'avg_gentrification_score', 'std_gentrification_score', 'total_gentrification_score', 'business_count',
            'gentrified_business_count', 'gentrification_rate',
            'avg_indicator_count',
            'high_price_business_count',
            'trendy_category_business_count',
            'high_engagement_business_count',
            'avg_rating', 'std_rating', 'median_rating',
            'avg_review_count', 'total_reviews', 'median_review_count',
            'high_price_business_pct'
        ]
        
        # Create composite features that capture gentrification impact
        zip_gentrification['gentrification_intensity'] = (
            zip_gentrification['gentrification_rate'] * 
            zip_gentrification['avg_gentrification_score']
        )
        
        zip_gentrification['business_quality_index'] = (
            zip_gentrification['avg_rating'] * 
            (zip_gentrification['avg_review_count'] / 100)  # Normalize review count
        )
        
        zip_gentrification['premium_business_density'] = (
            zip_gentrification['high_price_business_count'] / 
            zip_gentrification['business_count']
        )
        
        # Gentrification maturity (how established is the gentrification)
        zip_gentrification['gentrification_maturity'] = (
            zip_gentrification['gentrified_business_count'] * 
            zip_gentrification['avg_gentrification_score'] / 
            (zip_gentrification['business_count'] + 1)  # Avoid division by zero
        )
        
        # Business diversity (more diverse = more gentrified)
        zip_gentrification['business_diversity'] = (
            zip_gentrification['trendy_category_business_count'] / 
            (zip_gentrification['business_count'] + 1)
        )
        
        # High-end business concentration
        zip_gentrification['high_end_concentration'] = (
            zip_gentrification['high_price_business_count'] / 
            (zip_gentrification['business_count'] + 1)
        )
        
        print(f"✓ Expanded gentrification features")
        print(f"  Zip codes: {len(zip_gentrification)}")
        print(f"  Features: {len(zip_gentrification.columns) - 1}")  # -1 for zip
        
        return zip_gentrification, yelp_df
    
    def load_yelp_data(self):
        """Load Yelp business data - tries expanded dataset first, then falls back to original"""
        print("Loading Yelp business data...")
        
        # Try expanded dataset first (all zip codes)
        yelp_path = self.data_dir / "bay_area_yelp_businesses_all_zips.csv"
        
        if not yelp_path.exists():
            # Try original dataset
            yelp_path = self.data_dir / "bay_area_yelp_businesses.csv"
        
        if not yelp_path.exists():
            # Try EDA directory
            yelp_path = self.eda_dir / "bay_area_yelp_businesses_with_labels.csv"
        
        if not yelp_path.exists():
            raise FileNotFoundError(
                f"Yelp data not found. Tried:\n"
                f"  - {self.data_dir / 'bay_area_yelp_businesses_all_zips.csv'}\n"
                f"  - {self.data_dir / 'bay_area_yelp_businesses.csv'}\n"
                f"  - {self.eda_dir / 'bay_area_yelp_businesses_with_labels.csv'}\n"
                f"Run fetch_yelp_data.py to create expanded dataset."
            )
        
        yelp_df = pd.read_csv(yelp_path)
        
        # Ensure zip column exists
        if 'zip' not in yelp_df.columns:
            # Try to find it
            for col in yelp_df.columns:
                if 'zip' in col.lower():
                    yelp_df = yelp_df.rename(columns={col: 'zip'})
                    break
        
        print(f"✓ Loaded Yelp data: {len(yelp_df)} businesses from {yelp_path.name}")
        print(f"  Unique zip codes in Yelp data: {yelp_df['zip'].nunique()}")
        
        return yelp_df
    
    def merge_timeseries_data(self, zori_ts, gentrification_features):
        """Merge time series rent data with gentrification features"""
        print("Merging time series data with gentrification features...")
        
        # Gentrification features are static (per zip), so we merge them to all time points
        merged = zori_ts.merge(
            gentrification_features,
            on='zip',
            how='left'
        )
        
        # Fill NaN values in gentrification features with 0 (no gentrification data)
        gent_cols = [col for col in merged.columns if col.startswith(('gentrification', 'business', 'avg_', 'std_', 'median_', 'total_', 'high_', 'premium_'))]
        for col in gent_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        # Create interaction features between gentrification and time/trends
        # Focus on how gentrification interacts with temporal patterns
        
        # Gentrification × time (gentrification effect may change over time)
        if 'year' in merged.columns:
            merged['gentrification_time_interaction'] = (
                merged.get('gentrification_rate', 0).fillna(0) * merged['year']
            )
        
        # Gentrification × volatility (gentrified areas may have different volatility)
        if 'rent_std_3m' in merged.columns:
            merged['gentrification_volatility_interaction'] = (
                merged.get('gentrification_intensity', 0).fillna(0) * merged['rent_std_3m'].fillna(0)
            )
        
        # Gentrification × trend (gentrified areas may have steeper trends)
        if 'rent_yoy' in merged.columns:
            merged['gentrification_trend_interaction'] = (
                merged.get('gentrification_intensity', 0).fillna(0) * merged['rent_yoy'].fillna(0)
            )
        
        # Business quality × time (quality businesses may signal future rent increases)
        if 'business_quality_index' in merged.columns and 'year' in merged.columns:
            merged['business_quality_time_interaction'] = (
                merged['business_quality_index'].fillna(0) * merged['year']
            )
        
        # Time since gentrification (if we had temporal gentrification data)
        # For now, use static gentrification features
        
        print(f"✓ Merged dataset: {len(merged)} observations")
        print(f"  Zip codes: {merged['zip'].nunique()}")
        print(f"  Time points: {merged['date'].nunique()}")
        
        return merged
    
    def prepare_for_modeling(self, target_col='rent_price', prediction_horizon=1):
        """
        Prepare data for modeling
        
        Args:
            target_col: Column to predict
            prediction_horizon: How many months ahead to predict (1 = next month)
        """
        print(f"\nPreparing data for modeling (predicting {prediction_horizon} month(s) ahead)...")
        
        # Load all data
        zori_ts = self.load_zori_timeseries()
        zori_ts = self.create_timeseries_features(zori_ts)
        
        # Get zip codes from ZORI data
        zori_zips = set(zori_ts['zip'].unique())
        print(f"  ZORI zip codes: {len(zori_zips)}")
        
        # Load Yelp data and filter to match ZORI zip codes
        yelp_df = self.load_yelp_data()
        
        # Convert Yelp zip to string and filter to ZORI zip codes
        yelp_df['zip'] = yelp_df['zip'].astype(str).str.zfill(5)
        yelp_df = yelp_df[yelp_df['zip'].isin(zori_zips)].copy()
        
        print(f"  Yelp businesses in ZORI zip codes: {len(yelp_df)}")
        print(f"  Yelp zip codes matching ZORI: {yelp_df['zip'].nunique()}")
        
        gentrification_features, yelp_expanded = self.expand_gentrification_features(yelp_df)
        
        # Merge
        merged = self.merge_timeseries_data(zori_ts, gentrification_features)
        
        # Create target variable (future rent)
        merged = merged.sort_values(['zip', 'date'])
        merged['target'] = merged.groupby('zip')[target_col].shift(-prediction_horizon)
        
        # Remove rows where target is NaN (last N months for each zip)
        merged = merged.dropna(subset=['target'])
        
        # Separate features and target
        # EXCLUDE highly predictive rent features to focus on gentrification impact
        exclude_cols = [
            'target', 'zip', 'date', 'RegionID', 'RegionName', 'RegionType',
            'StateName', 'State', 'City', 'Metro', 'CountyName', 'year_month',
            'rent_price',  # Exclude current rent (data leakage)
            # Exclude lagged rent values (too predictive, defeats purpose)
            'rent_lag_1m', 'rent_lag_3m', 'rent_lag_6m', 'rent_lag_12m',
            # Exclude moving averages (too predictive)
            'rent_ma_3m', 'rent_ma_6m', 'rent_ma_12m',
            # Exclude direct rent derivatives
            'rent_pct_change_1m', 'rent_pct_change_3m', 'rent_pct_change_6m', 'rent_pct_change_12m',
            'rent_momentum_3m', 'rent_momentum_6m', 'rent_momentum_12m',
            'rent_vs_12m_avg', 'rent_vs_12m_min', 'rent_vs_12m_max'
        ]
        
        # KEEP: gentrification features, business features, time features, volatility, trends
        # These help us understand HOW gentrification affects rent
        
        feature_cols = [col for col in merged.columns if col not in exclude_cols]
        
        # Remove features with too many missing values
        missing_pct = merged[feature_cols].isna().sum() / len(merged)
        feature_cols = [col for col in feature_cols if missing_pct[col] < 0.5]
        
        X = merged[feature_cols]
        y = merged['target']
        
        # Fill remaining NaN values with median (for numeric) or 0
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
            else:
                X[col] = X[col].fillna(0)
        
        # Drop rows where target is NaN (shouldn't happen, but just in case)
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"  After cleaning: {len(X)} samples, {len(feature_cols)} features")
        print(f"  Missing values: {X.isna().sum().sum()} total")
        
        # Store metadata
        self.feature_names = feature_cols
        self.target_name = 'target'
        self.zip_codes = merged['zip'].values
        self.dates = merged['date'].values
        
        print(f"✓ Prepared dataset:")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {len(X)}")
        print(f"  Zip codes: {merged['zip'].nunique()}")
        print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
        
        return X, y, merged
    
    def save_processed_data(self, output_path="data/processed_timeseries_data.csv", prediction_horizon=1):
        """Save processed time series data"""
        X, y, merged = self.prepare_for_modeling(prediction_horizon=prediction_horizon)
        
        # Add target back for saving
        merged['target'] = y
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        
        print(f"✓ Saved processed time series data to {output_path}")
        return merged

if __name__ == "__main__":
    # Initialize time series data preparation
    prep = TimeSeriesDataPreparation()
    
    # Prepare and save data (predicting 1 month ahead)
    processed_data = prep.save_processed_data("data/processed_timeseries_data.csv", prediction_horizon=1)
    
    print("\n" + "="*60)
    print("TIME SERIES DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nDataset shape: {processed_data.shape}")
    print(f"\nFeatures available:")
    for i, col in enumerate(prep.feature_names, 1):
        print(f"  {i:2d}. {col}")

