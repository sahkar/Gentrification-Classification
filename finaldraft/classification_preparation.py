"""
Classification Data Preparation
Prepares data for binary classification of high rent growth zip codes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ClassificationDataPreparation:
    def __init__(self, data_dir="raw_data"):
        self.data_dir = Path(data_dir)
        self.processed_data = None
        self.feature_names = None
        
    def load_zori_rent_data(self):
        """Load ZORI rent data and calculate rent growth"""
        print("Loading ZORI rent data...")
        
        zori_path = self.data_dir / "Zip_zori_uc_sfrcondomfr_sm_month.csv"
        zori_df = pd.read_csv(zori_path)
        
        # Extract zip codes
        zori_df['zip'] = zori_df['RegionName'].astype(str).str.zfill(5)
        
        # Filter to Bay Area metros
        bay_area_metros = [
            "San Francisco-Oakland-Berkeley, CA",
            "San Jose-Sunnyvale-Santa Clara, CA"
        ]
        
        zori_df = zori_df[zori_df['Metro'].isin(bay_area_metros)]
        
        print(f"  Found {len(zori_df)} zip codes in Bay Area metros")
        
        date_cols = [col for col in zori_df.columns if '-' in col and col.count('-') == 2]
        valid_dates = []
        for col in date_cols:
            try:
                pd.to_datetime(col, errors='raise')
                valid_dates.append(col)
            except:
                pass
        date_cols = sorted(valid_dates)
        
        if len(date_cols) < 12:
            raise ValueError("Not enough date columns for rent growth calculation")
        
        start_dates = [col for col in date_cols if '2021' in col or '2020' in col]
        end_dates = [col for col in date_cols if '2024' in col or '2025' in col]
        
        if not start_dates:
            start_dates = date_cols[:12]
        if not end_dates:
            end_dates = date_cols[-12:]
        
        start_date = start_dates[0] if start_dates else date_cols[0]
        end_date = end_dates[-1] if end_dates else date_cols[-1]
        
        print(f"  Calculating rent growth from {start_date} to {end_date}")
        
        start_cols = [col for col in date_cols if col <= start_date][-12:]
        end_cols = [col for col in date_cols if col >= end_date][:12]
        
        if not start_cols or not end_cols:
            start_cols = [start_date]
            end_cols = [end_date]
        
        zori_df['rent_start'] = zori_df[start_cols].mean(axis=1)
        zori_df['rent_end'] = zori_df[end_cols].mean(axis=1)
        zori_df['rent_growth'] = (zori_df['rent_end'] - zori_df['rent_start']) / zori_df['rent_start']
        
        zori_df = zori_df.dropna(subset=['rent_growth'])
        zori_df = zori_df[zori_df['rent_growth'].between(-1, 2)]
        
        print(f"  Calculated rent growth for {len(zori_df)} zip codes")
        print(f"  Growth range: {zori_df['rent_growth'].min():.2%} to {zori_df['rent_growth'].max():.2%}")
        print(f"  Mean growth: {zori_df['rent_growth'].mean():.2%}")
        
        return zori_df[['zip', 'rent_growth', 'rent_start', 'rent_end', 'SizeRank', 
                       'Metro', 'City', 'CountyName']].copy()
    
    def create_binary_labels(self, zori_df, top_percentile=0.25):
        """Create binary labels: 1 = high growth (top 25%), 0 = normal growth"""
        print(f"\nCreating binary labels (top {top_percentile*100:.0f}% = high growth)...")
        
        threshold = zori_df['rent_growth'].quantile(1 - top_percentile)
        zori_df['high_growth'] = (zori_df['rent_growth'] >= threshold).astype(int)
        
        high_growth_count = zori_df['high_growth'].sum()
        total_count = len(zori_df)
        
        print(f"  Threshold: {threshold:.2%}")
        print(f"  High growth: {high_growth_count} ({high_growth_count/total_count*100:.1f}%)")
        print(f"  Normal growth: {total_count - high_growth_count} ({(total_count-high_growth_count)/total_count*100:.1f}%)")
        
        return zori_df, threshold
    
    def load_yelp_data(self):
        """Load Yelp business data"""
        print("\nLoading Yelp business data...")
        
        # Try expanded dataset first
        yelp_path = self.data_dir / "bay_area_yelp_businesses_all_zips.csv"
        
        if not yelp_path.exists():
            yelp_path = self.data_dir / "bay_area_yelp_businesses.csv"
        
        if not yelp_path.exists():
            raise FileNotFoundError(
                f"Yelp data not found. Tried:\n"
                f"  - {self.data_dir / 'bay_area_yelp_businesses_all_zips.csv'}\n"
                f"  - {self.data_dir / 'bay_area_yelp_businesses.csv'}\n"
                f"Run fetch_yelp_data.py to create expanded dataset."
            )
        
        yelp_df = pd.read_csv(yelp_path)
        
        # Ensure zip column exists
        if 'zip' not in yelp_df.columns:
            for col in yelp_df.columns:
                if 'zip' in col.lower():
                    yelp_df = yelp_df.rename(columns={col: 'zip'})
                    break
        
        yelp_df['zip'] = yelp_df['zip'].astype(str).str.zfill(5)
        
        print(f"  Loaded {len(yelp_df)} businesses from {yelp_path.name}")
        print(f"  Unique zip codes: {yelp_df['zip'].nunique()}")
        
        return yelp_df
    
    def expand_gentrification_features(self, yelp_df):
        """Create expanded gentrification features from Yelp data"""
        print("Expanding gentrification features...")
        
        yelp_df = yelp_df.copy()
        
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
        
        def calculate_gentrification_score(row):
            score = 0
            indicators = []
            
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
            
            categories_str = str(row.get('categories', '')).lower()
            for cat in TRENDY_CATEGORIES:
                if cat.lower() in categories_str:
                    score += 2
                    indicators.append(f'category_{cat.lower().replace(" ", "_")}')
                    break
            
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
            
            name_str = str(row.get('name', '')).lower()
            for keyword in HIP_KEYWORDS:
                if keyword in name_str:
                    score += 1
                    indicators.append(f'keyword_{keyword}')
                    break
            
            return pd.Series({
                'gentrification_score': score,
                'is_gentrified': score >= 5,
                'indicator_count': len(indicators),
                'has_high_price': row.get('price') in HIGH_END_PRICE if pd.notna(row.get('price')) else False,
                'has_trendy_category': any(cat.lower() in categories_str for cat in TRENDY_CATEGORIES),
                'high_engagement': (row.get('rating', 0) >= 4.0) and (row.get('review_count', 0) > 200)
            })
        
        gentrification_features = yelp_df.apply(calculate_gentrification_score, axis=1)
        yelp_df = pd.concat([yelp_df, gentrification_features], axis=1)
        
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
        
        zip_gentrification['gentrification_intensity'] = (
            zip_gentrification['gentrification_rate'] * 
            zip_gentrification['avg_gentrification_score']
        )
        
        zip_gentrification['business_quality_index'] = (
            zip_gentrification['avg_rating'] * 
            (zip_gentrification['avg_review_count'] / 100)
        )
        
        zip_gentrification['premium_business_density'] = (
            zip_gentrification['high_price_business_count'] / 
            (zip_gentrification['business_count'] + 1)
        )
        
        zip_gentrification['gentrification_maturity'] = (
            zip_gentrification['gentrified_business_count'] * 
            zip_gentrification['avg_gentrification_score'] / 
            (zip_gentrification['business_count'] + 1)
        )
        
        zip_gentrification['business_diversity'] = (
            zip_gentrification['trendy_category_business_count'] / 
            (zip_gentrification['business_count'] + 1)
        )
        
        zip_gentrification['high_end_concentration'] = (
            zip_gentrification['high_price_business_count'] / 
            (zip_gentrification['business_count'] + 1)
        )
        
        print(f"✓ Expanded gentrification features")
        print(f"  Zip codes: {len(zip_gentrification)}")
        print(f"  Features: {len(zip_gentrification.columns) - 1}")
        
        return zip_gentrification
    
    def prepare_for_classification(self, top_percentile=0.25):
        """Prepare final dataset for classification"""
        print("\n" + "="*70)
        print("PREPARING CLASSIFICATION DATASET")
        print("="*70)
        
        zori_df = self.load_zori_rent_data()
        zori_df, threshold = self.create_binary_labels(zori_df, top_percentile)
        yelp_df = self.load_yelp_data()
        gentrification_features = self.expand_gentrification_features(yelp_df)
        
        merged = zori_df.merge(gentrification_features, on='zip', how='left')
        
        gent_cols = [col for col in merged.columns if col.startswith(('gentrification', 'business', 'avg_', 'std_', 
                                                                      'median_', 'total_', 'high_', 'premium_'))]
        for col in gent_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        exclude_cols = [
            'high_growth',
            'rent_growth', 'rent_start', 'rent_end',
            'zip',
            'SizeRank',
            'City', 'CountyName', 'Metro'
        ]
        
        feature_cols = [col for col in merged.columns if col not in exclude_cols]
        numeric_cols = merged[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        X = merged[feature_cols]
        y = merged['high_growth']
        
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
            else:
                X[col] = X[col].fillna(0)
        
        self.feature_names = feature_cols
        self.zip_codes = merged['zip'].values
        self.rent_growths = merged['rent_growth'].values
        
        print(f"\n✓ Dataset prepared: {len(X)} samples, {len(feature_cols)} features")
        print(f"  High growth: {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"  Normal growth: {(~y.astype(bool)).sum()} ({(~y.astype(bool)).mean()*100:.1f}%)")
        
        gent_features = [f for f in feature_cols if 'gentrification' in f.lower() or 'business' in f.lower()]
        other_features = [f for f in feature_cols if f not in gent_features]
        print(f"  Gentrification/Business features: {len(gent_features)}")
        print(f"  Other features: {len(other_features)}")
        
        return X, y, merged
    
    def save_classification_data(self, output_path="data/classification_data.csv", top_percentile=0.25):
        """Save prepared classification data"""
        X, y, merged = self.prepare_for_classification(top_percentile=top_percentile)
        
        merged['target'] = y
        merged['rent_growth_pct'] = merged['rent_growth'] * 100
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        
        print(f"\n✓ Saved to: {output_path}")
        return merged

if __name__ == "__main__":
    prep = ClassificationDataPreparation(data_dir="raw_data")
    data = prep.save_classification_data("data/classification_data.csv", top_percentile=0.25)

