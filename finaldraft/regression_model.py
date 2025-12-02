"""
Housing Price Regression Model
=============================

This script implements multiple regression models to predict housing prices
based on property data and gentrification features from Yelp.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HousingPriceRegressor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, data_path="data/processed_data.csv", is_timeseries=False):
        """Load processed data"""
        print("Loading processed data...")
        data = pd.read_csv(data_path)
        
        # Handle date column if present
        dates = None
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date']).values
            print(f"  Found date column: {pd.to_datetime(dates).min()} to {pd.to_datetime(dates).max()}")
        
        # Handle zip code column
        zip_codes = None
        if 'zip' in data.columns:
            zip_codes = data['zip'].values
            print(f"  Found zip codes: {len(pd.Series(zip_codes).unique())} unique")
        
        # Separate features and target
        exclude_cols = ['target', 'zip']
        if is_timeseries:
            exclude_cols.extend(['date', 'year_month'])
        
        exclude_cols.extend([
            'RegionID', 'RegionName', 'RegionType', 'StateName', 
            'State', 'City', 'Metro', 'CountyName', 'city', 'year',
            'rent_price'  # CRITICAL: Exclude current rent price (data leakage!)
        ])
        
        # Also exclude interaction features that use current rent_price
        rent_interaction_cols = [col for col in data.columns if 'rent_gentrification_interaction' in col or 'rent_price' in col]
        exclude_cols.extend(rent_interaction_cols)
        
        if 'target' in data.columns:
            X = data.drop(columns=[col for col in exclude_cols if col in data.columns])
            y = data['target']
        else:
            raise ValueError("Target column 'target' not found in data")
        
        self.feature_names = X.columns.tolist()
        print(f"✓ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Show feature breakdown
        gent_features = [f for f in self.feature_names if 'gentrification' in f.lower() or 'business' in f.lower()]
        rent_features = [f for f in self.feature_names if 'rent' in f.lower() and 'rent_price' not in f.lower()]
        other_features = [f for f in self.feature_names if f not in gent_features + rent_features]
        
        print(f"\n  Feature breakdown:")
        print(f"    Gentrification/Yelp features: {len(gent_features)}")
        print(f"    Rent time series features: {len(rent_features)}")
        print(f"    Other features: {len(other_features)}")
        
        if gent_features:
            print(f"\n  Gentrification features: {gent_features[:5]}{'...' if len(gent_features) > 5 else ''}")
        else:
            print(f"\n  ⚠ WARNING: No gentrification features found! Yelp data may not be included.")
        
        return X, y, dates, zip_codes
    
    def prepare_data(self, X, y, test_size=0.2, dates=None, zip_codes=None, 
                     split_strategy='zip_holdout', n_test_zips=3):
        """
        Prepare train/test split
        
        Args:
            X: Features
            y: Target
            test_size: Proportion for test set (used if split_strategy='time' or 'random')
            dates: Date column for time-based splitting
            zip_codes: Zip code column for zip code holdout
            split_strategy: 'zip_holdout', 'time', or 'random'
            n_test_zips: Number of zip codes to hold out (if zip_holdout)
        """
        if split_strategy == 'zip_holdout' and zip_codes is not None:
            # Zip code holdout: hold out entire zip codes for testing
            zip_series = pd.Series(zip_codes, index=X.index)
            unique_zips = zip_series.unique()
            n_zips = len(unique_zips)
            
            # Calculate samples per zip to ensure balanced split
            samples_per_zip = zip_series.value_counts()
            total_samples = len(X)
            
            # Target: 20-30% of samples in test set
            target_test_samples = int(total_samples * 0.25)
            
            # Sort zip codes by number of samples (descending)
            sorted_zips = samples_per_zip.sort_values(ascending=False).index.tolist()
            
            # Greedily select zip codes for test set to get ~25% of samples
            test_zips = []
            test_sample_count = 0
            
            for zip_code in sorted_zips:
                zip_samples = samples_per_zip[zip_code]
                if test_sample_count + zip_samples <= target_test_samples * 1.5:  # Allow up to 37.5%
                    test_zips.append(zip_code)
                    test_sample_count += zip_samples
                    if test_sample_count >= target_test_samples * 0.8:  # At least 20% of samples
                        break
            
            # Ensure we have at least 5% of zip codes and at least 10% of samples
            min_test_zips = max(5, int(n_zips * 0.05))
            if len(test_zips) < min_test_zips:
                test_zips = sorted_zips[:min_test_zips]
                test_sample_count = sum(samples_per_zip[z] for z in test_zips)
            
            test_zips = np.array(test_zips)
            train_mask = ~zip_series.isin(test_zips).values
            test_mask = zip_series.isin(test_zips).values
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            # Verify split quality
            train_pct = len(X_train) / total_samples * 100
            test_pct = len(X_test) / total_samples * 100
            
            print(f"✓ Zip code holdout split:")
            train_zips = sorted([z for z in unique_zips if z not in test_zips])
            print(f"  Train zip codes: {len(train_zips)} ({train_pct:.1f}% of samples)")
            print(f"  Test zip codes: {len(test_zips)} ({test_pct:.1f}% of samples)")
            print(f"  Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
            
            if test_pct < 15:
                print(f"  ⚠ Warning: Test set is small ({test_pct:.1f}%). Consider using time-based split.")
            
        elif split_strategy == 'time' and dates is not None:
            # Time-based split: use earlier data for training, later for testing
            dates_series = pd.Series(dates, index=X.index)
            split_date = dates_series.quantile(1 - test_size)
            
            train_mask = dates_series < split_date
            test_mask = dates_series >= split_date
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            print(f"✓ Time-based split:")
            print(f"  Train: {dates_series[train_mask].min()} to {dates_series[train_mask].max()}")
            print(f"  Test: {dates_series[test_mask].min()} to {dates_series[test_mask].max()}")
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            print(f"✓ Random split")
        
        # Handle any remaining NaN values
        # Fill with median for numeric columns
        for col in X_train.columns:
            if X_train[col].isna().any():
                median_val = X_train[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
        
        # Scale features
        self.scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)
        
        print(f"  Train set: {X_train_scaled.shape[0]} samples")
        print(f"  Test set: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models"""
        print("\nTraining regression models...")
        
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=0.1, random_state=self.random_state),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        }
        
        for name, model in models_to_train.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
        
        print(f"✓ Trained {len(self.models)} models")
        return self.models
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all models"""
        print("\nEvaluating models...")
        
        results = []
        
        for name, model in self.models.items():
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            cv_std = np.sqrt(cv_scores.std())
            
            results.append({
                'model': name,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_rmse_mean': cv_rmse,
                'cv_rmse_std': cv_std
            })
            
            print(f"\n{name}:")
            print(f"  Test RMSE: {test_rmse:.2f}")
            print(f"  Test MAE:  {test_mae:.2f}")
            print(f"  Test R²:   {test_r2:.4f}")
            print(f"  CV RMSE:   {cv_rmse:.2f} (±{cv_std:.2f})")
        
        results_df = pd.DataFrame(results)
        
        # Select best model (lowest test RMSE)
        best_idx = results_df['test_rmse'].idxmin()
        self.best_model_name = results_df.loc[best_idx, 'model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n✓ Best model: {self.best_model_name}")
        
        return results_df
    
    def analyze_feature_importance(self, X_train, y_train):
        """Analyze feature importance using Random Forest"""
        print("\nAnalyzing feature importance...")
        
        # Use Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def visualize_results(self, X_test, y_test, results_df, importance_df, output_dir="outputs"):
        """Create visualizations"""
        print("\nCreating visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Model comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # RMSE comparison
        axes[0].bar(results_df['model'], results_df['test_rmse'], color='skyblue')
        axes[0].set_title('Test RMSE by Model')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1].bar(results_df['model'], results_df['test_r2'], color='lightcoral')
        axes[1].set_title('Test R² by Model')
        axes[1].set_ylabel('R² Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[2].bar(results_df['model'], results_df['test_mae'], color='lightgreen')
        axes[2].set_title('Test MAE by Model')
        axes[2].set_ylabel('MAE')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved model_comparison.png")
        
        # 2. Feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved feature_importance.png")
        
        # 3. Prediction vs Actual (best model)
        y_pred = self.best_model.predict(X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Housing Price ($)')
        plt.ylabel('Predicted Housing Price ($)')
        plt.title(f'Predicted vs Actual ({self.best_model_name})')
        
        # Add R² to plot
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        print("✓ Saved prediction_scatter.png")
        
        # 4. Residual plot
        residuals = y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Housing Price ($)')
        plt.ylabel('Residuals ($)')
        plt.title(f'Residual Plot ({self.best_model_name})')
        plt.tight_layout()
        plt.savefig(output_dir / 'residual_plot.png', dpi=300, bbox_inches='tight')
        print("✓ Saved residual_plot.png")
        
        plt.close('all')
    
    def save_model(self, output_path="models/best_model.pkl"):
        """Save the best model"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name
        }, output_path)
        
        print(f"✓ Saved best model to {output_path}")
    
    def run_full_pipeline(self, data_path="data/processed_data.csv", output_dir="outputs", 
                         is_timeseries=False, time_based_split=True, use_zip_holdout=True):
        """Run the complete modeling pipeline"""
        print("="*60)
        print("HOUSING PRICE REGRESSION MODEL")
        print("="*60)
        
        # Load data
        X, y, dates, zip_codes = self.load_data(data_path, is_timeseries=is_timeseries)
        
        # Determine split strategy
        if use_zip_holdout and zip_codes is not None:
            unique_zips = len(pd.Series(zip_codes).unique())
            if unique_zips >= 6:
                # Use zip code holdout if we have enough zip codes
                split_strategy = 'zip_holdout'
                print("\nUsing ZIP CODE HOLDOUT (most realistic validation)")
            else:
                print(f"\n⚠ Only {unique_zips} zip codes - using time-based split instead")
                split_strategy = 'time' if (time_based_split and dates is not None) else 'random'
        elif time_based_split and dates is not None:
            split_strategy = 'time'
            print("\nUsing TIME-BASED SPLIT")
        else:
            split_strategy = 'random'
            print("\nUsing RANDOM SPLIT")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            X, y, dates=dates, zip_codes=zip_codes, split_strategy=split_strategy
        )
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        results_df = self.evaluate_models(X_train, X_test, y_train, y_test)
        
        # Feature importance
        importance_df = self.analyze_feature_importance(X_train, y_train)
        
        # Visualizations
        self.visualize_results(X_test, y_test, results_df, importance_df, output_dir)
        
        # Save model
        self.save_model("models/best_model.pkl")
        
        # Save results
        results_df.to_csv(Path(output_dir) / "model_results.csv", index=False)
        importance_df.to_csv(Path(output_dir) / "feature_importance.csv", index=False)
        
        print("\n" + "="*60)
        print("MODELING COMPLETE")
        print("="*60)
        
        return results_df, importance_df

if __name__ == "__main__":
    # Initialize regressor
    regressor = HousingPriceRegressor()
    
    # Run full pipeline
    results, importance = regressor.run_full_pipeline()
    
    print("\nSummary:")
    print(results[['model', 'test_rmse', 'test_r2']].to_string(index=False))
