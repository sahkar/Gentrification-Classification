"""
Gentrification Classification Model
Binary classification to predict high rent growth zip codes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

class GentrificationClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, data_path="data/classification_data.csv"):
        """Load classification data"""
        print("Loading classification data...")
        data = pd.read_csv(data_path)
        
        # Handle zip code column
        zip_codes = None
        if 'zip' in data.columns:
            zip_codes = data['zip'].values
            print(f"  Found zip codes: {len(pd.Series(zip_codes).unique())} unique")
        
        exclude_cols = ['target', 'high_growth', 'zip', 'rent_growth', 'rent_start', 'rent_end',
                       'rent_growth_pct', 'City', 'CountyName', 'Metro', 'SizeRank']
        
        if 'target' in data.columns:
            y = data['target']
        elif 'high_growth' in data.columns:
            y = data['high_growth']
        else:
            raise ValueError("Target column ('target' or 'high_growth') not found in data")
        
        X = data.drop(columns=[col for col in exclude_cols if col in data.columns])
        
        # Ensure only numeric columns are included
        X = X.select_dtypes(include=[np.number])
        
        self.feature_names = X.columns.tolist()
        print(f"✓ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Show feature breakdown
        gent_features = [f for f in self.feature_names if 'gentrification' in f.lower() or 'business' in f.lower()]
        other_features = [f for f in self.feature_names if f not in gent_features]
        
        print(f"  Gentrification/Business features: {len(gent_features)}")
        print(f"  Other features: {len(other_features)}")
        
        return X, y, zip_codes
    
    def prepare_data(self, X, y, zip_codes=None, test_size=0.2, split_strategy='random'):
        """Prepare train/test split"""
        if split_strategy == 'zip_holdout' and zip_codes is not None:
            zip_series = pd.Series(zip_codes, index=X.index)
            unique_zips = zip_series.unique()
            samples_per_zip = zip_series.value_counts()
            total_samples = len(X)
            target_test_samples = int(total_samples * 0.25)
            sorted_zips = samples_per_zip.sort_values(ascending=False).index.tolist()
            
            test_zips = []
            test_sample_count = 0
            for zip_code in sorted_zips:
                zip_samples = samples_per_zip[zip_code]
                if test_sample_count + zip_samples <= target_test_samples * 1.5:
                    test_zips.append(zip_code)
                    test_sample_count += zip_samples
                    if test_sample_count >= target_test_samples * 0.8:
                        break
            
            min_test_zips = max(5, int(len(unique_zips) * 0.05))
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
            
            print(f"✓ Zip code holdout: {len(X_train)} train, {len(X_test)} test")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            print(f"✓ Random split: {len(X_train)} train, {len(X_test)} test")
        
        for col in X_train.columns:
            if X_train[col].isna().any():
                median_val = X_train[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
        
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple classification models"""
        print("\nTraining models...")
        
        self.models['Logistic Regression'] = LogisticRegression(
            random_state=self.random_state, max_iter=1000, class_weight='balanced'
        )
        self.models['Logistic Regression'].fit(X_train, y_train)
        
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train, y_train)
        
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100, random_state=self.random_state
        )
        self.models['Gradient Boosting'].fit(X_train, y_train)
        
        if HAS_XGBOOST:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, random_state=self.random_state, eval_metric='logloss'
            )
            self.models['XGBoost'].fit(X_train, y_train)
        
        print(f"✓ Trained {len(self.models)} models")
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all classification models"""
        print("\nEvaluating models...")
        
        results = []
        
        for name, model in self.models.items():
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            cv_roc_auc = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results.append({
                'model': name,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'cv_roc_auc_mean': cv_roc_auc,
                'cv_roc_auc_std': cv_std
            })
            
            print(f"{name}:")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  Test Precision: {test_precision:.4f}")
            print(f"  Test Recall: {test_recall:.4f}")
            print(f"  Test F1: {test_f1:.4f}")
            print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
            print(f"  CV ROC-AUC: {cv_roc_auc:.4f} (±{cv_std:.4f})")
        
        results_df = pd.DataFrame(results)
        
        best_idx = results_df['test_roc_auc'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n✓ Best model: {self.best_model_name} (ROC-AUC: {results_df.loc[best_idx, 'test_roc_auc']:.4f})")
        
        return results_df
    
    def analyze_feature_importance(self, X_train, y_train):
        """Analyze feature importance from Random Forest"""
        print("\nAnalyzing feature importance...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced')
        rf.fit(X_train, y_train)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def create_visualizations(self, X_test, y_test, results_df, importance_df, output_dir="outputs"):
        """Create classification visualizations"""
        print("\nCreating visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].bar(results_df['model'], results_df['test_accuracy'], color='steelblue')
        axes[0, 0].set_title('Test Accuracy by Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim([0, 1])
        
        axes[0, 1].bar(results_df['model'], results_df['test_roc_auc'], color='lightgreen')
        axes[0, 1].set_title('Test ROC-AUC by Model')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim([0, 1])
        
        axes[1, 0].bar(results_df['model'], results_df['test_f1'], color='coral')
        axes[1, 0].set_title('Test F1-Score by Model')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim([0, 1])
        
        axes[1, 1].bar(results_df['model'], results_df['test_precision'], label='Precision', alpha=0.7)
        axes[1, 1].bar(results_df['model'], results_df['test_recall'], label='Recall', alpha=0.7)
        axes[1, 1].set_title('Precision vs Recall by Model')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'classification_model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved classification_model_comparison.png")
        
        plt.figure(figsize=(10, 8))
        for name, model in self.models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved roc_curves.png")
        
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal Growth', 'High Growth'],
                   yticklabels=['Normal Growth', 'High Growth'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved confusion_matrix.png")
        
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'classification_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved classification_feature_importance.png")
        
        plt.figure(figsize=(10, 8))
        for name, model in self.models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved precision_recall_curves.png")
        
        plt.close('all')
    
    def save_model(self, output_path="models/best_classifier.pkl"):
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
    
    def run_full_pipeline(self, data_path="data/classification_data.csv", output_dir="outputs",
                         use_zip_holdout=False):
        """Run the complete classification pipeline"""
        print("="*70)
        print("GENTRIFICATION CLASSIFICATION MODEL")
        print("="*70)
        
        X, y, zip_codes = self.load_data(data_path)
        X_train, X_test, y_train, y_test = self.prepare_data(
            X, y, zip_codes=zip_codes, split_strategy='zip_holdout' if use_zip_holdout else 'random'
        )
        
        self.train_models(X_train, y_train)
        results_df = self.evaluate_models(X_train, X_test, y_train, y_test)
        importance_df = self.analyze_feature_importance(X_train, y_train)
        self.create_visualizations(X_test, y_test, results_df, importance_df, output_dir)
        
        results_df.to_csv(Path(output_dir) / 'classification_results.csv', index=False)
        importance_df.to_csv(Path(output_dir) / 'classification_feature_importance.csv', index=False)
        self.save_model(Path(output_dir).parent / "models" / "best_classifier.pkl")
        
        print("\n" + "="*70)
        print("CLASSIFICATION COMPLETE")
        print("="*70)
        
        return results_df, importance_df

