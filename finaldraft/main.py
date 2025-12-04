#!/usr/bin/env python3
"""
Gentrification Classification Model
Predicts high rent growth zip codes using Yelp business ecosystem features.
"""

from classification_preparation import ClassificationDataPreparation
from classification_model import GentrificationClassifier
from pathlib import Path

def main():
    print("="*70)
    print("GENTRIFICATION CLASSIFICATION MODEL")
    print("="*70)
    
    prep = ClassificationDataPreparation(data_dir="raw_data")
    processed_data = prep.save_classification_data(
        "data/classification_data.csv",
        top_percentile=0.25
    )
    
    print(f"\nData: {processed_data.shape[0]} zip codes, {len(prep.feature_names)} features")
    print(f"High growth: {processed_data['target'].sum()} ({processed_data['target'].mean()*100:.1f}%)")
    
    classifier = GentrificationClassifier(random_state=42)
    results_df, importance_df = classifier.run_full_pipeline(
        data_path="data/classification_data.csv",
        output_dir="outputs",
        use_zip_holdout=False
    )
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE")
    print("="*70)
    print(results_df[['model', 'test_accuracy', 'test_precision', 'test_recall', 
                     'test_f1', 'test_roc_auc']].to_string(index=False))
    
    print("\n" + "="*70)
    print("TOP 10 FEATURES")
    print("="*70)
    print(importance_df.head(10).to_string(index=False))
    
    gent_features = importance_df[importance_df['feature'].str.contains('gentrification|business', case=False, na=False)]
    if len(gent_features) > 0:
        print("\n" + "="*70)
        print("TOP GENTRIFICATION/BUSINESS FEATURES")
        print("="*70)
        print(gent_features.head(10).to_string(index=False))
    
    print(f"\n✓ Results saved to: outputs/")
    print(f"✓ Model saved to: models/best_classifier.pkl")

if __name__ == "__main__":
    main()

