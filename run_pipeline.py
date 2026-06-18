#!/usr/bin/env python3
"""
Complete ML Pipeline Runner
Runs all stages: preprocessing → model training → evaluation
"""

import sys
from pathlib import Path

# Add code directory to path
CODE_DIR = Path(__file__).parent / "code"
sys.path.insert(0, str(CODE_DIR))

from preprocessing import DataPreprocessor
from baseline_models import BaselineModels
from model_evaluation import ModelEvaluator

# Define paths
PROJECT_ROOT = Path(__file__).parent
DATA_FILE = PROJECT_ROOT / "data/processed/flood_weather_dataset.csv"

def main():
    print("\n" + "🚀 " * 30)
    print("COMPLETE ML PIPELINE EXECUTION")
    print("🚀 " * 30 + "\n")
    
    # Phase 1: Preprocessing
    print("\n" + "=" * 70)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 70)
    preprocessor = DataPreprocessor(DATA_FILE)
    preprocessor_output = preprocessor.run_full_pipeline()
    
    # Phase 2: Model Training
    print("\n" + "=" * 70)
    print("PHASE 2: BASELINE MODEL TRAINING")
    print("=" * 70)
    models_trainer = BaselineModels(
        preprocessor_output['X_train'],
        preprocessor_output['X_test'],
        preprocessor_output['y_train'],
        preprocessor_output['y_test'],
        preprocessor_output['feature_names']
    )
    results, comparison_df = models_trainer.run_full_training()
    
    # Phase 3: Model Evaluation
    print("\n" + "=" * 70)
    print("PHASE 3: MODEL EVALUATION & VISUALIZATION")
    print("=" * 70)
    evaluator = ModelEvaluator(
        results=results,
        feature_importance=models_trainer.feature_importance,
        feature_names=preprocessor_output['feature_names'],
        y_test=preprocessor_output['y_test']
    )
    figs, report_path = evaluator.run_full_evaluation()
    
    # Final Summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE EXECUTION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Results saved to: {PROJECT_ROOT / 'results'}")
    print("\n📈 Output Files:")
    print("   • results/training_data.csv")
    print("   • results/test_data.csv")
    print("   • results/model_metrics.csv")
    print("   • results/random_forest_model.pkl")
    print("   • results/logistic_regression_model.pkl")
    print("   • results/model_performance_comparison.png")
    print("   • results/roc_curves.png")
    print("   • results/confusion_matrices.png")
    print("   • results/feature_importance_*.png")
    print("   • results/evaluation_report.txt")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
