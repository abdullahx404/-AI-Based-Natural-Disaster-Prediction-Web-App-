"""
Baseline ML Models Module
Trains and evaluates baseline models: Logistic Regression, Random Forest, and XGBoost
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️  XGBoost not available: {type(e).__name__}")
    print("   Proceeding with Logistic Regression and Random Forest only")
    print("   (To enable XGBoost on macOS: brew install libomp)")

from pathlib import Path
import pickle
import json

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class BaselineModels:
    """Train and evaluate baseline ML models for flood prediction"""
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        """Initialize with training and test data"""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def train_logistic_regression(self):
        """Train Logistic Regression baseline"""
        print("\n" + "="*60)
        print("🔵 LOGISTIC REGRESSION")
        print("="*60)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1
        )
        
        print("📚 Training...")
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
        print(f"✅ Model trained!")
        print(f"   • Cross-validation F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance (coefficients)
        self.feature_importance['Logistic Regression'] = np.abs(model.coef_[0])
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest baseline"""
        print("\n" + "="*60)
        print("🌲 RANDOM FOREST")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print("📚 Training...")
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
        print(f"✅ Model trained!")
        print(f"   • Cross-validation F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        self.feature_importance['Random Forest'] = model.feature_importances_
        
        return model
    
    def train_xgboost(self):
        """Train XGBoost baseline"""
        if not XGBOOST_AVAILABLE:
            print("\n⚠️  XGBoost skipped - OpenMP not available")
            print("   To fix: brew install libomp")
            return None
        
        print("\n" + "="*60)
        print("⚡ XGBOOST")
        print("="*60)
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        print("📚 Training...")
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
        print(f"✅ Model trained!")
        print(f"   • Cross-validation F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        self.feature_importance['XGBoost'] = model.feature_importances_
        
        return model
    
    def evaluate_model(self, model_name, model):
        """Evaluate a single model on test set"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[model_name] = results
        
        print(f"   ✅ Accuracy:   {accuracy:.4f}")
        print(f"   ✅ Precision:  {precision:.4f}")
        print(f"   ✅ Recall:     {recall:.4f}")
        print(f"   ✅ F1-Score:   {f1:.4f}")
        print(f"   ✅ AUC-ROC:    {auc:.4f}")
        print(f"   ✅ Specificity: {specificity:.4f}")
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("📋 MODEL EVALUATION ON TEST SET")
        print("="*60)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON REPORT")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1'],
                'AUC-ROC': result['auc'],
                'Specificity': result['specificity']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n🏆 Performance Summary:")
        print(comparison_df.to_string(index=False))
        
        # Identify best models
        print("\n⭐ Best Models:")
        print(f"   • Highest Accuracy:   {comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']}")
        print(f"   • Highest Precision:  {comparison_df.loc[comparison_df['Precision'].idxmax(), 'Model']}")
        print(f"   • Highest Recall:     {comparison_df.loc[comparison_df['Recall'].idxmax(), 'Model']}")
        print(f"   • Highest F1-Score:   {comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']}")
        print(f"   • Highest AUC-ROC:    {comparison_df.loc[comparison_df['AUC-ROC'].idxmax(), 'Model']}")
        
        # Save to CSV
        csv_path = RESULTS_DIR / "model_metrics.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n💾 Metrics saved to: {csv_path}")
        
        return comparison_df
    
    def generate_detailed_reports(self):
        """Generate detailed classification reports for each model"""
        print("\n" + "="*60)
        print("📝 DETAILED CLASSIFICATION REPORTS")
        print("="*60)
        
        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            print(classification_report(
                self.y_test,
                result['y_pred'],
                target_names=['No Flood', 'Flood'],
                zero_division=0
            ))
    
    def save_models(self):
        """Save trained models to disk"""
        print("\n" + "="*60)
        print("💾 SAVING MODELS")
        print("="*60)
        
        for model_name, model in self.models.items():
            model_path = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ {model_name} saved: {model_path}")
    
    def save_feature_importance(self):
        """Save feature importance data"""
        print("\n" + "="*60)
        print("💾 SAVING FEATURE IMPORTANCE")
        print("="*60)
        
        importance_data = {}
        for model_name, importance in self.feature_importance.items():
            feature_importance_dict = {
                name: float(imp) for name, imp in zip(self.feature_names, importance)
            }
            importance_data[model_name] = feature_importance_dict
        
        json_path = RESULTS_DIR / "feature_importance.json"
        with open(json_path, 'w') as f:
            json.dump(importance_data, f, indent=2)
        
        print(f"✅ Feature importance saved: {json_path}")
        
        # Also save as CSV for each model
        for model_name, importance in self.feature_importance.items():
            df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            csv_path = RESULTS_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ {model_name} importance saved: {csv_path}")
    
    def run_full_training(self):
        """Execute complete training and evaluation pipeline"""
        print("\n" + "🚀 "*30)
        print("STARTING BASELINE MODELS TRAINING")
        print("🚀 "*30 + "\n")
        
        # Train models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        # Evaluate
        self.evaluate_all_models()
        comparison_df = self.generate_comparison_report()
        self.generate_detailed_reports()
        
        # Save
        self.save_models()
        self.save_feature_importance()
        
        print("\n" + "✅ "*30)
        print("TRAINING COMPLETE!")
        print("✅ "*30)
        
        return self.results, comparison_df


if __name__ == "__main__":
    print("⚠️  This module should be imported from the main pipeline script")
    print("   or run: python -c \"from preprocessing import DataPreprocessor; from baseline_models import BaselineModels\"")
