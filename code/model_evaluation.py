"""
Model Evaluation & Visualization Module
Generates performance metrics, visualizations, and evaluation reports
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Matplotlib/Seaborn not installed for visualizations")

from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self, results, feature_importance, feature_names, y_test):
        """Initialize evaluator with model results"""
        self.results = results
        self.feature_importance = feature_importance
        self.feature_names = feature_names
        self.y_test = y_test
        self.figs = {}
        
    def plot_performance_comparison(self):
        """Create performance comparison bar charts"""
        if not PLOTTING_AVAILABLE:
            print("⚠️  Plotting skipped (matplotlib not available)")
            return
        
        print("\n📊 Generating performance comparison plots...")
        
        # Extract metrics
        metrics_data = {
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'AUC-ROC': []
        }
        
        for model_name, result in self.results.items():
            metrics_data['Model'].append(model_name)
            metrics_data['Accuracy'].append(result['accuracy'])
            metrics_data['Precision'].append(result['precision'])
            metrics_data['Recall'].append(result['recall'])
            metrics_data['F1-Score'].append(result['f1'])
            metrics_data['AUC-ROC'].append(result['auc'])
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(metrics):
                metric = metrics[idx]
                bars = ax.bar(df_metrics['Model'], df_metrics[metric], color=colors, alpha=0.7, edgecolor='black')
                ax.set_title(metric, fontweight='bold', fontsize=12)
                ax.set_ylabel('Score', fontsize=10)
                ax.set_ylim([0, 1])
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
                
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.axis('off')
        
        # Remove the extra subplot
        axes.flat[5].axis('off')
        
        plt.tight_layout()
        fig_path = RESULTS_DIR / "model_performance_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {fig_path}")
        self.figs['performance'] = fig_path
        plt.close()
    
    def plot_roc_curves(self):
        """Generate ROC curves for all models"""
        if not PLOTTING_AVAILABLE:
            print("⚠️  ROC plotting skipped (matplotlib not available)")
            return
        
        print("\n📈 Generating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors_roc = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for (model_name, result), color in zip(self.results.items(), colors_roc):
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = RESULTS_DIR / "roc_curves.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {fig_path}")
        self.figs['roc'] = fig_path
        plt.close()
    
    def plot_confusion_matrices(self):
        """Generate confusion matrices for all models"""
        if not PLOTTING_AVAILABLE:
            print("⚠️  Confusion matrix plotting skipped (matplotlib not available)")
            return
        
        print("\n🔲 Generating confusion matrices...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
        
        for (model_name, result), ax in zip(self.results.items(), axes):
            cm = confusion_matrix(self.y_test, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=False, square=True, annot_kws={'size': 12})
            
            ax.set_title(model_name, fontweight='bold')
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_xticklabels(['No Flood', 'Flood'])
            ax.set_yticklabels(['No Flood', 'Flood'])
        
        plt.tight_layout()
        fig_path = RESULTS_DIR / "confusion_matrices.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {fig_path}")
        self.figs['confusion'] = fig_path
        plt.close()
    
    def plot_feature_importance(self):
        """Generate feature importance visualizations"""
        if not PLOTTING_AVAILABLE:
            print("⚠️  Feature importance plotting skipped (matplotlib not available)")
            return
        
        print("\n🔧 Generating feature importance plots...")
        
        # Create individual plots for each model
        for model_name, importance in self.feature_importance.items():
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort features by importance
            sorted_idx = np.argsort(importance)[::-1][:15]  # Top 15 features
            top_features = [self.feature_names[i] for i in sorted_idx]
            top_importance = importance[sorted_idx]
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(top_features)), top_importance, color='steelblue', edgecolor='navy', alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Importance Score', fontweight='bold', fontsize=11)
            ax.set_title(f'Top 15 Features - {model_name}', fontweight='bold', fontsize=13)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, top_importance)):
                ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
            
            plt.tight_layout()
            safe_name = model_name.lower().replace(' ', '_')
            fig_path = RESULTS_DIR / f"feature_importance_{safe_name}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {fig_path}")
            self.figs[f'importance_{safe_name}'] = fig_path
            plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive text summary report"""
        print("\n" + "="*60)
        print("📄 GENERATING SUMMARY REPORT")
        print("="*60)
        
        report_path = RESULTS_DIR / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FLOOD PREDICTION MODEL - EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("PROJECT: AI-Based Natural Disaster Prediction Web App\n")
            f.write("FOCUS: Flood Risk Prediction for Khyber Pakhtunkhwa Region\n")
            f.write("TARGET REGION: Swat & Upper Dir Districts\n\n")
            
            # Model Performance Summary
            f.write("-" * 70 + "\n")
            f.write("1. MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 70 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  • Accuracy:   {result['accuracy']:.4f}\n")
                f.write(f"  • Precision:  {result['precision']:.4f}\n")
                f.write(f"  • Recall:     {result['recall']:.4f}\n")
                f.write(f"  • F1-Score:   {result['f1']:.4f}\n")
                f.write(f"  • AUC-ROC:    {result['auc']:.4f}\n")
                f.write(f"  • Specificity: {result['specificity']:.4f}\n")
                f.write(f"  Confusion Matrix:\n")
                f.write(f"    - True Negatives:  {result['tn']}\n")
                f.write(f"    - False Positives: {result['fp']}\n")
                f.write(f"    - False Negatives: {result['fn']}\n")
                f.write(f"    - True Positives:  {result['tp']}\n\n")
            
            # Findings
            f.write("-" * 70 + "\n")
            f.write("2. KEY FINDINGS\n")
            f.write("-" * 70 + "\n\n")
            
            best_f1_model = max(self.results.items(), key=lambda x: x[1]['f1'])
            f.write(f"✓ Best Overall Model (F1-Score): {best_f1_model[0]}\n")
            f.write(f"✓ Highest Recall (Flood Detection): {max(self.results.items(), key=lambda x: x[1]['recall'])[0]}\n")
            f.write(f"✓ Highest Precision (False Alarm Minimization): {max(self.results.items(), key=lambda x: x[1]['precision'])[0]}\n\n")
            
            # Recommendations
            f.write("-" * 70 + "\n")
            f.write("3. RECOMMENDATIONS\n")
            f.write("-" * 70 + "\n\n")
            
            f.write("• DEPLOYMENT: Use the model with highest recall for public safety\n")
            f.write("  (Better to warn unnecessarily than miss actual floods)\n\n")
            f.write("• FEATURE ENGINEERING: Consider domain-specific features:\n")
            f.write("  - Historical flood data integration\n")
            f.write("  - Terrain elevation and watershed data\n")
            f.write("  - Antecedent moisture conditions\n\n")
            f.write("• MODEL IMPROVEMENTS:\n")
            f.write("  - Collect more positive (flood) samples for better balance\n")
            f.write("  - Implement ensemble methods combining all three models\n")
            f.write("  - Use temporal models (LSTM) for time-series patterns\n\n")
            f.write("• INTEGRATION:\n")
            f.write("  - Connect with real-time weather APIs\n")
            f.write("  - Deploy as REST API for web application\n")
            f.write("  - Add SHAP-based explainability for user trust\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Generated by: AI-Based Disaster Prediction System\n")
            f.write("=" * 70 + "\n")
        
        print(f"✅ Report saved: {report_path}")
        return report_path
    
    def run_full_evaluation(self):
        """Execute complete evaluation pipeline"""
        print("\n" + "🚀 "*30)
        print("STARTING MODEL EVALUATION")
        print("🚀 "*30 + "\n")
        
        # Visualizations
        self.plot_performance_comparison()
        self.plot_roc_curves()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        
        # Report
        report_path = self.generate_summary_report()
        
        print("\n" + "✅ "*30)
        print("EVALUATION COMPLETE!")
        print("✅ "*30)
        
        print(f"\n📁 All results saved to: {RESULTS_DIR}")
        
        return self.figs, report_path


if __name__ == "__main__":
    print("⚠️  This module should be imported from the main pipeline script")
