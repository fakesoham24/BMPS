import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, model_name="Model", custom_threshold=0.5):
    """Evaluate mapping of predictions to actual values and compute metrics."""
    logger.info(f"Evaluating {model_name} with threshold {custom_threshold}...")
    
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    if y_prob is not None:
        y_pred = (y_prob >= custom_threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = y_pred
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    return metrics, y_pred, y_prob

def plot_confusion_matrix(y_test, y_pred, save_path="../models/confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Confusion Matrix saved at {save_path}")

def plot_roc_curve(y_test, y_prob, roc_auc, save_path="../models/roc_curve.png"):
    """Plot and save ROC Curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', color='blue')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"ROC Curve saved at {save_path}")

def plot_feature_importance(model, preprocessor, save_path="../models/feature_importance.png"):
    """Extract and plot feature importance if applicable."""
    try:
        classifier = model.named_steps['classifier']
        if not hasattr(classifier, 'feature_importances_'):
            return
            
        importances = classifier.feature_importances_
        
        # Extract feature names from ColumnTransformer
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out().tolist()
        all_features = num_features + cat_features
        
        # Sort features conceptually
        indices = np.argsort(importances)[::-1][:20] # Top 20 features
        
        plt.figure(figsize=(10, 8))
        plt.title('Top 20 Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [all_features[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Feature Importance saved at {save_path}")
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")
