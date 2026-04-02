import os
import joblib
import logging
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline # Changed to imblearn Pipeline
from imblearn.over_sampling import SMOTE
from data_processing import load_data, get_preprocessor
from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_best_threshold(model, X_val, y_val):
    """Iterate over possible thresholds to find the one that maximizes F1-score."""
    logger.info("Finding optimal decision threshold for F1-score...")
    y_prob = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_val, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh
            
    logger.info(f"Optimal Threshold: {best_threshold:.2f} (F1: {best_f1:.4f} on Validation Set)")
    return best_threshold

def main():
    data_path = '../data/data.csv'
    
    logger.info("Starting training process with SMOTE...")
    # 1. Load Data
    X, y = load_data(data_path)
    
    # 2. Train Test Split
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Get Preprocessing Pipeline
    preprocessor = get_preprocessor(X_train)
    
    # 4. Define base XGBoost Model (no class weights, relying on SMOTE)
    logger.info("Configuring SMOTE Pipeline...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    
    # 5. Advanced Hyperparameter Tuning
    logger.info("Starting advanced Hyperparameter Tuning for XGBoost...")
    # These parameters help prevent overfitting on synthetic SMOTE data
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__subsample': [0.8],
        'classifier__colsample_bytree': [0.8]
    }
    
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=1)
    
    logger.info("Fitting GridSearch (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    tuned_model = grid_search.best_estimator_
    logger.info(f"Best Params: {grid_search.best_params_}")
    
    # 6. Optimal Threshold Finding 
    # Use training split for validation so we don't leak the test set
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    logger.info("Refitting optimal model momentarily to find threshold safely...")
    threshold_model = grid_search.best_estimator_
    threshold_model.fit(X_train_sub, y_train_sub)
    optimal_threshold = find_best_threshold(threshold_model, X_val, y_val)
    
    # Refit once again on the full training set (GridSearchCV does this by default, but our threshold model overwrote inner state if we aren't careful, so best to re-fit just in case)
    logger.info("Finalizing model on full Train Set...")
    tuned_model.fit(X_train, y_train)
    
    # 7. Final Evaluation with optimal threshold
    logger.info("Evaluating Final Model on independent Test Set...")
    tuned_metrics, tuned_y_pred, tuned_y_prob = evaluate_model(tuned_model, X_test, y_test, model_name="Tuned XGBoost (SMOTE)", custom_threshold=optimal_threshold)
    
    logger.info(f"Final Tuned metrics (Threshold {optimal_threshold:.2f}):\n" + "\n".join([f"  {k}: {v:.4f}" for k, v in tuned_metrics.items()]))
    
    # 8. Visualizations and Export
    os.makedirs('../models', exist_ok=True)
    plot_confusion_matrix(y_test, tuned_y_pred, save_path="../models/confusion_matrix.png")
    plot_roc_curve(y_test, tuned_y_prob, tuned_metrics['ROC-AUC'], save_path="../models/roc_curve.png")
    
    try:
        plot_feature_importance(tuned_model, tuned_model.named_steps['preprocessor'], save_path="../models/feature_importance.png")
    except Exception as e:
        logger.warning(f"Feature importance skip: {e}")
    
    model_path = '../models/best_model.pkl'
    joblib.dump(tuned_model, model_path)
    logger.info(f"Model successfully saved to {model_path}")
    
    threshold_path = '../models/optimal_threshold.pkl'
    joblib.dump(optimal_threshold, threshold_path)
    logger.info(f"Threshold successfully saved to {threshold_path}")

if __name__ == '__main__':
    main()
