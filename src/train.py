import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE


def main():
    # Load already preprocessed & split data
    X_train = pd.read_csv('data/train_X.csv')
    y_train = pd.read_csv('data/train_y.csv').values.ravel()

    # -------------------------------
    # Handle class imbalance
    # -------------------------------
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # =====================================================
    # Model 1: Random Forest (Bagging-based ensemble)
    # =====================================================
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(random_state=42)

    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    grid_rf = GridSearchCV(
        rf,
        param_grid_rf,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_rf.fit(X_train_res, y_train_res)

    print("Best RF params:", grid_rf.best_params_)
    print("Best RF CV f1_macro:", grid_rf.best_score_)

    joblib.dump(grid_rf.best_estimator_, 'model_rf_best.pkl')
    print("Random Forest model saved: model_rf_best.pkl")

    # =====================================================
    # Model 2: Logistic Regression (Linear baseline)
    # =====================================================
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_res, y_train_res)

    joblib.dump(lr, 'model_lr.pkl')
    print("Logistic Regression model saved: model_lr.pkl")

    # =====================================================
    # Model 3: Gradient Boosting (Boosting-based ensemble)
    # =====================================================
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingClassifier(random_state=42)

    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }

    grid_gb = GridSearchCV(
        gb,
        param_grid_gb,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_gb.fit(X_train_res, y_train_res)

    print("Best GB params:", grid_gb.best_params_)
    print("Best GB CV f1_macro:", grid_gb.best_score_)

    joblib.dump(grid_gb.best_estimator_, 'model_gb_best.pkl')
    print("Gradient Boosting model saved: model_gb_best.pkl")


if __name__ == '__main__':
    main()
