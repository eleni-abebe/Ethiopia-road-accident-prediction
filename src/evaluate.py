import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, model_name, X_test, y_test):
    print("\n" + "=" * 50)
    print(f"Evaluating model: {model_name}")
    print("=" * 50)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['Fatal', 'Serious', 'Slight']
    ))

    f1 = f1_score(y_test, y_pred, average='macro')
    print("Macro F1-score:", f1)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Fatal', 'Serious', 'Slight'],
        yticklabels=['Fatal', 'Serious', 'Slight']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.show()

    print(f"Confusion matrix saved as: confusion_matrix_{model_name}.png")


def main():
    # Load test data
    X_test = pd.read_csv('data/test_X.csv')
    y_test = pd.read_csv('data/test_y.csv').values.ravel()

    # Models to evaluate
    models = {
        "Logistic_Regression": "model_lr.pkl",
        "Random_Forest": "model_rf_best.pkl",
        "Gradient_Boosting": "model_gb_best.pkl"
    }

    for model_name, model_path in models.items():
        model = joblib.load(model_path)
        print("\nLoaded model:", type(model).__name__)
        evaluate_model(model, model_name, X_test, y_test)


if __name__ == '__main__':
    main()
