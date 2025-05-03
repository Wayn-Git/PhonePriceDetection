# Improved logistic-regression pipeline with cross-validation, hyperparameter tuning, detailed metrics, and cleaner structure
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(train_path, test_path=None):
    """
    Load training (and optional test) datasets from CSV files.
    """
    train = pd.read_csv(train_path)
    X = train.drop("price_range", axis=1)
    y = train["price_range"]
    test = None
    if test_path:
        test = pd.read_csv(test_path)
    return X, y, test


def plot_ram_vs_price(X, y):
    """
    Display a labeled 2D histogram of RAM vs. price range.
    """
    plt.figure()
    plt.hist2d(X['ram'], y, bins=(50, 4))
    plt.xlabel('RAM (MB)')
    plt.ylabel('Price Range (0=low, 3=high)')
    plt.title('2D Histogram: RAM vs. Price Range')
    plt.colorbar(label='Count')
    plt.show()


def build_pipeline():
    """
    Create a sklearn Pipeline with scaling and multinomial logistic regression.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=2000,
            random_state=42
        ))
    ])


def evaluate_model(pipe, X, y):
    """
    Evaluate the pipeline using 5-fold cross-validation, and print accuracy, classification report, and confusion matrix.
    """
    # cross-validated accuracy
    cv_scores = cross_val_score(pipe, X, y, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # cross-validated predictions for detailed metrics
    y_pred = cross_val_predict(pipe, X, y, cv=5)
    print("\nClassification report:")
    print(classification_report(y, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))


def tune_hyperparameters(pipe, X, y):
    """
    Perform GridSearchCV to find optimal regularization C and penalty.
    """
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l2'],  # lbfgs supports only l2 for multinomial
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid.fit(X, y)
    print(f"Best CV score: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")
    return grid.best_estimator_


def train_final_model(pipe, X, y):
    """
    Fit the pipeline on the full training data.
    """
    pipe.fit(X, y)
    return pipe


def make_submission(pipe, test_df, out_path='submission.csv'):
    """
    Generate predictions on test set and save submission CSV.
    """
    X_test = test_df.drop('id', axis=1)
    preds = pipe.predict(X_test)
    submission = pd.DataFrame({
        'id': test_df['id'],
        'price_range': preds
    })
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


if __name__ == '__main__':
    # Load data
    X, y, test = load_data('train.csv', 'test.csv')

    # Exploratory plot
    plot_ram_vs_price(X, y)

    # Build baseline pipeline
    pipeline = build_pipeline()

    # Evaluate with cross-validation
    evaluate_model(pipeline, X, y)

    # Hyperparameter tuning
    best_pipeline = tune_hyperparameters(pipeline, X, y)

    # Final training on all data
    final_model = train_final_model(best_pipeline, X, y)

    # Generate submission
    make_submission(final_model, test)
