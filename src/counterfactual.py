import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


def flip_attribute(X_df, column, prob=0.2):
    """
    flips a protected demographic attribute according to a probability
    :param X_df: the dataset
    :param column: the attribute to flip
    :param prob: the probability
    :return: the dataset with the flipped attributes
    """
    if column not in X_df.columns:
        raise ValueError(f"Column '{column}' not found in data.")
    X_flipped = X_df.copy()
    mask = np.random.rand(len(X_flipped)) < prob
    X_flipped.loc[mask, column] = 1 - X_flipped.loc[mask, column]
    return X_flipped


def make_bld_from_df(X, y, protected_attr='sex', label_name='income'):
    """
    convert to an AIF360 binaryLabelDataset
    :param X: the data
    :param y: the labels
    :param protected_attr: the protected attribute's column name
    :param label_name: the label's column name
    :return: a binaryLabelDataset of the data we sent to the function
    """
    df_combined = X.copy()
    df_combined[label_name] = y
    return BinaryLabelDataset(
        df=df_combined,
        label_names=[label_name],
        protected_attribute_names=[protected_attr]
    )


def compute_fairness_metrics(X_test, y_test, y_pred, model_name, setting, fold, protected_attr):
    """
    calculates fairness metrics
    :param X_test: the data
    :param y_test: the real labels
    :param y_pred: the predictions
    :param model_name: the ML model
    :param setting: probability
    :param fold: fold number
    :param protected_attr: the protected attribute
    :return: a csv file with the fairness metrics
    """
    test_bld_true = make_bld_from_df(X_test, y_test, protected_attr)
    test_bld_pred = make_bld_from_df(X_test, y_pred, protected_attr)

    unprivileged = {protected_attr: 0}
    privileged = {protected_attr: 1}

    metric = ClassificationMetric(
        test_bld_true, test_bld_pred,
        unprivileged_groups=[unprivileged],
        privileged_groups=[privileged]
    )

    return {
        "Fold": fold,
        "Model": model_name,
        "Setting": setting,
        "Statistical Parity": metric.statistical_parity_difference(),
        "Disparate Impact": metric.disparate_impact(),
        "Equal Opportunity": metric.equal_opportunity_difference(),
        "Average Odds": metric.average_odds_difference()
    }

def train_eval_fold(X_train, y_train, X_test, y_test, setting, fold, protected_attr):
    """
    train the model and evaluate on a single fold
    :param X_train: the data (train)
    :param y_train: the labels (train)
    :param X_test: the data (test)
    :param y_test: the labels (test)
    :param setting: probability
    :param fold: fold number
    :param protected_attr: name of the protected attributes
    :return: 2 csv files, one with performance metrics and the other with fairness metrics
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "MLP": MLPClassifier(max_iter=1000)
    }

    perf_results = []
    fairness_results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        perf_results.append({
            "Fold": fold,
            "Model": name,
            "Setting": setting,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0)
        })

        fairness = compute_fairness_metrics(X_test, y_test, y_pred, name, setting, fold, protected_attr)
        fairness_results.append(fairness)

    return perf_results, fairness_results


def run_counterfactual_experiment(
    df, protected_attr='sex', label_col='income',
    output_prefix='gender', export_csv=True
):
    """
    runs the full counterfactual experiment
    :param df: the dataset
    :param protected_attr: the name of the protected attribute
    :param label_col: the name of the label column
    :param output_prefix: protected attribute, to save csv file according to it
    :param export_csv: if set to true, saves the results to a csv file
    :return: a csv file with the results for the 5 fold cross validation counterfactual experiment
    """
    X_all = df.drop(columns=[label_col]).copy()
    y_all = df[label_col].copy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_perf = []
    all_fairness = []

    # Baseline: No flipping
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
        X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]

        perf, fair = train_eval_fold(X_train, y_train, X_val, y_val, "Original", fold, protected_attr)
        all_perf.extend(perf)
        all_fairness.extend(fair)

    # Flip experiments
    for prob in np.arange(0.1, 1.1, 0.1):
        setting_name = f"Flipped {protected_attr} p={prob:.1f}"
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
            X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
            X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]

            X_train_flipped = flip_attribute(X_train, protected_attr, prob=prob)
            perf, fair = train_eval_fold(X_train_flipped, y_train, X_val, y_val, setting_name, fold, protected_attr)
            all_perf.extend(perf)
            all_fairness.extend(fair)

    # Summarize results
    df_perf = pd.DataFrame(all_perf)
    df_fair = pd.DataFrame(all_fairness)

    perf_summary = (
        df_perf.groupby(["Model", "Setting"], as_index=False)
        [["Accuracy", "Precision", "Recall", "F1 Score"]]
        .mean()
        .round(4)
    )

    fair_summary = (
        df_fair.groupby(["Model", "Setting"], as_index=False)
        [["Statistical Parity", "Disparate Impact", "Equal Opportunity", "Average Odds"]]
        .mean()
        .round(4)
    )

    print(f"🔹 Performance Summary ({protected_attr} flip):\n", perf_summary)
    print(f"\n🔸 Fairness Summary ({protected_attr} flip):\n", fair_summary)

    if export_csv:
        perf_summary.to_csv(f'performance_{output_prefix}.csv', index=False)
        fair_summary.to_csv(f'fairness_{output_prefix}.csv', index=False)

    return perf_summary, fair_summary
