import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_models():
    """
    decides on the models for the experiment
    :return: a dict of 'model_name' and model
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "MLP": MLPClassifier(max_iter=1000)
    }

def compute_fairness_by_attr(attr_values, privileged_value, y_true, y_pred):
    """
    computes fairnes scores by protected attribute
    :param attr_values: sex/race
    :param privileged_value: the privileged group
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :return: stat_parity, disparate_impact, eq_odds_diff
    """
    mask_priv = attr_values == privileged_value
    mask_unpriv = ~mask_priv

    p_priv = np.mean(y_pred[mask_priv])
    p_unpriv = np.mean(y_pred[mask_unpriv])
    stat_parity = p_unpriv - p_priv
    disparate_impact = p_unpriv / p_priv if p_priv > 0 else np.nan

    tn, fp, fn, tp = confusion_matrix(y_true[mask_priv], y_pred[mask_priv], labels=[0, 1]).ravel()
    tpr_priv = tp / (tp + fn) if (tp + fn) else 0
    fpr_priv = fp / (fp + tn) if (fp + tn) else 0

    tn, fp, fn, tp = confusion_matrix(y_true[mask_unpriv], y_pred[mask_unpriv], labels=[0, 1]).ravel()
    tpr_unpriv = tp / (tp + fn) if (tp + fn) else 0
    fpr_unpriv = fp / (fp + tn) if (fp + tn) else 0

    eq_odds_diff = abs(tpr_priv - tpr_unpriv) + abs(fpr_priv - fpr_unpriv)
    return stat_parity, disparate_impact, eq_odds_diff

def evaluate_models(dataset, n_splits=5, seed=42, output_csv='baseline_results.csv'):
    """
    conducts experiments using cross validation
    :param dataset: the dataset to run the experiments on
    :param n_splits: number of folds
    :param seed: random seed for reproducibility
    :param output_csv: name of the file to save the results to
    :return: a csv file with the baseline results
    """
    models = get_models()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []

    for name, model in models.items():
        accs, precs, recs, f1s = [], [], [], []
        sex_spds, sex_dis, sex_eods = [], [], []
        race_spds, race_dis, race_eods = [], [], []

        for train_idx, test_idx in kf.split(dataset.features, dataset.labels.ravel()):
            X_train, X_test = dataset.features[train_idx], dataset.features[test_idx]
            y_train = dataset.labels[train_idx].ravel()
            y_test = dataset.labels[test_idx].ravel()
            prot_attr = dataset.protected_attributes[test_idx]

            sex_attr = prot_attr[:, 0]
            race_attr = prot_attr[:, 1]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred))
            recs.append(recall_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))

            sex_stat, sex_di, sex_eq = compute_fairness_by_attr(sex_attr, 1, y_true=y_test, y_pred=y_pred)
            race_stat, race_di, race_eq = compute_fairness_by_attr(race_attr, 1, y_true=y_test, y_pred=y_pred)

            sex_spds.append(sex_stat)
            sex_dis.append(sex_di)
            sex_eods.append(sex_eq)
            race_spds.append(race_stat)
            race_dis.append(race_di)
            race_eods.append(race_eq)

        results.append({
            "Model": name,
            "Accuracy": np.mean(accs),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "F1 Score": np.mean(f1s),
            "Stat Parity (Sex)": np.mean(sex_spds),
            "Disparate Impact (Sex)": np.mean(sex_dis),
            "Equalized Odds (Sex)": np.mean(sex_eods),
            "Stat Parity (Race)": np.mean(race_spds),
            "Disparate Impact (Race)": np.mean(race_dis),
            "Equalized Odds (Race)": np.mean(race_eods),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    return results_df
