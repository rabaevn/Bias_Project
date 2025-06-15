import pandas as pd
from scipy.stats import ttest_ind


def run_ttest_on_predictions(predictions_df, model_col='Model', setting_col='Setting',
                             success_col='Success', baseline_label='Original',
                             export_csv=True, output_file='ttest_results.csv'):
    """
    Run t-tests comparing success rates of counterfactual settings to baseline.

    Parameters:
        predictions_df (pd.DataFrame): Must include model, setting, and binary success (0/1)
        model_col (str): Column name for model identifier
        setting_col (str): Column name for flip setting
        success_col (str): Binary success values
        baseline_label (str): Value in `setting_col` representing the unflipped baseline
        export_csv (bool): Save results to CSV
        output_file (str): Output CSV file name

    Returns:
        pd.DataFrame: T-test results
    """

    results = []
    grouped = predictions_df.groupby([model_col, setting_col])

    for (model, setting), test_group in grouped:
        if setting == baseline_label:
            continue

        baseline_group = predictions_df[
            (predictions_df[model_col] == model) &
            (predictions_df[setting_col] == baseline_label)
            ]

        if baseline_group.empty:
            continue

        test_success = test_group[success_col]
        baseline_success = baseline_group[success_col]

        t_stat, p_val = ttest_ind(test_success, baseline_success, equal_var=False)

        results.append({
            'Model': model,
            'Flip_Setting': setting,
            'T_stat': t_stat,
            'P_value': p_val,
            'Significant (p<0.05)': p_val < 0.05
        })

    df_ttest = pd.DataFrame(results).sort_values(by=['Model', 'Flip_Setting'])

    print("T-test Results Comparing Counterfactual to Baseline:")
    print(df_ttest)

    if export_csv:
        df_ttest.to_csv(output_file, index=False)

    return df_ttest
