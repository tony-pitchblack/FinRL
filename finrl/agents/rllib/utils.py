RESULT_KEYS_TO_INCLUDE = [
    'sharpe_ratio_MA',
    'ann_return_MA',
    'mdd_MA',

    'sharpe_ratio_EMA',
    'ann_return_EMA',
    'mdd_EMA',
]

def print_result(result):
    print()
    for key in result['env_runners'].keys():
        for include_key in RESULT_KEYS_TO_INCLUDE:
            if key.startswith(include_key):
                print(f"train/{key}: {round(result['env_runners'][key], 2)}")
                break

    for key in result['evaluation']['env_runners'].keys():
        for include_key in RESULT_KEYS_TO_INCLUDE:
            if key.startswith(include_key):
                print(f"val/{key}: {round(result['evaluation']['env_runners'][key], 2)}")
                break
    print()

import pandas as pd

def aggregate_results(results):
    """
    Combines all metrics from a list of results into a single DataFrame.

    Args:
        results (list): A list of result dictionaries with train and validation metrics.

    Returns:
        pd.DataFrame: A DataFrame containing the combined metrics.
    """
    metrics = []

    for i, result in enumerate(results):
        row = {'iteration': i + 1}  # Add iteration number
        for key in result['env_runners'].keys():
            if any(include_key in key for include_key in RESULT_KEYS_TO_INCLUDE):
                row[f"train/{key}"] = round(result['env_runners'][key], 2)
                row[f"val/{key}"] = round(result['evaluation']['env_runners'].get(key, float('nan')), 2)  # Use `.get()` for safety
        metrics.append(row)

    return pd.DataFrame(metrics)