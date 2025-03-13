import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def replace_with_term(s):
    if "SVM+BCELoss" in s:
        return s.replace("SVM+BCELoss", "SVM")
    elif "TabPFN+BCELoss" in s:
        return s.replace("TabPFN+BCELoss", "TabPFN")
    else:
        return s

def extract_one_log(file_path):
    pattern = r"Iteration (\d+): Accuracy: ([0-9.]+), Precision: ([0-9.]+), Recall: ([0-9.]+), F1 Score: ([0-9.]+)"
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                accuracy = float(match.group(2))
                precision = float(match.group(3))
                recall = float(match.group(4))
                f1_score = float(match.group(5))

                data.append({
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1_score
                })
    return pd.DataFrame(data)

def extract_all_log(log_dir):
    all_data = []

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                file_path = os.path.join(root, file)
                df = extract_one_log(file_path)

                # Extract configuration details from filename
                config = file.replace('.log', '').split('_')
                for col, val in zip(["data", "label", "resample", "dim_red", "dim_num", "loss", "model", "type"], config):
                    df[col] = val

                all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def impute_spec_npv(df):
    """
    Calculates Specificity and Negative Predictive Value (NPV) for each row in the DataFrame.
    The DataFrame must contain the columns: 'Accuracy', 'Precision', 'Recall'.
    
    Args:
    df (pd.DataFrame): DataFrame containing the metrics.
    
    Returns:
    pd.DataFrame: Original DataFrame with added 'Specificity' and 'NPV' columns.
    """
    # Calculate intermediate variables A, B, and C
    df['A'] = 1 / df['Precision'] - 1
    df['B'] = 1 / df['Recall'] - 1
    df['C'] = (df['Accuracy'] * (1 / df['Precision'] + 1 / df['Recall'] - 1) - 1) / (1 - df['Accuracy'])
    
    # Calculate Specificity and NPV
    df['Specificity'] = df['C'] / (df['A'] + df['C'])
    df['NPV'] = df['C'] / (df['B'] + df['C'])
    
    # Drop the intermediate columns A, B, and C
    df.drop(['A', 'B', 'C'], axis=1, inplace=True)
    
    return df

def find_max_avg_accuracy(df, metric='Recall', label_value='13080', type_value='MH'):
    """
    Finds the combination of conditions with the maximal average accuracy.

    :param df: DataFrame containing the data.
    :param label_value: The value of the label to filter by.
    :param type_value: The type value to filter by.
    :return: A DataFrame row representing the combination with the highest average metric.
    """
    # Filter the DataFrame for the specified label and type
    filtered_df = df[(df['label'] == label_value) & (df['type'] == type_value)]

    avg_metric = filtered_df.groupby(['data', 'resample', 'dim_red', 'dim_num', 'loss', 'model']).agg({metric: 'mean'})

    # Find the index of the maximum average accuracy
    max_avg_metric_idx = avg_metric[metric].idxmax()

    return avg_metric.loc[max_avg_metric_idx]

def plot_evaluation_metrics(df, compare_factor, fixed_factors):
    """
    Plots box plots for each evaluation metric.
    
    :param df: DataFrame with all data.
    :param compare_factor: The column name to be used for comparison in box plots.
    :param fixed_factors: Dictionary of factors (column names) and their fixed values.
    """
    # Filter the dataframe based on fixed factors
    for factor, value in fixed_factors.items():
        df = df[df[factor] == value]
    
    # Define metrics to plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Plot box plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=compare_factor, y=metric, data=df)
        plt.title('{} Comparison by {compare_factor}'.format(metric))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('{}_{}.png'.format(compare_factor, metric))

def plot_sorted_metric_for_combinations(df, compare_factors, fixed_factors, metric):
    """
    Plots a sorted box plot for a single evaluation metric across different combinations,
    with the combinations sorted by their mean metric value, grouped by 'model'.
    
    :param df: DataFrame with all data.
    :param compare_factors: List of column names to form different combinations.
    :param fixed_factors: Dictionary of factors (column names) and their fixed values to filter the DataFrame.
    :param metric: The evaluation metric to plot.
    """
    # Filter the DataFrame based on fixed factors
    for factor, value in fixed_factors.items():
        df = df[df[factor] == value]
    
    # Ensure 'model' is not in compare_factors
    if 'model' in compare_factors:
        compare_factors.remove('model')

    # Create a 'Combination' column that concatenates the values of the compare_factors
    df['Combination'] = df[compare_factors].apply(lambda row: '+'.join(row.values.astype(str)), axis=1)
    df['Combination'] = df['Combination'].apply(replace_with_term)
    
    # Calculate mean metric values for each combination and sort
    sorted_combinations = df.groupby('Combination')[metric].mean().sort_values(ascending=False).index
    
    # Sort the DataFrame by the combinations based on the metric's mean values
    df['Combination'] = pd.Categorical(df['Combination'], categories=sorted_combinations, ordered=True)
    df = df.sort_values('Combination')
    
    # Extract unique values of the 'Combination' column for x-axis labels
    unique_combinations = df['Combination'].unique()

    # File name
    fixed_keys = fixed_factors.keys()
    fixed_list = [fixed_factors[key] for key in fixed_keys]
    fixed_name = '_'.join(fixed_list)
    compare_name = '_'.join(compare_factors)
    compare_title = '×'.join([factor.capitalize() for factor in compare_factors])

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Combination', y=metric, hue='model', hue_order=['TabPFN', 'MLP', 'SVM'], data=df, palette='Set2')
    plt.title('{} Comparison (Sorted)'.format(metric))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(compare_title)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.legend(title='Model', loc='upper right')
    plt.savefig('./results/{}/{}-{}.png'.format(metric, fixed_name, compare_name))

log_dir = 'results/records'  # Replace with your log directory

# # Example usage
# combined_df = extract_all_log(log_dir)
# print(find_max_avg_accuracy(combined_df, metric='Recall', label_value='13080', type_value='MH'))
# print(find_max_avg_accuracy(combined_df, metric='Recall', label_value='13080', type_value='WCH'))

# log_MH = impute_spec_npv(extract_one_log('results/records/demographic_13080_Tomek_PCA_6_BCELoss_TabPFN_MH.log'))
# log_WCH = impute_spec_npv(extract_one_log('results/records/demographic_13080_Tomek_PCA_6_BCELoss_TabPFN_WCH.log'))
# print(log_MH.mean())
# print(log_MH.std())
# print(log_WCH.mean())
# print(log_WCH.std())

benchmark_MH = impute_spec_npv(extract_one_log(os.path.join(log_dir, "benchmark_MH.log")))
benchmark_WCH = impute_spec_npv(extract_one_log(os.path.join(log_dir, "benchmark_WCH.log")))
print(benchmark_MH.mean())
print(benchmark_MH.std())
print(benchmark_WCH.mean())
print(benchmark_WCH.std())

# # Example usage
# compare_factor = 'data'  # The factor you want to compare (e.g., 'dim_red', 'model', etc.)
# fixed_factors = {'label': '14090', 'resample': 'None', 'dim_red': 'None', 'loss': 'BCELoss', 'model': 'MLP', 'type': 'WCH'}  # Fixed factors
# plot_evaluation_metrics(combined_df, compare_factor, fixed_factors)

# # Example usage
# compare_factors = ['model', 'loss']  # Adjust based on your data structure
# fixed_factors = {'type': 'MH', 'label': '13080', 'data': 'demographic', 'resample': 'None', 'dim_red': 'None'}  # Example fixed factors
# metric = 'F1 Score'
# plot_sorted_metric_for_combinations(combined_df, compare_factors, fixed_factors, metric)
