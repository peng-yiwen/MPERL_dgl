'''
Created on 2023-11-20
Author: Yiwen Peng

This script uses Bootstrap Resampling for calculating confidence intervals (CI) of the results,
basd on the given rank file.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_confidence_interval(
    stats: np.ndarray,
    seed: int,
    confidence_alpha: float,
    num_iterations: int = 1000,
    ):
    '''
    Calculate the confidence interval of a statistics function.
    Returns:
        The confidence interval for each metric.
    '''

    if not (0.0 < confidence_alpha < 1.0):
        raise ValueError(f"Invalid confidence_alpha: {confidence_alpha}")
    
    _seed = seed if seed is not None else np.random.SeedSequence()
    num_samples = stats.shape[0]
    
    all_indices = np.array(range(num_samples))
    rng = np.random.default_rng(_seed)
    all_indices = rng.choice(
        all_indices, size=(num_iterations, num_samples), replace=True
    )
    # filter out data with counterpart indices and aggregate them
    filtered = stats[all_indices.flatten()]
    filt_stats = filtered.reshape(all_indices.shape + (stats.shape[1],))
    agg_stats = np.mean(filt_stats, axis=-2) # shape: (num_iterations, num_statistics)

    agg_stats.sort(axis=0)
    low = int(num_iterations * confidence_alpha / 2.0)
    high = int(num_iterations * (1.0 - confidence_alpha / 2.0))
    return list(zip(agg_stats[low], agg_stats[high]))


def calc_metric_per_sample(rank_file: str):
    '''
    Calculate the metrics for each sample.
    Returns:
        A numpy array of shape (dataset_length, num_metrics).
    '''
    df = pd.read_csv(rank_file, sep='\t|\s+', header=None, engine='python')
    df.columns = ['entity', 'type', 'rank']
    df['mrr'] = 1.0 / df['rank']
    df['hit@1'] = np.where(df['rank'] <= 1, 1.0, 0.0)
    df['hit@3'] = np.where(df['rank'] <= 3, 1.0, 0.0)
    df['hit@10'] = np.where(df['rank'] <= 10, 1.0, 0.0)
    df = df.dropna()
    # turn to numpy array
    stats = df[['mrr', 'hit@1', 'hit@3', 'hit@10']].astype(float).values
    return stats


# Visualization
def visualize_results(model_names, means, std_devs):
    '''
    Visualize the results with confidence interval.
    '''
    categories = ['MRR', 'Hit@1', 'Hit@3', 'Hit@10']
    bar_width = 0.14
    fig, ax = plt.subplots(figsize=(20, 6))
    bar_positions = np.arange(len(categories))
    for i, model_name in enumerate(model_names):
        ax.bar(bar_positions + i * bar_width, means[:, i], yerr=std_devs[:, i], capsize=5, width=bar_width, label=model_name)
    
    ax.legend()
    ax.set_title('Results with Confidence Interval')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values (%) with CI')
    ax.set_xticks(bar_positions + (len(model_names) - 1) * bar_width / 2)
    ax.set_xticklabels(categories)
    plt.show()



if __name__ == "__main__":
    
    rank_file = 'rank.txt'
    stats = calc_metric_per_sample(rank_file)
    confidence_interval = calc_confidence_interval(stats, seed=0, confidence_alpha=0.05, num_iterations=10000)
    metrics_name = ['MRR', 'HIT@1', 'HIT@3', 'HIT@10']
    for i, metric in enumerate(metrics_name):
        print(f'{metric}: {confidence_interval[i]}')
        