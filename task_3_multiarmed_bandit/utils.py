import numpy as np
import time

from scipy.stats import randint, uniform
import pandas as pd

import matplotlib.pyplot as plt

from sim_lib import simulation


def convert_seconds(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f'{minutes} min {seconds} sec'


def validate_policy(policy, n, seed, verbose=True):
    np.random.seed(seed=seed)

    start = time.time()
    output = simulation(policy, n=n, seed=seed, verbose=verbose)
    end = time.time()

    if verbose:
        print(f'\nValidation took {convert_seconds(end - start)}')

    return output


def eps_greedy(history: pd.DataFrame, eps: float):
    if uniform.rvs() < eps:
        n = history.shape[0]
        return history.index[randint.rvs(0, n)]

    ctr = history['clicks'] / (history['impressions'] + 10)
    n = np.argmax(ctr)
    return history.index[n]


def visualize_comparison(comparison_results, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.title('Regrets per round')

    for result in comparison_results:
        plt.plot(result['data'], label=result['policy_name'])

    if ylim is not None:
        plt.ylim(top=ylim)

    plt.legend()
    plt.show()
