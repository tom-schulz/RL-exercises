import numpy as np
import matplotlib.pyplot as plt
from rliable import metrics
from rliable import plot_utils
from rliable import library as rly

def main():
    # 1. Load the data from the 5 runs
    seeds = [1, 2, 3, 4, 5]
    all_runs = []
    
    for s in seeds:
        try:
            data = np.load(f"dqn_seed_{s}.npy")
            all_runs.append(data)
        except FileNotFoundError:
            print(f"Could not find dqn_seed_{s}.npy! Make sure you ran all 5 seeds.")
            return

    # 2. Fix uneven lengths
    # Depending on how long CartPole stayed up, runs might have different episode counts.
    # We truncate all runs to the length of the shortest run to stack them neatly.
    min_length = min(len(run) for run in all_runs)
    trimmed_runs = [run[:min_length] for run in all_runs]
    
    # Shape: (Number of Runs, Number of Episodes)
    stacked_data = np.array(trimmed_runs)

    # ---------------------------------------------------------
    # PLOT 1: The Standard Training Curve (Mean + Shaded Std Dev)
    # ---------------------------------------------------------
    mean_rewards = np.mean(stacked_data, axis=0)
    std_rewards = np.std(stacked_data, axis=0)
    x_axis = np.arange(min_length)

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, mean_rewards, label='DQN Mean Reward', color='blue')
    plt.fill_between(x_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.2)
    plt.title('DQN CartPole Training Curve (Averaged over 5 seeds)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('dqn_training_curve_seeds.png')
    print("Saved training curve to dqn_training_curve_seeds.png")

    # ---------------------------------------------------------
    # PLOT 2: RLiable Aggregate Metrics (IQM, Median, Mean, Gap)
    # ---------------------------------------------------------
    # RLiable expects a dictionary of shape (num_runs, num_tasks). 
    # We only have 1 task (CartPole), so we take the final 100 episodes of each run as our "final score"
    final_scores = np.mean(stacked_data[:, -100:], axis=1).reshape(-1, 1)
    
    # Normalize scores (CartPole max is usually 500)
    normalized_scores = final_scores / 500.0
    
    score_dict = {"DQN": normalized_scores}
    
    print("\nCalculating RLiable Metrics with Stratified Bootstrap...")

    # 1. Create a function that calculates all 4 metrics on an array
    def aggregate_func(scores):
        return np.array([
            metrics.aggregate_mean(scores),
            metrics.aggregate_median(scores),
            metrics.aggregate_iqm(scores),
            metrics.aggregate_optimality_gap(scores)
        ])

    # 2. Use RLiable's official wrapper to handle the dictionary and 2000 bootstraps
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=2000
    )

    # 3. Generate the plot
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores, 
        aggregate_score_cis,
        metric_names=['Mean', 'Median', 'IQM', 'Optimality Gap'],
        algorithms=['DQN'],
        xlabel='Normalized Score'
    )
    
    plt.savefig('dqn_rliable_metrics.png', bbox_inches='tight')
    print("Saved RLiable metrics to dqn_rliable_metrics.png")
    plt.show()

if __name__ == "__main__":
    main()