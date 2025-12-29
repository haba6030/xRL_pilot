"""
Rollout-Free Posterior Estimation of Planning Depth (h)

KEY INNOVATION: No rollout simulation during inference
- Training: Used real human s_{t+h} from data
- Inference: Also uses real human s_{t+h} from data
→ Eliminates distribution mismatch artifact

Method:
1. For each move t in human games:
   - Extract actual future states: s_{t+1}, s_{t+2}, s_{t+3}, s_{t+4}
   - Compute likelihood for each h: ℓ_h(t) = P(a_t | s_t, s_{t+h}^human)
   - Apply Bayes rule: P(h|t) ∝ ℓ_h(t) * P(h)

2. Per-player aggregation:
   - P̄_i(h) = mean over all moves by player i
   - E[h]_i = Σ h * P̄_i(h)

3. Compare with Elo ratings:
   - If paradox resolves: Random rollout was artifact
   - If paradox persists: Experts genuinely plan differently

Usage:
    python3 estimate_player_h_rollout_free.py
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler


def load_h_models(h_values=[1, 2, 3, 4]):
    """Load trained h-specific inverse kinematics models"""
    models = {}

    for h in h_values:
        model_path = f'models/separate_h/model_h{h}.pkl'
        data = joblib.load(model_path)
        models[h] = data['model']  # Extract the actual model from the dict
        print(f"Loaded model h={h} (val_acc={data['val_acc']:.3f})")

    return models


def load_human_games(data_path='/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/opendata/raw_data.csv'):
    """Load human game data"""
    df = pd.read_csv(data_path)

    # Filter for human vs human games
    df = df[df['experiment'] == 'human-vs-human'].copy()

    print(f"Loaded {len(df)} moves from human games")
    print(f"Unique participants: {df['participant'].nunique()}")

    return df


def parse_board_state(black_str, white_str, current_player='Black'):
    """
    Convert board strings to state vector (89-dim)

    Format: [black (36), white (36), van_opheusden_features (17)] = 89

    This matches the format used in multistep IK training.
    """
    try:
        from features import extract_van_opheusden_features
    except ImportError:
        # Fallback: use dummy features if not available
        features = np.zeros(17, dtype=np.float32)
        black_pieces = np.array([int(c) for c in black_str], dtype=np.float32)
        white_pieces = np.array([int(c) for c in white_str], dtype=np.float32)
        return np.concatenate([black_pieces, white_pieces, features])

    # Parse strings to numpy arrays
    black_pieces = np.array([int(c) for c in black_str], dtype=np.float32)
    white_pieces = np.array([int(c) for c in white_str], dtype=np.float32)

    # Extract van Opheusden features (same as in training)
    features = extract_van_opheusden_features(
        black_pieces,
        white_pieces,
        current_player
    )

    # Create observation: [black (36), white (36), features (17)] = 89
    state = np.concatenate([black_pieces, white_pieces, features]).astype(np.float32)

    return state


def detect_games_and_extract_sequences(df):
    """
    Detect individual games and extract move sequences with future states

    Returns:
        List of games, where each game is a list of dicts:
        {
            'participant': player ID,
            'state_t': current state (72-dim),
            'action_t': current action,
            'state_t1': state after 1 move (if exists),
            'state_t2': state after 2 moves (if exists),
            'state_t3': state after 3 moves (if exists),
            'state_t4': state after 4 moves (if exists),
        }
    """
    games = []
    current_game = []

    # Add piece count for game detection
    df['num_pieces'] = df['black_pieces'].apply(lambda x: x.count('1')) + \
                       df['white_pieces'].apply(lambda x: x.count('1'))

    prev_pieces = 0

    for idx, row in df.iterrows():
        num_pieces = row['num_pieces']

        # New game detected (piece count reset)
        if num_pieces <= prev_pieces and len(current_game) > 0:
            if len(current_game) >= 5:  # Need at least 5 moves for h=4
                games.append(current_game)
            current_game = []

        # Parse state (with current player info for features)
        state = parse_board_state(
            row['black_pieces'],
            row['white_pieces'],
            current_player=row['color']
        )

        # Store move
        current_game.append({
            'participant': row['participant'],
            'state': state,
            'action': row['move'],
        })

        prev_pieces = num_pieces

    # Add last game
    if len(current_game) >= 5:
        games.append(current_game)

    print(f"\nDetected {len(games)} games")
    print(f"Average game length: {np.mean([len(g) for g in games]):.1f} moves")

    # Now add future states to each move
    processed_games = []

    for game in games:
        processed_game = []

        for t in range(len(game)):
            move_data = {
                'participant': game[t]['participant'],
                'state_t': game[t]['state'],
                'action_t': game[t]['action'],
            }

            # Add future states (if they exist)
            for h in range(1, 5):
                if t + h < len(game):
                    move_data[f'state_t{h}'] = game[t + h]['state']
                else:
                    move_data[f'state_t{h}'] = None  # Game ended

            # Only include moves where we have at least h=1 future
            if move_data['state_t1'] is not None:
                processed_game.append(move_data)

        if len(processed_game) > 0:
            processed_games.append(processed_game)

    print(f"Processed {len(processed_games)} games with future states")

    return processed_games


def compute_move_likelihood(model, state_current, state_future, action):
    """
    Compute P(action | state_current, state_future) using trained h-model

    Returns:
        log_prob: log probability of the action
    """
    # Concatenate states as model input
    X = np.concatenate([state_current, state_future]).reshape(1, -1)

    # Get probability distribution over actions
    probs = model.predict_proba(X)[0]  # Shape: (36,)

    # Get probability of the actual action
    prob = probs[action]

    # Return log probability (for numerical stability)
    log_prob = np.log(prob + 1e-10)  # Add small epsilon to avoid log(0)

    return log_prob


def compute_rollout_free_posterior(games, models, h_values=[1, 2, 3, 4], prior=None):
    """
    Compute rollout-free posterior P(h|move) for all moves

    Args:
        games: List of games with move sequences
        models: Dict of h-specific models
        h_values: List of h values to consider
        prior: Prior P(h). If None, use uniform

    Returns:
        player_moves: Dict mapping participant -> list of move-level posteriors
    """
    if prior is None:
        prior = np.ones(len(h_values)) / len(h_values)  # Uniform prior

    player_moves = defaultdict(list)

    total_moves = 0
    valid_moves_per_h = {h: 0 for h in h_values}

    for game in games:
        for move_data in game:
            participant = move_data['participant']
            state_t = move_data['state_t']
            action_t = move_data['action_t']

            # Compute likelihood for each h
            log_likelihoods = []
            valid_h = []

            for i, h in enumerate(h_values):
                state_future = move_data.get(f'state_t{h}')

                if state_future is not None:
                    # Compute log P(a_t | s_t, s_{t+h})
                    log_lik = compute_move_likelihood(
                        models[h],
                        state_t,
                        state_future,
                        action_t
                    )
                    log_likelihoods.append(log_lik)
                    valid_h.append(i)
                    valid_moves_per_h[h] += 1

            if len(log_likelihoods) == 0:
                continue  # No valid future states

            # Convert to numpy arrays
            log_likelihoods = np.array(log_likelihoods)

            # Compute posterior using log-space for numerical stability
            # log P(h|t) = log P(a_t|s_t,s_{t+h}) + log P(h) - log Z
            log_prior = np.log(prior[valid_h])
            log_posterior_unnorm = log_likelihoods + log_prior

            # Normalize using log-sum-exp trick
            log_Z = np.logaddexp.reduce(log_posterior_unnorm)
            log_posterior = log_posterior_unnorm - log_Z
            posterior = np.exp(log_posterior)

            # Create full posterior (with zeros for invalid h)
            full_posterior = np.zeros(len(h_values))
            full_posterior[valid_h] = posterior

            # Store
            player_moves[participant].append({
                'posterior': full_posterior,
                'valid_h': [h_values[i] for i in valid_h],
                'log_likelihoods': log_likelihoods,
            })

            total_moves += 1

    print(f"\nProcessed {total_moves} total moves")
    print(f"Valid moves per h: {valid_moves_per_h}")

    return player_moves


def aggregate_player_posteriors(player_moves, h_values=[1, 2, 3, 4]):
    """
    Aggregate move-level posteriors to player-level estimates

    Returns:
        DataFrame with columns: participant, E[h], P(h=1), P(h=2), P(h=3), P(h=4), num_moves
    """
    results = []

    for participant, moves in player_moves.items():
        # Average posterior over all moves
        posteriors = np.array([m['posterior'] for m in moves])
        mean_posterior = posteriors.mean(axis=0)

        # Compute expected h
        E_h = np.sum(h_values * mean_posterior)

        # Store results
        result = {
            'participant': participant,
            'E_h': E_h,
            'num_moves': len(moves),
        }

        for i, h in enumerate(h_values):
            result[f'P_h{h}'] = mean_posterior[i]

        results.append(result)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('participant')

    return df_results


def load_elo_ratings():
    """Load Elo ratings from previous analysis"""
    elo_path = '/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/data/human_elo_ratings.csv'

    if Path(elo_path).exists():
        df_elo = pd.read_csv(elo_path)
        print(f"\nLoaded Elo ratings for {len(df_elo)} players")
        return df_elo
    else:
        print("\nWarning: Elo ratings file not found")
        return None


def analyze_expertise_correlation(df_results, df_elo):
    """Analyze correlation between E[h] and expertise"""
    # Merge with Elo
    df_merged = df_results.merge(df_elo, on='participant', how='inner')

    print(f"\n{'='*60}")
    print("EXPERTISE ANALYSIS (Rollout-Free Posterior)")
    print(f"{'='*60}")

    # Overall statistics
    print(f"\nOverall E[h] statistics:")
    print(f"  Mean: {df_merged['E_h'].mean():.3f} ± {df_merged['E_h'].std():.3f}")
    print(f"  Range: [{df_merged['E_h'].min():.3f}, {df_merged['E_h'].max():.3f}]")

    # Correlation with Elo
    r_elo, p_elo = stats.spearmanr(df_merged['elo'], df_merged['E_h'])
    print(f"\nCorrelation with Elo rating:")
    print(f"  Spearman r = {r_elo:.3f}, p = {p_elo:.3f}")

    # Correlation with win rate
    if 'win_rate' in df_merged.columns:
        r_win, p_win = stats.spearmanr(df_merged['win_rate'], df_merged['E_h'])
        print(f"\nCorrelation with win rate:")
        print(f"  Spearman r = {r_win:.3f}, p = {p_win:.3f}")

    # Group by expertise level
    if 'expertise' in df_merged.columns:
        print(f"\nE[h] by expertise level:")
        for level in ['expert', 'intermediate', 'novice']:
            level_data = df_merged[df_merged['expertise'] == level]['E_h']
            if len(level_data) > 0:
                print(f"  {level.capitalize()}: {level_data.mean():.3f} ± {level_data.std():.3f} (n={len(level_data)})")

        # Statistical test: expert vs novice
        expert_h = df_merged[df_merged['expertise'] == 'expert']['E_h']
        novice_h = df_merged[df_merged['expertise'] == 'novice']['E_h']

        if len(expert_h) > 0 and len(novice_h) > 0:
            u_stat, p_val = stats.mannwhitneyu(expert_h, novice_h, alternative='two-sided')
            cohens_d = (expert_h.mean() - novice_h.mean()) / np.sqrt((expert_h.std()**2 + novice_h.std()**2) / 2)

            print(f"\nExpert vs Novice comparison:")
            print(f"  Mann-Whitney U = {u_stat:.1f}, p = {p_val:.3f}")
            print(f"  Cohen's d = {cohens_d:.3f}")

    return df_merged


def visualize_results(df_merged, h_values=[1, 2, 3, 4]):
    """Create comprehensive visualization of rollout-free results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. E[h] distribution
    ax = axes[0, 0]
    ax.hist(df_merged['E_h'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(df_merged['E_h'].mean(), color='red', linestyle='--',
               label=f"Mean = {df_merged['E_h'].mean():.2f}")
    ax.set_xlabel('Expected Planning Depth E[h]')
    ax.set_ylabel('Number of Players')
    ax.set_title('Distribution of E[h] (Rollout-Free)')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. E[h] vs Elo
    ax = axes[0, 1]
    scatter = ax.scatter(df_merged['elo'], df_merged['E_h'],
                        c=df_merged['E_h'], cmap='viridis', s=100, alpha=0.6)

    # Add regression line
    z = np.polyfit(df_merged['elo'], df_merged['E_h'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_merged['elo'].min(), df_merged['elo'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    r, p_val = stats.spearmanr(df_merged['elo'], df_merged['E_h'])
    ax.set_xlabel('Elo Rating')
    ax.set_ylabel('Expected Planning Depth E[h]')
    ax.set_title(f'E[h] vs Elo (r={r:.3f}, p={p_val:.3f})')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='E[h]')

    # 3. Posterior distribution (aggregate)
    ax = axes[1, 0]
    mean_posteriors = [df_merged[f'P_h{h}'].mean() for h in h_values]
    std_posteriors = [df_merged[f'P_h{h}'].std() for h in h_values]

    ax.bar(h_values, mean_posteriors, yerr=std_posteriors,
           capsize=5, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Planning Depth h')
    ax.set_ylabel('Average Probability P(h)')
    ax.set_title('Aggregate Posterior Distribution (Rollout-Free)')
    ax.set_xticks(h_values)
    ax.grid(alpha=0.3, axis='y')

    # Add percentage labels
    for i, h in enumerate(h_values):
        ax.text(h, mean_posteriors[i] + std_posteriors[i] + 0.01,
                f'{mean_posteriors[i]*100:.1f}%', ha='center', fontsize=9)

    # 4. Box plot by expertise level
    ax = axes[1, 1]
    if 'expertise' in df_merged.columns:
        expertise_order = ['expert', 'intermediate', 'novice']
        data_to_plot = [df_merged[df_merged['expertise'] == level]['E_h'].values
                        for level in expertise_order if level in df_merged['expertise'].values]
        labels = [level.capitalize() for level in expertise_order
                 if level in df_merged['expertise'].values]

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Expected Planning Depth E[h]')
        ax.set_title('E[h] by Expertise Level (Rollout-Free)')
        ax.grid(alpha=0.3, axis='y')

        # Add n annotations
        for i, level in enumerate(labels):
            n = len(data_to_plot[i])
            ax.text(i+1, ax.get_ylim()[0], f'n={n}', ha='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Expertise labels not available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('E[h] by Expertise Level')

    plt.tight_layout()

    # Save figure
    output_path = 'figures/rollout_free_posterior_results.png'
    Path('figures').mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")

    plt.show()


def main():
    """Main execution pipeline"""
    print("="*60)
    print("ROLLOUT-FREE POSTERIOR ESTIMATION")
    print("="*60)

    # 1. Load models
    print("\n[1/6] Loading h-specific models...")
    models = load_h_models(h_values=[1, 2, 3, 4])

    # 2. Load human data
    print("\n[2/6] Loading human game data...")
    df_human = load_human_games()

    # 3. Extract game sequences with future states
    print("\n[3/6] Extracting game sequences with future states...")
    games = detect_games_and_extract_sequences(df_human)

    # 4. Compute rollout-free posteriors
    print("\n[4/6] Computing rollout-free posteriors...")
    player_moves = compute_rollout_free_posterior(games, models, h_values=[1, 2, 3, 4])

    # 5. Aggregate to player level
    print("\n[5/6] Aggregating to player-level estimates...")
    df_results = aggregate_player_posteriors(player_moves, h_values=[1, 2, 3, 4])

    print(f"\nEstimated E[h] for {len(df_results)} players")
    print(df_results.head(10))

    # Save results
    output_path = 'results/human_h_rollout_free_estimates.csv'
    Path('results').mkdir(exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # 6. Analyze correlation with expertise
    print("\n[6/6] Analyzing correlation with expertise...")
    df_elo = load_elo_ratings()

    if df_elo is not None:
        df_merged = analyze_expertise_correlation(df_results, df_elo)
        visualize_results(df_merged)
    else:
        print("Skipping expertise analysis (Elo data not found)")

    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)

    return df_results


if __name__ == '__main__':
    df_results = main()
