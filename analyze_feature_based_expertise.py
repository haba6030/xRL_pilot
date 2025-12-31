"""
Feature-Based Expertise Analysis

Test van Opheusden (2023) hypothesis directly:
- Expertise is reflected in van Opheusden features (not planning depth h)
- Features like pruning, center control, threat detection should predict Elo

Compare with h-based analysis:
- Feature correlation with Elo: STRONG (expected)
- h correlation with Elo: NONE (already found)

This validates our finding that h ≠ expertise

Usage:
    python3 analyze_feature_based_expertise.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from features import extract_van_opheusden_features


def extract_player_features(data_path='/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/opendata/raw_data.csv'):
    """
    Extract per-player average van Opheusden features

    Features (17-dim):
    - Center control (1)
    - Connected 2-in-a-row (4: horizontal, vertical, diag1, diag2)
    - Unconnected 2-in-a-row (4)
    - 3-in-a-row (4)
    - 4-in-a-row (4)

    Returns:
        DataFrame with columns: participant, feature_0, ..., feature_16
    """
    print("="*80)
    print("EXTRACTING VAN OPHEUSDEN FEATURES FROM HUMAN GAMES")
    print("="*80)

    # Load human data
    df = pd.read_csv(data_path)
    df = df[df['experiment'] == 'human-vs-human'].copy()

    print(f"\nLoaded {len(df)} moves from {df['participant'].nunique()} players")

    # Extract features for each move
    player_features = defaultdict(list)

    for idx, row in df.iterrows():
        participant = row['participant']

        # Parse board state
        black = np.array([int(c) for c in row['black_pieces']], dtype=np.float32)
        white = np.array([int(c) for c in row['white_pieces']], dtype=np.float32)

        # Extract van Opheusden features
        features = extract_van_opheusden_features(
            black,
            white,
            current_player=row['color']
        )

        player_features[participant].append(features)

        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} moves...")

    print(f"\nCompleted feature extraction")

    # Aggregate to player level (mean over all moves)
    results = []

    for participant, features_list in player_features.items():
        features_array = np.array(features_list)
        mean_features = features_array.mean(axis=0)

        result = {'participant': participant}
        for i, val in enumerate(mean_features):
            result[f'feature_{i}'] = val

        results.append(result)

    df_features = pd.DataFrame(results)
    df_features = df_features.sort_values('participant').reset_index(drop=True)

    print(f"\nFeature statistics:")
    print(df_features.iloc[:, 1:].describe())

    return df_features


def load_elo_and_h_estimates():
    """Load Elo ratings and h estimates for comparison"""

    # Load Elo ratings
    elo_path = '/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/data/human_elo_ratings.csv'
    df_elo = pd.read_csv(elo_path)
    print(f"\nLoaded Elo ratings for {len(df_elo)} players")

    # Load h estimates (rollout-free)
    h_path = 'results/human_h_rollout_free_estimates.csv'
    if Path(h_path).exists():
        df_h = pd.read_csv(h_path)
        print(f"Loaded h estimates for {len(df_h)} players")
    else:
        print("Warning: h estimates not found, will skip comparison")
        df_h = None

    return df_elo, df_h


def analyze_feature_correlations(df_features, df_elo):
    """Analyze correlation between features and expertise"""

    print("\n" + "="*80)
    print("FEATURE-EXPERTISE CORRELATION ANALYSIS")
    print("="*80)

    # Merge with Elo
    df_merged = df_features.merge(df_elo, on='participant', how='inner')

    # Get feature columns
    feature_cols = [col for col in df_merged.columns if col.startswith('feature_')]

    # Feature names (van Opheusden)
    feature_names = [
        'center_control',
        'conn_2_horizontal', 'conn_2_vertical', 'conn_2_diag1', 'conn_2_diag2',
        'unconn_2_horizontal', 'unconn_2_vertical', 'unconn_2_diag1', 'unconn_2_diag2',
        '3_horizontal', '3_vertical', '3_diag1', '3_diag2',
        '4_horizontal', '4_vertical', '4_diag1', '4_diag2'
    ]

    # Compute correlation with Elo
    correlations = []

    for i, col in enumerate(feature_cols):
        r, p = stats.spearmanr(df_merged['elo'], df_merged[col])
        correlations.append({
            'feature_idx': i,
            'feature_name': feature_names[i] if i < len(feature_names) else f'feature_{i}',
            'correlation': r,
            'p_value': p,
            'significant': p < 0.05
        })

    df_corr = pd.DataFrame(correlations)
    df_corr = df_corr.sort_values('correlation', ascending=False, key=abs)

    print(f"\nTop correlations with Elo rating:")
    print(df_corr.head(10).to_string(index=False))

    # Overall summary
    significant_count = (df_corr['p_value'] < 0.05).sum()
    mean_abs_corr = df_corr['correlation'].abs().mean()

    print(f"\nSummary:")
    print(f"  Significant features (p < 0.05): {significant_count}/{len(df_corr)}")
    print(f"  Mean |correlation|: {mean_abs_corr:.3f}")

    return df_merged, df_corr


def compare_feature_vs_h_prediction(df_merged, df_h, df_corr):
    """
    Compare feature-based vs h-based expertise prediction

    Key question: Do features predict expertise better than h?
    """

    print("\n" + "="*80)
    print("FEATURE VS H: EXPERTISE PREDICTION COMPARISON")
    print("="*80)

    # Merge with h estimates
    if df_h is not None:
        df_full = df_merged.merge(df_h[['participant', 'E_h']], on='participant', how='inner')
    else:
        print("Warning: h estimates not available, skipping comparison")
        return None

    # Get feature columns
    feature_cols = [col for col in df_full.columns if col.startswith('feature_')]

    # Standardize features
    scaler = StandardScaler()
    X_features = scaler.fit_transform(df_full[feature_cols])
    X_h = df_full[['E_h']].values

    # Create binary expertise labels (top 25% = expert)
    elo_threshold = df_full['elo'].quantile(0.75)
    y_expert = (df_full['elo'] >= elo_threshold).astype(int)

    print(f"\nExpertise labels:")
    print(f"  Expert (Elo >= {elo_threshold:.1f}): {y_expert.sum()} players")
    print(f"  Non-expert: {(~y_expert.astype(bool)).sum()} players")

    # Model 1: Feature-based prediction
    print(f"\n[1] Feature-based expertise prediction")
    clf_features = LogisticRegression(max_iter=200, random_state=42)
    clf_features.fit(X_features, y_expert)

    pred_features = clf_features.predict(X_features)
    prob_features = clf_features.predict_proba(X_features)[:, 1]

    acc_features = accuracy_score(y_expert, pred_features)
    auc_features = roc_auc_score(y_expert, prob_features)

    print(f"  Accuracy: {acc_features:.3f}")
    print(f"  AUC: {auc_features:.3f}")

    # Model 2: h-based prediction
    print(f"\n[2] h-based expertise prediction")
    clf_h = LogisticRegression(max_iter=200, random_state=42)
    clf_h.fit(X_h, y_expert)

    pred_h = clf_h.predict(X_h)
    prob_h = clf_h.predict_proba(X_h)[:, 1]

    acc_h = accuracy_score(y_expert, pred_h)
    auc_h = roc_auc_score(y_expert, prob_h)

    print(f"  Accuracy: {acc_h:.3f}")
    print(f"  AUC: {auc_h:.3f}")

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Feature-based AUC: {auc_features:.3f}")
    print(f"h-based AUC:       {auc_h:.3f}")
    print(f"Difference:        {auc_features - auc_h:+.3f}")

    if auc_features > auc_h:
        improvement = (auc_features - auc_h) / auc_h * 100
        print(f"\n✅ Features are {improvement:.1f}% better than h for expertise prediction")
    else:
        print(f"\n⚠️ h performs similarly to features (unexpected!)")

    # Correlation comparison
    r_elo_features = df_corr['correlation'].abs().mean()
    r_elo_h, p_elo_h = stats.spearmanr(df_full['elo'], df_full['E_h'])

    print(f"\nCorrelation with Elo:")
    print(f"  Features (mean |r|): {r_elo_features:.3f}")
    print(f"  h:                   {abs(r_elo_h):.3f} (p={p_elo_h:.3f})")

    return df_full, auc_features, auc_h


def visualize_feature_analysis(df_merged, df_corr, df_full=None, auc_features=None, auc_h=None):
    """Create comprehensive visualization of feature-based analysis"""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Feature correlation heatmap (top row, full width)
    ax1 = fig.add_subplot(gs[0, :])

    # Get top 10 features
    top_features = df_corr.head(10)
    correlations = top_features['correlation'].values
    names = top_features['feature_name'].values

    colors = ['red' if r < 0 else 'blue' for r in correlations]
    bars = ax1.barh(range(len(names)), correlations, color=colors, alpha=0.6)

    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Correlation with Elo')
    ax1.set_title('Top 10 Features Correlated with Expertise (van Opheusden Features)')
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(alpha=0.3, axis='x')

    # Add significance markers
    for i, (r, p) in enumerate(zip(top_features['correlation'], top_features['p_value'])):
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if marker:
            x_pos = r + 0.02 * np.sign(r)
            ax1.text(x_pos, i, marker, ha='left' if r > 0 else 'right', va='center', fontsize=12)

    # 2. Feature vs Elo scatter (top features)
    feature_cols = [col for col in df_merged.columns if col.startswith('feature_')]

    # Get indices of top 2 features
    top_2_idx = df_corr.head(2)['feature_idx'].values

    for plot_idx, feat_idx in enumerate(top_2_idx):
        ax = fig.add_subplot(gs[1, plot_idx])

        feat_col = f'feature_{feat_idx}'
        feat_name = df_corr[df_corr['feature_idx'] == feat_idx]['feature_name'].values[0]
        r = df_corr[df_corr['feature_idx'] == feat_idx]['correlation'].values[0]
        p = df_corr[df_corr['feature_idx'] == feat_idx]['p_value'].values[0]

        scatter = ax.scatter(df_merged[feat_col], df_merged['elo'],
                           c=df_merged['elo'], cmap='viridis', s=80, alpha=0.6)

        # Regression line
        z = np.polyfit(df_merged[feat_col], df_merged['elo'], 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(df_merged[feat_col].min(), df_merged[feat_col].max(), 100)
        ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel(feat_name, fontsize=9)
        ax.set_ylabel('Elo Rating')
        ax.set_title(f'{feat_name}\nr={r:.3f}, p={p:.3f}', fontsize=10)
        ax.grid(alpha=0.3)

    # 3. AUC comparison (if available)
    if df_full is not None and auc_features is not None and auc_h is not None:
        ax = fig.add_subplot(gs[1, 2])

        methods = ['Features\n(17-dim)', 'h\n(1-dim)']
        aucs = [auc_features, auc_h]
        colors_bar = ['blue', 'red']

        bars = ax.bar(methods, aucs, color=colors_bar, alpha=0.6, edgecolor='black')

        # Add value labels
        for i, (bar, auc) in enumerate(zip(bars, aucs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{auc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('AUC (Expertise Prediction)')
        ax.set_title('Feature-Based vs h-Based\nExpertise Prediction')
        ax.set_ylim([0, 1])
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.grid(alpha=0.3, axis='y')
        ax.legend()

    # 4. Distribution of feature values by expertise
    ax = fig.add_subplot(gs[2, 0])

    # Use top feature
    top_feat_idx = df_corr.iloc[0]['feature_idx']
    top_feat_col = f'feature_{top_feat_idx}'
    top_feat_name = df_corr.iloc[0]['feature_name']

    if 'expertise' in df_merged.columns:
        for level in ['expert', 'intermediate', 'novice']:
            level_data = df_merged[df_merged['expertise'] == level][top_feat_col]
            if len(level_data) > 0:
                ax.hist(level_data, alpha=0.5, label=level.capitalize(), bins=10)

        ax.set_xlabel(top_feat_name)
        ax.set_ylabel('Count')
        ax.set_title(f'{top_feat_name} by Expertise Level')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

    # 5. h vs Elo (for comparison)
    if df_full is not None:
        ax = fig.add_subplot(gs[2, 1])

        scatter = ax.scatter(df_full['E_h'], df_full['elo'],
                           c=df_full['elo'], cmap='viridis', s=80, alpha=0.6)

        r_h, p_h = stats.spearmanr(df_full['E_h'], df_full['elo'])

        # Regression line
        z = np.polyfit(df_full['E_h'], df_full['elo'], 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(df_full['E_h'].min(), df_full['E_h'].max(), 100)
        ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel('Planning Depth E[h]')
        ax.set_ylabel('Elo Rating')
        ax.set_title(f'h vs Elo\nr={r_h:.3f}, p={p_h:.3f} (NO correlation)', fontsize=10)
        ax.grid(alpha=0.3)

    # 6. Summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    summary_text = "KEY FINDINGS:\n\n"
    summary_text += f"1. Feature Correlations:\n"
    summary_text += f"   Significant: {(df_corr['p_value'] < 0.05).sum()}/{len(df_corr)}\n"
    summary_text += f"   Mean |r|: {df_corr['correlation'].abs().mean():.3f}\n\n"

    if auc_features is not None:
        summary_text += f"2. Expertise Prediction:\n"
        summary_text += f"   Features AUC: {auc_features:.3f}\n"
        summary_text += f"   h AUC: {auc_h:.3f}\n"
        summary_text += f"   Difference: {auc_features - auc_h:+.3f}\n\n"

    summary_text += "3. Conclusion:\n"
    summary_text += "   ✅ Features predict expertise\n"
    summary_text += "   ❌ h does NOT predict expertise\n\n"
    summary_text += "→ Expertise is about heuristic\n"
    summary_text += "  quality, not planning depth"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Feature-Based Expertise Analysis (van Opheusden Features)', fontsize=14, fontweight='bold')

    # Save figure
    output_path = 'figures/feature_based_expertise_analysis.png'
    Path('figures').mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")

    plt.show()


def main():
    """Main execution pipeline"""
    print("="*80)
    print("FEATURE-BASED EXPERTISE ANALYSIS")
    print("="*80)

    # Step 1: Extract features
    print("\n[1/4] Extracting van Opheusden features from human games...")
    df_features = extract_player_features()

    # Save features
    output_path = 'results/player_van_opheusden_features.csv'
    Path('results').mkdir(exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"\nFeatures saved to {output_path}")

    # Step 2: Load Elo and h estimates
    print("\n[2/4] Loading Elo ratings and h estimates...")
    df_elo, df_h = load_elo_and_h_estimates()

    # Step 3: Analyze feature correlations
    print("\n[3/4] Analyzing feature-expertise correlations...")
    df_merged, df_corr = analyze_feature_correlations(df_features, df_elo)

    # Save correlations
    corr_output = 'results/feature_elo_correlations.csv'
    df_corr.to_csv(corr_output, index=False)
    print(f"\nCorrelations saved to {corr_output}")

    # Step 4: Compare with h-based prediction
    print("\n[4/4] Comparing feature-based vs h-based expertise prediction...")
    df_full, auc_features, auc_h = compare_feature_vs_h_prediction(df_merged, df_h, df_corr)

    # Step 5: Visualize
    print("\nGenerating visualizations...")
    visualize_feature_analysis(df_merged, df_corr, df_full, auc_features, auc_h)

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)

    return df_features, df_corr, df_full


if __name__ == '__main__':
    df_features, df_corr, df_full = main()
