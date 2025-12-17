"""
즉시 가능한 분석: PV Depth, Expertise Discrimination, RT 상관관계
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 80)
print("즉시 가능한 분석 실행")
print("=" * 80)

# Load participant data with expertise labels
participant_data = pd.read_csv('analysis_participant_with_expertise.csv', index_col=0)

print("\n[1] Planning Depth (PV Depth) 분석")
print("-" * 80)

# Load depth data
depth_file = Path("xRL_pilot/Analysis notebooks/old/depth_by_session.txt")
if depth_file.exists():
    # Load depth matrix (participants x sessions)
    depth_matrix = np.loadtxt(depth_file)
    print(f"✓ Depth 데이터 로딩: {depth_matrix.shape}")

    # Calculate mean depth per participant
    mean_depth_per_participant = depth_matrix.mean(axis=1)
    std_depth_per_participant = depth_matrix.std(axis=1)

    print(f"\n평균 Planning Depth 통계:")
    print(f"  평균: {mean_depth_per_participant.mean():.3f}")
    print(f"  표준편차: {mean_depth_per_participant.std():.3f}")
    print(f"  범위: [{mean_depth_per_participant.min():.3f}, {mean_depth_per_participant.max():.3f}]")

    # Add to participant data
    # Assuming first 30 participants match
    if len(mean_depth_per_participant) <= len(participant_data):
        participant_data.loc[participant_data.index[:len(mean_depth_per_participant)], 'pv_depth_mean'] = mean_depth_per_participant
        participant_data.loc[participant_data.index[:len(mean_depth_per_participant)], 'pv_depth_std'] = std_depth_per_participant

        # Drop participants without depth data
        participant_with_depth = participant_data.dropna(subset=['pv_depth_mean'])
        print(f"\nDepth 데이터가 있는 참가자: {len(participant_with_depth)}명")

        # Test: PV depth vs expertise
        expert_depth = participant_with_depth[participant_with_depth['expertise_category'] == 'Expert']['pv_depth_mean']
        novice_depth = participant_with_depth[participant_with_depth['expertise_category'] == 'Novice']['pv_depth_mean']

        print(f"\nExpertise별 Planning Depth:")
        print(f"  Expert (n={len(expert_depth)}): {expert_depth.mean():.3f} ± {expert_depth.std():.3f}")
        print(f"  Novice (n={len(novice_depth)}): {novice_depth.mean():.3f} ± {novice_depth.std():.3f}")

        # Statistical test
        t_stat, p_value = stats.ttest_ind(expert_depth, novice_depth)
        print(f"\n  t-test: t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((expert_depth.std()**2 + novice_depth.std()**2) / 2)
        cohens_d = (expert_depth.mean() - novice_depth.mean()) / pooled_std
        print(f"  Cohen's d: {cohens_d:.3f}")

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Distribution
        axes[0].hist(expert_depth, alpha=0.6, label='Expert', bins=10, edgecolor='black')
        axes[0].hist(novice_depth, alpha=0.6, label='Novice', bins=10, edgecolor='black')
        axes[0].set_xlabel('Planning Depth', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('PV Depth Distribution by Expertise', fontsize=14, fontweight='bold')
        axes[0].legend()

        # 2. Box plot
        bp = axes[1].boxplot([novice_depth, expert_depth], labels=['Novice', 'Expert'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightgreen')
        axes[1].set_ylabel('Planning Depth', fontsize=12)
        axes[1].set_title(f'PV Depth: Novice vs Expert\n(p={p_value:.4f})', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        # 3. Correlation with log-likelihood
        axes[2].scatter(participant_with_depth['pv_depth_mean'],
                       participant_with_depth['log-likelihood'],
                       c=participant_with_depth['expertise_label'],
                       cmap='RdYlGn', s=100, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Planning Depth', fontsize=12)
        axes[2].set_ylabel('Log-Likelihood', fontsize=12)
        axes[2].set_title('PV Depth vs Model Fit', fontsize=14, fontweight='bold')

        # Add correlation
        corr, corr_p = stats.pearsonr(participant_with_depth['pv_depth_mean'],
                                       participant_with_depth['log-likelihood'])
        axes[2].text(0.05, 0.95, f'r={corr:.3f}, p={corr_p:.4f}',
                    transform=axes[2].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('immediate_pv_depth_analysis.png', dpi=150, bbox_inches='tight')
        print("\n✓ PV Depth 분석 플롯 저장: immediate_pv_depth_analysis.png")
        plt.close()

    else:
        print(f"⚠ Depth 데이터 크기 불일치: {len(mean_depth_per_participant)} vs {len(participant_data)}")
        participant_with_depth = participant_data.copy()

else:
    print("✗ Depth 파일 없음 - C++ 코드 실행 필요")
    participant_with_depth = participant_data.copy()

print("\n[2] Expertise Discrimination Test")
print("-" * 80)

# Features for discrimination
feature_sets = {
    'Baseline (Parameters only)': ['pruning threshold', 'lapse rate', 'log-likelihood'],
    'Behavioral (RT only)': ['mean', 'median', 'std'],
    'Combined (Parameters + RT)': ['pruning threshold', 'lapse rate', 'log-likelihood', 'mean', 'median'],
}

# Add PV depth if available
if 'pv_depth_mean' in participant_with_depth.columns:
    feature_sets['With PV Depth'] = ['pruning threshold', 'lapse rate', 'log-likelihood', 'pv_depth_mean']
    feature_sets['Full Model'] = ['pruning threshold', 'lapse rate', 'log-likelihood',
                                   'mean', 'pv_depth_mean']

results = {}

for model_name, features in feature_sets.items():
    # Prepare data
    X = participant_with_depth[features].dropna()
    y = participant_with_depth.loc[X.index, 'expertise_label']

    if len(X) < 10:
        print(f"\n⚠ {model_name}: 데이터 부족 (n={len(X)})")
        continue

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    auc = roc_auc_score(y, y_prob)
    accuracy = (y_pred == y).mean()

    results[model_name] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Features': features,
        'Coefficients': dict(zip(features, model.coef_[0])),
        'n_samples': len(X)
    }

    print(f"\n{model_name}:")
    print(f"  샘플 수: {len(X)}")
    print(f"  AUC: {auc:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Feature 중요도:")
    for feat, coef in sorted(zip(features, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:30s}: {coef:+.3f}")

# Visualize discrimination results
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. AUC comparison
ax = axes[0, 0]
model_names = list(results.keys())
aucs = [results[m]['AUC'] for m in model_names]
accuracies = [results[m]['Accuracy'] for m in model_names]

x_pos = np.arange(len(model_names))
ax.bar(x_pos, aucs, alpha=0.7, label='AUC', color='steelblue', edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Expertise Discrimination: Model Comparison', fontsize=14, fontweight='bold')
ax.axhline(0.5, color='red', linestyle='--', label='Chance level')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# 2. Feature importance (best model)
ax = axes[0, 1]
best_model = max(results.keys(), key=lambda k: results[k]['AUC'])
coefs = results[best_model]['Coefficients']
features = list(coefs.keys())
values = list(coefs.values())

colors = ['green' if v > 0 else 'red' for v in values]
ax.barh(features, values, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Coefficient (Expert direction →)', fontsize=12)
ax.set_title(f'Feature Importance: {best_model}\n(AUC={results[best_model]["AUC"]:.3f})',
             fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(axis='x', alpha=0.3)

# 3. Confusion matrix (best model)
ax = axes[1, 0]
if best_model in results:
    X = participant_with_depth[results[best_model]['Features']].dropna()
    y = participant_with_depth.loc[X.index, 'expertise_label']
    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression(random_state=42, max_iter=1000).fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Novice', 'Expert'], yticklabels=['Novice', 'Expert'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix: {best_model}', fontsize=14, fontweight='bold')

# 4. ROC curve comparison
ax = axes[1, 1]
for model_name, features in feature_sets.items():
    X = participant_with_depth[features].dropna()
    y = participant_with_depth.loc[X.index, 'expertise_label']

    if len(X) < 10:
        continue

    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression(random_state=42, max_iter=1000).fit(X_scaled, y)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Calculate ROC curve points
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = results[model_name]['AUC']

    ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.2f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Chance')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('immediate_expertise_discrimination.png', dpi=150, bbox_inches='tight')
print("\n✓ Expertise Discrimination 플롯 저장: immediate_expertise_discrimination.png")
plt.close()

print("\n[3] 응답 시간과 파라미터 상관관계")
print("-" * 80)

# Correlation matrix
rt_param_cols = ['mean', 'median', 'std', 'pruning threshold', 'lapse rate',
                 'log-likelihood', 'feature drop rate', 'center weight']

if 'pv_depth_mean' in participant_data.columns:
    rt_param_cols.append('pv_depth_mean')

corr_data = participant_data[rt_param_cols].dropna()
corr_matrix = corr_data.corr()

print("\n주요 상관관계:")
print("\n1. Response Time (mean) 상관:")
rt_corrs = corr_matrix['mean'].sort_values(ascending=False)
for param, corr in rt_corrs.items():
    if param != 'mean':
        print(f"  {param:30s}: r={corr:+.3f}")

print("\n2. Planning Depth 상관 (if available):")
if 'pv_depth_mean' in corr_matrix.columns:
    pv_corrs = corr_matrix['pv_depth_mean'].sort_values(ascending=False)
    for param, corr in pv_corrs.items():
        if param != 'pv_depth_mean':
            print(f"  {param:30s}: r={corr:+.3f}")

print("\n3. Expertise Score 상관:")
if 'expertise_score' in participant_data.columns:
    for param in rt_param_cols:
        if param in participant_data.columns:
            corr, p = stats.pearsonr(participant_data[param].dropna(),
                                     participant_data.loc[participant_data[param].dropna().index, 'expertise_score'])
            print(f"  {param:30s}: r={corr:+.3f}, p={p:.4f}")

# Visualize correlations
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Correlation heatmap
ax = axes[0, 0]
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title('RT-Parameter Correlation Matrix', fontsize=14, fontweight='bold')

# 2. RT vs Planning Depth
ax = axes[0, 1]
if 'pv_depth_mean' in participant_data.columns:
    data_for_plot = participant_data.dropna(subset=['mean', 'pv_depth_mean'])
    colors = data_for_plot['expertise_label'].map({0: 'red', 1: 'green'})
    ax.scatter(data_for_plot['mean'], data_for_plot['pv_depth_mean'],
              c=colors, s=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mean Response Time (s)', fontsize=12)
    ax.set_ylabel('Planning Depth', fontsize=12)
    ax.set_title('Response Time vs Planning Depth', fontsize=14, fontweight='bold')

    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(data_for_plot['mean'],
                                                               data_for_plot['pv_depth_mean'])
    x_line = np.array([data_for_plot['mean'].min(), data_for_plot['mean'].max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'b--', linewidth=2,
           label=f'r={r_value:.3f}, p={p_value:.4f}')
    ax.legend()

    # Add expert/novice labels
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Novice'),
                      Patch(facecolor='green', label='Expert')]
    ax.legend(handles=legend_elements, loc='best')
else:
    ax.text(0.5, 0.5, 'PV Depth data not available',
           ha='center', va='center', transform=ax.transAxes, fontsize=14)

# 3. RT vs Log-Likelihood
ax = axes[1, 0]
colors = participant_data['expertise_label'].map({0: 'red', 1: 'green'})
ax.scatter(participant_data['mean'], participant_data['log-likelihood'],
          c=colors, s=100, edgecolor='black', alpha=0.7)
ax.set_xlabel('Mean Response Time (s)', fontsize=12)
ax.set_ylabel('Log-Likelihood', fontsize=12)
ax.set_title('Response Time vs Model Fit', fontsize=14, fontweight='bold')

corr, p = stats.pearsonr(participant_data['mean'], participant_data['log-likelihood'])
ax.text(0.05, 0.95, f'r={corr:.3f}, p={p:.4f}',
       transform=ax.transAxes, va='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Expertise breakdown
ax = axes[1, 1]
expert_data = participant_data[participant_data['expertise_category'] == 'Expert']
novice_data = participant_data[participant_data['expertise_category'] == 'Novice']

metrics = ['mean', 'log-likelihood']
if 'pv_depth_mean' in participant_data.columns:
    metrics.append('pv_depth_mean')

x_pos = np.arange(len(metrics))
width = 0.35

expert_means = [expert_data[m].mean() for m in metrics]
novice_means = [novice_data[m].mean() for m in metrics]
expert_stds = [expert_data[m].std() for m in metrics]
novice_stds = [novice_data[m].std() for m in metrics]

# Normalize to 0-1 for comparison
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
all_vals = np.concatenate([expert_means, novice_means]).reshape(-1, 1)
normalized = scaler.fit_transform(all_vals)
expert_norm = normalized[:len(metrics)].flatten()
novice_norm = normalized[len(metrics):].flatten()

ax.bar(x_pos - width/2, expert_norm, width, label='Expert',
      color='green', alpha=0.7, edgecolor='black')
ax.bar(x_pos + width/2, novice_norm, width, label='Novice',
      color='red', alpha=0.7, edgecolor='black')

ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.set_ylabel('Normalized Value', fontsize=12)
ax.set_title('Expert vs Novice: Key Metrics (Normalized)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('immediate_rt_correlation.png', dpi=150, bbox_inches='tight')
print("\n✓ RT 상관관계 플롯 저장: immediate_rt_correlation.png")
plt.close()

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('immediate_discrimination_results.csv')
print("\n✓ Discrimination 결과 저장: immediate_discrimination_results.csv")

print("\n" + "=" * 80)
print("즉시 분석 완료!")
print("=" * 80)

print("\n생성된 파일:")
print("  - immediate_pv_depth_analysis.png")
print("  - immediate_expertise_discrimination.png")
print("  - immediate_rt_correlation.png")
print("  - immediate_discrimination_results.csv")

print("\n주요 발견 요약:")
print("\n1. Planning Depth (PV):")
if 'pv_depth_mean' in participant_with_depth.columns:
    print(f"   - Expert vs Novice 차이 유의성: p={p_value:.4f}")
    print(f"   - Effect size (Cohen's d): {cohens_d:.3f}")
else:
    print("   - 데이터 부족으로 분석 불가")

print("\n2. Expertise Discrimination:")
best_auc = max([r['AUC'] for r in results.values()])
best_model_name = max(results.keys(), key=lambda k: results[k]['AUC'])
print(f"   - 최고 성능 모델: {best_model_name}")
print(f"   - AUC: {best_auc:.3f}")
print(f"   - Baseline (parameters only) vs Best: 개선 여부 확인")

print("\n3. RT-Parameter Correlations:")
print(f"   - RT와 log-likelihood 상관: r={corr:.3f}")
if 'pv_depth_mean' in corr_matrix.columns and 'mean' in corr_matrix.columns:
    rt_pv_corr = corr_matrix.loc['mean', 'pv_depth_mean']
    print(f"   - RT와 Planning Depth 상관: r={rt_pv_corr:.3f}")

print("\n다음 단계:")
print("  1. PV depth가 유의미하다면 → Fixed-h model 구현 정당화")
print("  2. Baseline AUC가 낮다면 → Planning-aware 모델의 필요성 입증")
print("  3. RT-depth 관계가 명확하다면 → Cognitive cost 모델링 가능")
