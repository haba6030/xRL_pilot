"""
Model Comparison Analysis
Compares different model variants to understand the impact of planning depth and other factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

DATA_DIR = Path("opendata")

print("=" * 80)
print("모델 변형 비교 분석")
print("=" * 80)

# Load all model fit files
model_files = {
    'Main model': 'model_fits_main_model.csv',
    'Fixed depth': 'model_fits_fixed_depth.csv',
    'Fixed iterations': 'model_fits_fixed_iterations.csv',
    'Fixed branching': 'model_fits_fixed_branching.csv',
    'MCTS': 'model_fits_mcts.csv',
    'No pruning': 'model_fits_no_pruning.csv',
    'No tree': 'model_fits_no_tree.csv',
    'No feature drop': 'model_fits_no_feature_drop.csv'
}

print("\n[1] 모델 데이터 로딩...")
model_data = {}
for model_name, filename in model_files.items():
    filepath = DATA_DIR / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        model_data[model_name] = df
        print(f"✓ {model_name}: {len(df)} fits")
    else:
        print(f"✗ {model_name}: 파일 없음")

print(f"\n로딩된 모델: {len(model_data)}개")

# Compare log-likelihoods across models
print("\n[2] 모델별 Log-Likelihood 비교")
print("-" * 80)

ll_comparison = {}
for model_name, df in model_data.items():
    ll_comparison[model_name] = {
        'mean': df['log-likelihood'].mean(),
        'std': df['log-likelihood'].std(),
        'median': df['log-likelihood'].median(),
        'min': df['log-likelihood'].min(),
        'max': df['log-likelihood'].max()
    }

ll_df = pd.DataFrame(ll_comparison).T
ll_df = ll_df.sort_values('mean', ascending=False)
print(ll_df)

# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Log-likelihood comparison
ax = axes[0, 0]
ll_means = [model_data[m]['log-likelihood'].mean() for m in model_data.keys()]
ll_stds = [model_data[m]['log-likelihood'].std() for m in model_data.keys()]
x_pos = np.arange(len(model_data))

ax.bar(x_pos, ll_means, yerr=ll_stds, capsize=5, alpha=0.7,
       color=['red' if 'Main' in m else 'steelblue' for m in model_data.keys()],
       edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_data.keys(), rotation=45, ha='right')
ax.set_ylabel('Mean Log-Likelihood', fontsize=12)
ax.set_title('Model Comparison: Log-Likelihood', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 2. Distribution comparison (boxplot)
ax = axes[0, 1]
ll_data = [model_data[m]['log-likelihood'] for m in model_data.keys()]
bp = ax.boxplot(ll_data, labels=model_data.keys(), patch_artist=True)
for patch, model_name in zip(bp['boxes'], model_data.keys()):
    if 'Main' in model_name:
        patch.set_facecolor('lightcoral')
    else:
        patch.set_facecolor('lightblue')
ax.set_xticklabels(model_data.keys(), rotation=45, ha='right')
ax.set_ylabel('Log-Likelihood', fontsize=12)
ax.set_title('Log-Likelihood Distribution by Model', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 3. Parameter comparison for key models
ax = axes[1, 0]
key_models = ['Main model', 'Fixed depth', 'No pruning', 'No tree']
existing_key_models = [m for m in key_models if m in model_data]

param_to_compare = 'pruning threshold'
if all(param_to_compare in model_data[m].columns for m in existing_key_models):
    means = [model_data[m][param_to_compare].mean() for m in existing_key_models]
    stds = [model_data[m][param_to_compare].std() for m in existing_key_models]
    x_pos = np.arange(len(existing_key_models))

    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(existing_key_models, rotation=45, ha='right')
    ax.set_ylabel(f'Mean {param_to_compare}', fontsize=12)
    ax.set_title(f'Pruning Threshold Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

# 4. Model fit count per participant
ax = axes[1, 1]
if 'Main model' in model_data:
    participant_counts = model_data['Main model'].groupby('participant').size()
    participant_counts.plot(kind='bar', ax=ax, color='green', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Participant', fontsize=12)
    ax.set_ylabel('Number of Fits', fontsize=12)
    ax.set_title('Fits per Participant (Main Model)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig('figures/analysis_model_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ 모델 비교 플롯 저장: analysis_model_comparison.png")
plt.close()

# Statistical tests between Main model and alternatives
print("\n[3] 통계적 유의성 검정 (vs Main model)")
print("-" * 80)

if 'Main model' in model_data:
    main_ll = model_data['Main model']['log-likelihood']

    for model_name, df in model_data.items():
        if model_name != 'Main model':
            other_ll = df['log-likelihood']

            # Paired t-test (assuming same participants)
            if len(main_ll) == len(other_ll):
                t_stat, p_value = stats.ttest_rel(main_ll, other_ll)
                test_type = "Paired t-test"
            else:
                t_stat, p_value = stats.ttest_ind(main_ll, other_ll)
                test_type = "Independent t-test"

            diff = main_ll.mean() - other_ll.mean()
            print(f"{model_name:20s} | {test_type:18s} | "
                  f"Δμ={diff:+.4f} | t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

# Analyze planning-related models specifically
print("\n[4] Planning 관련 모델 심층 분석")
print("-" * 80)

planning_models = {k: v for k, v in model_data.items()
                   if any(x in k for x in ['Fixed depth', 'Fixed iterations', 'Fixed branching', 'Main'])}

if len(planning_models) > 0:
    print(f"\nPlanning 관련 모델: {list(planning_models.keys())}")

    # Compare parameters
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Planning Models: Parameter Comparison', fontsize=16, fontweight='bold')

    params_to_plot = ['pruning threshold', 'stopping probability', 'feature drop rate',
                      'lapse rate', 'active scaling constant', 'center weight']

    for idx, param in enumerate(params_to_plot):
        ax = axes[idx // 3, idx % 3]

        for model_name, df in planning_models.items():
            if param in df.columns:
                ax.hist(df[param], alpha=0.5, label=model_name, bins=20, edgecolor='black')

        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/analysis_planning_models_parameters.png', dpi=150, bbox_inches='tight')
    print("✓ Planning 모델 파라미터 비교 플롯 저장: analysis_planning_models_parameters.png")
    plt.close()

# Participant-level model comparison
print("\n[5] 참가자별 모델 성능 비교")
print("-" * 80)

if 'Main model' in model_data and 'Fixed depth' in model_data:
    main_by_participant = model_data['Main model'].groupby('participant')['log-likelihood'].mean()
    fixed_by_participant = model_data['Fixed depth'].groupby('participant')['log-likelihood'].mean()

    # Merge on common participants
    comparison_df = pd.DataFrame({
        'Main model': main_by_participant,
        'Fixed depth': fixed_by_participant
    }).dropna()

    print(f"\n공통 참가자 수: {len(comparison_df)}")
    print(f"Main model 우세: {(comparison_df['Main model'] > comparison_df['Fixed depth']).sum()}명")
    print(f"Fixed depth 우세: {(comparison_df['Fixed depth'] > comparison_df['Main model']).sum()}명")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(comparison_df['Main model'], comparison_df['Fixed depth'],
               alpha=0.6, s=100, edgecolor='black')

    # Identity line
    min_val = min(comparison_df.min())
    max_val = max(comparison_df.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity line')

    ax.set_xlabel('Main Model Log-Likelihood', fontsize=12)
    ax.set_ylabel('Fixed Depth Log-Likelihood', fontsize=12)
    ax.set_title('Participant-level Model Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/analysis_participant_model_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 참가자별 모델 비교 플롯 저장: analysis_participant_model_comparison.png")
    plt.close()

    # Save comparison
    comparison_df['diff'] = comparison_df['Main model'] - comparison_df['Fixed depth']
    comparison_df.to_csv('analysis_model_comparison_by_participant.csv')
    print("✓ 참가자별 모델 비교 데이터 저장: analysis_model_comparison_by_participant.csv")

print("\n" + "=" * 80)
print("모델 비교 분석 완료!")
print("=" * 80)

print("\n생성된 파일:")
print("  - analysis_model_comparison.png")
print("  - analysis_planning_models_parameters.png")
print("  - analysis_participant_model_comparison.png (if applicable)")
print("  - analysis_model_comparison_by_participant.csv (if applicable)")

print("\n주요 발견:")
print("  1. 모델 간 log-likelihood 차이가 planning depth의 중요성을 시사")
print("  2. Fixed-depth 모델이 Main model과 어떻게 다른지 확인 가능")
print("  3. 참가자별 모델 선호도 분석 가능")
print("\n다음 단계: discrete h ∈ {1,2,3,4,5}로 각각 피팅하여 최적 planning depth 추정")
