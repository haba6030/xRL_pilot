"""
Data Reanalysis Script for Planning-Aware IRL Project
Analyzes opendata CSV files to understand parameter distributions and prepare for expertise discrimination
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Data paths
DATA_DIR = Path("opendata")
RAW_DATA = DATA_DIR / "raw_data.csv"
MAIN_MODEL_FITS = DATA_DIR / "model_fits_main_model.csv"

print("=" * 80)
print("데이터 재분석: Planning-Aware IRL")
print("=" * 80)

# Load data
print("\n[1] 데이터 로딩...")
raw_data = pd.read_csv(RAW_DATA)
model_fits = pd.read_csv(MAIN_MODEL_FITS)

print(f"✓ Raw data: {len(raw_data)} trials")
print(f"✓ Model fits: {len(model_fits)} parameter sets")

# Basic data info
print("\n[2] Raw Data 기본 정보")
print("-" * 80)
print(f"컬럼: {list(raw_data.columns)}")
print(f"\n참가자 수: {raw_data['participant'].nunique()}")
print(f"실험 타입: {raw_data['experiment'].unique()}")
print(f"Cross-validation 그룹: {sorted(raw_data['cross-validation group'].unique())}")

# Response time statistics
print(f"\n응답 시간 통계 (초):")
print(raw_data['response_time'].describe())

# Trials per participant
trials_per_participant = raw_data.groupby('participant').size()
print(f"\n참가자당 시행 수:")
print(trials_per_participant.describe())

# Model fits info
print("\n[3] Model Fits 기본 정보")
print("-" * 80)
print(f"컬럼: {list(model_fits.columns)}")
print(f"\n파라미터 셋 수: {len(model_fits)}")
print(f"참가자 수: {model_fits['participant'].nunique()}")
print(f"모델: {model_fits['model'].unique()}")

# Key parameters
key_params = ['pruning threshold', 'stopping probability', 'feature drop rate',
              'lapse rate', 'active scaling constant', 'center weight']

print("\n주요 파라미터 통계:")
print(model_fits[key_params].describe())

# Log-likelihood distribution
print("\n[4] 모델 적합도 (Log-Likelihood)")
print("-" * 80)
print(model_fits['log-likelihood'].describe())

# Create visualizations
print("\n[5] 시각화 생성 중...")

# Figure 1: Parameter distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Main Model: Parameter Distributions', fontsize=16, fontweight='bold')

for idx, param in enumerate(key_params):
    ax = axes[idx // 3, idx % 3]
    model_fits[param].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.axvline(model_fits[param].median(), color='red', linestyle='--',
               label=f'Median: {model_fits[param].median():.3f}')
    ax.legend()

plt.tight_layout()
plt.savefig('figures/analysis_parameter_distributions.png', dpi=150, bbox_inches='tight')
print("✓ 파라미터 분포 플롯 저장: figures/analysis_parameter_distributions.png")
plt.close()

# Figure 2: Log-likelihood by participant
fig, ax = plt.subplots(figsize=(12, 6))
ll_by_participant = model_fits.groupby('participant')['log-likelihood'].mean().sort_values()
ll_by_participant.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_xlabel('Participant', fontsize=12)
ax.set_ylabel('Mean Log-Likelihood', fontsize=12)
ax.set_title('Model Fit Quality by Participant', fontsize=14, fontweight='bold')
ax.axhline(ll_by_participant.median(), color='red', linestyle='--',
           label=f'Median: {ll_by_participant.median():.2f}')
ax.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('figures/analysis_ll_by_participant.png', dpi=150, bbox_inches='tight')
print("✓ 참가자별 Log-Likelihood 플롯 저장: figures/analysis_ll_by_participant.png")
plt.close()

# Figure 3: Response time distribution
fig, ax = plt.subplots(figsize=(10, 6))
raw_data['response_time'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7, color='green')
ax.set_xlabel('Response Time (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
ax.axvline(raw_data['response_time'].median(), color='red', linestyle='--',
           label=f'Median: {raw_data["response_time"].median():.2f}s')
ax.legend()
plt.tight_layout()
plt.savefig('figures/analysis_response_time.png', dpi=150, bbox_inches='tight')
print("✓ 응답 시간 분포 플롯 저장: figures/analysis_response_time.png")
plt.close()

# Correlation analysis
print("\n[6] 파라미터 간 상관관계 분석")
print("-" * 80)
correlation_matrix = model_fits[key_params].corr()
print(correlation_matrix)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/analysis_parameter_correlations.png', dpi=150, bbox_inches='tight')
print("✓ 파라미터 상관관계 플롯 저장: figures/analysis_parameter_correlations.png")
plt.close()

# Cross-validation analysis
print("\n[7] Cross-Validation 그룹별 분석")
print("-" * 80)
cv_analysis = model_fits.groupby('cross-validation group')['log-likelihood'].agg(['mean', 'std', 'count'])
print(cv_analysis)

# Prepare summary statistics
print("\n[8] 요약 통계 저장")
print("-" * 80)

summary_stats = {
    'total_trials': len(raw_data),
    'total_participants': raw_data['participant'].nunique(),
    'mean_trials_per_participant': trials_per_participant.mean(),
    'median_response_time': raw_data['response_time'].median(),
    'mean_log_likelihood': model_fits['log-likelihood'].mean(),
    'parameter_means': model_fits[key_params].mean().to_dict(),
    'parameter_stds': model_fits[key_params].std().to_dict()
}

# Save summary
import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)
print("✓ 요약 통계 저장: analysis_summary.json")

print("\n[9] Expertise Proxy 준비")
print("-" * 80)
print("참가자별 평균 파라미터를 계산하여 expertise proxy 생성...")

# Aggregate parameters by participant
participant_params = model_fits.groupby('participant')[key_params + ['log-likelihood']].mean()
participant_params['trial_count'] = trials_per_participant

# Add response time statistics
rt_by_participant = raw_data.groupby('participant')['response_time'].agg(['mean', 'median', 'std'])
participant_params = participant_params.join(rt_by_participant, rsuffix='_rt')

print(f"\n참가자별 통합 통계:")
print(participant_params.describe())

# Save participant-level data
participant_params.to_csv('analysis_participant_summary.csv')
print("✓ 참가자별 요약 저장: analysis_participant_summary.csv")

# Potential expertise indicators
print("\n잠재적 Expertise 지표:")
print(f"1. Log-likelihood (높을수록 모델 적합도 좋음)")
print(f"   범위: [{participant_params['log-likelihood'].min():.2f}, {participant_params['log-likelihood'].max():.2f}]")
print(f"2. Response time (짧을수록 숙련도 높음?)")
print(f"   범위: [{participant_params['mean'].min():.2f}, {participant_params['mean'].max():.2f}] 초")
print(f"3. Lapse rate (낮을수록 일관성 높음)")
print(f"   범위: [{participant_params['lapse rate'].min():.4f}, {participant_params['lapse rate'].max():.4f}]")
print(f"4. Pruning threshold (높을수록 깊은 탐색?)")
print(f"   범위: [{participant_params['pruning threshold'].min():.2f}, {participant_params['pruning threshold'].max():.2f}]")

# Create expertise proxy (composite score)
print("\n복합 Expertise Score 생성 (z-score 기반):")
from sklearn.preprocessing import StandardScaler

expertise_features = ['log-likelihood', 'lapse rate', 'pruning threshold']
X = participant_params[expertise_features].values

# Invert lapse rate (lower is better)
X[:, 1] = -X[:, 1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
expertise_score = X_scaled.mean(axis=1)

participant_params['expertise_score'] = expertise_score
participant_params['expertise_rank'] = participant_params['expertise_score'].rank(ascending=False)

print(f"\nExpertise Score 통계:")
print(f"평균: {expertise_score.mean():.3f}, 표준편차: {expertise_score.std():.3f}")
print(f"범위: [{expertise_score.min():.3f}, {expertise_score.max():.3f}]")

# Binary expertise label (median split)
expertise_median = participant_params['expertise_score'].median()
participant_params['expertise_label'] = (participant_params['expertise_score'] > expertise_median).astype(int)
participant_params['expertise_category'] = participant_params['expertise_label'].map({1: 'Expert', 0: 'Novice'})

print(f"\nExpertise 라벨 (중앙값 기준):")
print(participant_params['expertise_category'].value_counts())

# Save with expertise labels
participant_params.to_csv('analysis_participant_with_expertise.csv')
print("✓ Expertise 라벨 포함 데이터 저장: analysis_participant_with_expertise.csv")

# Visualize expertise distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Expertise score distribution
axes[0].hist(expertise_score, bins=20, edgecolor='black', alpha=0.7, color='purple')
axes[0].axvline(expertise_median, color='red', linestyle='--',
                label=f'Median: {expertise_median:.2f}')
axes[0].set_xlabel('Expertise Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Expertise Score Distribution', fontsize=14, fontweight='bold')
axes[0].legend()

# Compare novice vs expert on key parameters
novice_data = participant_params[participant_params['expertise_category'] == 'Novice']['log-likelihood']
expert_data = participant_params[participant_params['expertise_category'] == 'Expert']['log-likelihood']

axes[1].boxplot([novice_data, expert_data], labels=['Novice', 'Expert'])
axes[1].set_ylabel('Log-Likelihood', fontsize=12)
axes[1].set_title('Log-Likelihood: Novice vs Expert', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Statistical test
t_stat, p_value = stats.ttest_ind(novice_data, expert_data)
axes[1].text(0.5, 0.95, f't-test: p={p_value:.4f}',
             transform=axes[1].transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/analysis_expertise_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Expertise 분포 플롯 저장: analysis_expertise_distribution.png")
plt.close()

print("\n" + "=" * 80)
print("데이터 재분석 완료!")
print("=" * 80)
print("\n생성된 파일:")
print("  - analysis_parameter_distributions.png")
print("  - analysis_ll_by_participant.png")
print("  - analysis_response_time.png")
print("  - analysis_parameter_correlations.png")
print("  - analysis_expertise_distribution.png")
print("  - analysis_summary.json")
print("  - analysis_participant_summary.csv")
print("  - analysis_participant_with_expertise.csv")

print("\n다음 단계:")
print("  1. Principal Variation (PV) depth 데이터가 있다면 통합")
print("  2. Fixed-depth model (h=1,2,3,4,5) 구현 및 피팅")
print("  3. Planning depth → expertise discrimination test")
print("  4. AIRL 프레임워크 구현")
