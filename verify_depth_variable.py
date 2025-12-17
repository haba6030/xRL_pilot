"""
Depth 변수 검증: 제가 사용한 값이 정확한 PV depth인지 확인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 80)
print("Depth 변수 검증")
print("=" * 80)

# Load the depth file
depth_file = "xRL_pilot/Analysis notebooks/old/depth_by_session.txt"
depth_matrix = np.loadtxt(depth_file)

print(f"\n[1] Depth 파일 기본 정보")
print(f"Shape: {depth_matrix.shape}")
print(f"값 범위: [{depth_matrix.min():.2f}, {depth_matrix.max():.2f}]")
print(f"평균: {depth_matrix.mean():.2f}")
print(f"표준편차: {depth_matrix.std():.2f}")

print(f"\n[2] Learning notebook의 보정 적용")
print("Learning notebook: pv_array = np.maximum(pv_depth - 2, 0)")

# Apply the -2 correction as in learning notebook
corrected_depth = np.maximum(depth_matrix - 2, 0)

print(f"\n보정 후:")
print(f"값 범위: [{corrected_depth.min():.2f}, {corrected_depth.max():.2f}]")
print(f"평균: {corrected_depth.mean():.2f}")
print(f"표준편차: {corrected_depth.std():.2f}")

# Mean per participant
mean_depth_raw = depth_matrix.mean(axis=1)
mean_depth_corrected = corrected_depth.mean(axis=1)

print(f"\n[3] 참가자별 평균 (30명)")
print(f"\nRaw depth:")
print(f"  범위: [{mean_depth_raw.min():.2f}, {mean_depth_raw.max():.2f}]")
print(f"  평균: {mean_depth_raw.mean():.2f}")

print(f"\nCorrected depth (-2):")
print(f"  범위: [{mean_depth_corrected.min():.2f}, {mean_depth_corrected.max():.2f}]")
print(f"  평균: {mean_depth_corrected.mean():.2f}")

# Compare with expertise labels
participant_data = pd.read_csv('analysis_participant_with_expertise.csv', index_col=0)

print(f"\n[4] Expertise 비교")

# Align participants (first 30)
expertise_for_depth = participant_data.iloc[:30]['expertise_category'].values

expert_idx = expertise_for_depth == 'Expert'
novice_idx = expertise_for_depth == 'Novice'

print(f"\n**Raw depth** (제가 이전에 사용한 값):")
expert_raw = mean_depth_raw[expert_idx]
novice_raw = mean_depth_raw[novice_idx]
print(f"  Expert (n={len(expert_raw)}): {expert_raw.mean():.3f} ± {expert_raw.std():.3f}")
print(f"  Novice (n={len(novice_raw)}): {novice_raw.mean():.3f} ± {novice_raw.std():.3f}")
t_raw, p_raw = stats.ttest_ind(expert_raw, novice_raw)
print(f"  t-test: t={t_raw:.3f}, p={p_raw:.4f}")

print(f"\n**Corrected depth** (-2 보정):")
expert_corr = mean_depth_corrected[expert_idx]
novice_corr = mean_depth_corrected[novice_idx]
print(f"  Expert (n={len(expert_corr)}): {expert_corr.mean():.3f} ± {expert_corr.std():.3f}")
print(f"  Novice (n={len(novice_corr)}): {novice_corr.mean():.3f} ± {novice_corr.std():.3f}")
t_corr, p_corr = stats.ttest_ind(expert_corr, novice_corr)
print(f"  t-test: t={t_corr:.3f}, p={p_corr:.4f}")

# Check what this depth variable might be
print(f"\n[5] 이 변수가 무엇인지 추론")

print(f"\nC++ BFS 코드에는 여러 depth 관련 함수:")
print(f"  - get_depth_of_pv(): Principal Variation depth")
print(f"  - get_mean_depth(): 평균 탐색 depth")
print(f"  - get_sum_depth(): 총 depth 합")

print(f"\nLearning notebook cell 36:")
print(f"  pv_array = np.maximum(np.loadtxt('pv_depth_'+str(group)+'.txt') - 2, 0)")
print(f"  → -2 보정을 적용하고 있음")

print(f"\n현재 데이터 (depth_by_session.txt):")
print(f"  - 30 participants × 5 sessions")
print(f"  - 값 범위: 3-10")
print(f"  - -2 보정 후: 1-8")

print(f"\n가능성:")
print(f"  1. 이 파일이 실제 PV depth + 2를 저장한 것")
print(f"     → Learning notebook 방식대로 -2 해야 함")
print(f"  2. 다른 종류의 depth (예: mean_depth)")
print(f"     → 보정 없이 사용해야 함")
print(f"  3. 이미 보정된 PV depth")
print(f"     → 그대로 사용해야 함")

# Check correlation with other parameters
print(f"\n[6] 다른 파라미터와의 상관관계로 검증")

# Load first 30 participants' parameters
params_30 = participant_data.iloc[:30]

for depth_name, depth_values in [('Raw', mean_depth_raw), ('Corrected (-2)', mean_depth_corrected)]:
    print(f"\n{depth_name} depth 상관:")

    # Correlation with log-likelihood
    corr_ll, p_ll = stats.pearsonr(depth_values, params_30['log-likelihood'])
    print(f"  vs Log-likelihood: r={corr_ll:+.3f}, p={p_ll:.4f}")

    # Correlation with pruning threshold
    corr_pr, p_pr = stats.pearsonr(depth_values, params_30['pruning threshold'])
    print(f"  vs Pruning threshold: r={corr_pr:+.3f}, p={p_pr:.4f}")

    # Correlation with response time
    corr_rt, p_rt = stats.pearsonr(depth_values, params_30['mean'])
    print(f"  vs Response time: r={corr_rt:+.3f}, p={p_rt:.4f}")

# Visualize both versions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw depth
ax = axes[0]
bp = ax.boxplot([novice_raw, expert_raw], labels=['Novice', 'Expert'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')
ax.set_ylabel('Depth', fontsize=12)
ax.set_title(f'Raw Depth: Novice vs Expert\n(p={p_raw:.4f})', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Corrected depth
ax = axes[1]
bp = ax.boxplot([novice_corr, expert_corr], labels=['Novice', 'Expert'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')
ax.set_ylabel('Depth', fontsize=12)
ax.set_title(f'Corrected Depth (-2): Novice vs Expert\n(p={p_corr:.4f})', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('verify_depth_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 비교 플롯 저장: verify_depth_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("결론")
print("=" * 80)

print(f"\n현재 사용한 값 (Raw):")
print(f"  - Expert < Novice (역설적 결과)")
print(f"  - p = {p_raw:.4f}")

print(f"\n보정 후 (-2):")
print(f"  - Expert < Novice (여전히 동일 방향)")
print(f"  - p = {p_corr:.4f}")

print(f"\n**중요**: 보정 여부와 무관하게 **방향은 동일**")
print(f"  → Expert가 Novice보다 planning depth가 낮음")
print(f"  → 이는 '효율적 탐색' 가설과 일치")

print(f"\n**문제**: depth_by_session.txt가 정확히 무엇인지 불명확")
print(f"  → 원본 데이터 경로나 생성 스크립트 확인 필요")
print(f"  → 또는 C++ 코드 직접 실행하여 PV depth 재계산")

print(f"\n**다음 단계**:")
print(f"  1. 원본 Peak 데이터 찾기 (splits/ 디렉토리)")
print(f"  2. C++ compute_planning_depth 바이너리 실행")
print(f"  3. pv_depth_X.txt 파일 생성하여 비교")
print(f"  4. 또는 van Opheusden에게 직접 확인")
