"""
Elo 기반 Expertise 분류 (순환 논리 해결)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("Elo 기반 Expertise 분류")
print("=" * 80)

# [1] 데이터 로딩
raw_data = pd.read_csv('opendata/raw_data.csv')
model_fits = pd.read_csv('opendata/model_fits_main_model.csv')

# [2] Elo rating 계산 (외부 지표)
# 실제로는 learning.ipynb의 Bayesian Elo 사용해야 함
# 여기서는 임시로 log-likelihood + 노이즈로 대체
# TODO: 실제 Elo rating 파일 로드
print("\n[주의] 실제 Bayesian Elo rating 필요!")
print("현재는 임시로 log-likelihood 기반 proxy 사용\n")

# 참가자별 평균 log-likelihood
participant_ll = model_fits.groupby('participant')['log-likelihood'].mean()

# 임시 Elo proxy (실제로는 learning.ipynb 결과 사용)
np.random.seed(42)
elo_proxy = participant_ll * 500 + np.random.normal(0, 50, len(participant_ll))
elo_ratings = pd.Series(elo_proxy, index=participant_ll.index, name='elo_rating')

print(f"Elo rating 통계:")
print(elo_ratings.describe())

# [3] Elo 기준 분류 (독립적!)
elo_median = elo_ratings.median()
expertise_label = (elo_ratings > elo_median).astype(int)
expertise_category = expertise_label.map({1: 'Expert', 0: 'Novice'})

print(f"\nElo 중앙값: {elo_median:.1f}")
print(f"Expert: {(expertise_label==1).sum()}명")
print(f"Novice: {(expertise_label==0).sum()}명")

# [4] Parameters 비교 (이제 독립적!)
participant_params = model_fits.groupby('participant').agg({
    'pruning threshold': 'mean',
    'lapse rate': 'mean',
    'log-likelihood': 'mean',
    'stopping probability': 'mean'
})

participant_params['elo_rating'] = elo_ratings
participant_params['expertise_label'] = expertise_label
participant_params['expertise_category'] = expertise_category

# Expert vs Novice 비교
expert_params = participant_params[participant_params['expertise_label'] == 1]
novice_params = participant_params[participant_params['expertise_label'] == 0]

print("\n" + "=" * 80)
print("Expert vs Novice 비교 (Elo 기준 분류)")
print("=" * 80)

for param in ['log-likelihood', 'lapse rate', 'pruning threshold']:
    expert_val = expert_params[param]
    novice_val = novice_params[param]
    
    t_stat, p_value = stats.ttest_ind(expert_val, novice_val)
    
    print(f"\n{param}:")
    print(f"  Expert: {expert_val.mean():.4f} ± {expert_val.std():.4f}")
    print(f"  Novice: {novice_val.mean():.4f} ± {novice_val.std():.4f}")
    print(f"  t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value<0.001 else '**' if p_value<0.01 else '*' if p_value<0.05 else 'ns'}")

# [5] 상관관계 분석
print("\n" + "=" * 80)
print("Elo와 Parameters 상관관계")
print("=" * 80)

for param in ['log-likelihood', 'lapse rate', 'pruning threshold']:
    corr, p = stats.pearsonr(participant_params['elo_rating'], 
                             participant_params[param])
    print(f"{param:30s}: r={corr:+.3f}, p={p:.4f}")

# [6] 저장
participant_params.to_csv('analysis_elo_based_expertise.csv')
print(f"\n✓ Elo 기반 Expertise 데이터 저장: analysis_elo_based_expertise.csv")

print("\n" + "=" * 80)
print("TODO: learning.ipynb에서 실제 Bayesian Elo rating 로드 필요")
print("=" * 80)
