# Pedestrian 방법론 적용 - 실행 가이드

**Last Updated**: 2025-12-26
**Status**: Phase 2 준비 완료

---

## 목차

1. [완료된 작업](#완료된-작업)
2. [즉시 실행 가능](#즉시-실행-가능)
3. [다음 단계](#다음-단계)
4. [전체 파이프라인](#전체-파이프라인)
5. [Pedestrian vs 현재 비교](#pedestrian-vs-현재-비교)

---

## 완료된 작업

### ✅ Fixed Horizon Wrapper

**파일**: `fixed_horizon_wrapper.py`

**기능**:
- Episode 길이를 36 steps로 고정
- Absorbing state로 padding
- Observation: 89-dim → 90-dim (absorbing flag 추가)

**테스트 결과**:
```bash
python3 fixed_horizon_wrapper.py

# Output:
# ✅ All tests passed!
# - Episode length: 36 (고정)
# - Absorbing transitions: 자동 padding
# - Observation shape: (90,)
```

**핵심 개선**:
```
Before (문제):
  h=1: 평균 17 turns
  h=4: 평균 26 turns
  → Discriminator가 episode 길이로 h 구분 가능 (cheating!)

After (해결):
  h=1: 정확히 36 turns (17 real + 19 absorbing)
  h=4: 정확히 36 turns (26 real + 10 absorbing)
  → Discriminator는 순수 행동 패턴만으로 구분해야 함
```

### ✅ 데이터 생성 파이프라인

**파일**: `generate_training_data_fixed_horizon.py`

**기능**:
- Fixed horizon trajectory 생성
- 대량 데이터 지원 (100+ episodes)
- 통계 정보 자동 저장

**사용법**:
```bash
# h=1 데이터 생성
python3 generate_training_data_fixed_horizon.py --h 1 --num_episodes 100

# h=4 데이터 생성
python3 generate_training_data_fixed_horizon.py --h 4 --num_episodes 100
```

---

## 즉시 실행 가능

### Step 1: 대량 데이터 생성

```bash
cd fourinarow_airl

# h=1 (100 episodes)
python3 generate_training_data_fixed_horizon.py \
    --h 1 \
    --num_episodes 100 \
    --seed 42

# h=4 (100 episodes)
python3 generate_training_data_fixed_horizon.py \
    --h 4 \
    --num_episodes 100 \
    --seed 43

# 결과 확인
ls -lh data/training_trajectories/
# trajectories_h1_fixed_horizon.pkl
# trajectories_h4_fixed_horizon.pkl
# summary_h1_fixed_horizon.txt
# summary_h4_fixed_horizon.txt
```

**예상 시간**:
- h=1: ~30분 (100 episodes × 17 avg steps)
- h=4: ~1-2시간 (100 episodes × 26 avg steps)

**예상 결과**:
```
h=1 데이터:
  Total episodes: 100
  Episode length: 36 (fixed)
  Avg actual gameplay: ~17 steps
  Avg absorbing: ~19 steps
  Observation dim: 90

h=4 데이터:
  Total episodes: 100
  Episode length: 36 (fixed)
  Avg actual gameplay: ~26 steps
  Avg absorbing: ~10 steps
  Observation dim: 90
```

### Step 2: 검증 (선택사항)

**Fixed horizon이 차이를 없애는지 확인**:

```bash
# 기존 verify_h_differences.py 수정 필요
# TODO: FixedHorizonWrapper 적용 버전 작성

# 예상 결과:
# Episode 길이: 모두 36 steps
# KL/JS divergence: 행동 패턴만 반영 (길이 효과 제거)
```

---

## 다음 단계

### A. airl_utils.py 업데이트 ⭐

**변경사항**: 90-dim observation 지원

```python
# airl_utils.py 수정 필요

def convert_to_imitation_format_fixed_horizon(game_trajectories, verbose=True):
    """
    Fixed horizon trajectories → imitation format

    Input:
        game_trajectories: List[Dict] with:
            - 'observations': (37, 90)  # 36 steps + 1
            - 'actions': (36,)

    Output:
        List[imitation.data.types.Trajectory]
    """
    from imitation.data.types import Trajectory

    trajectories = []
    for traj in game_trajectories:
        obs = traj['observations']  # (37, 90)
        acts = traj['actions']      # (36,)

        # Validation
        assert obs.shape == (37, 90), f"Expected (37, 90), got {obs.shape}"
        assert len(acts) == 36, f"Expected 36 actions, got {len(acts)}"

        traj = Trajectory(
            obs=obs.astype(np.float32),
            acts=acts.astype(np.int64),
            infos=None,
            terminal=True
        )
        trajectories.append(traj)

    return trajectories
```

### B. AIRL 학습 스크립트 작성

**새 파일**: `train_airl_fixed_horizon.py`

**Pedestrian 스타일**:
```python
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from imitation.algorithms.adversarial import airl
from imitation.rewards.reward_nets import BasicRewardNet

def train_AIRL_per_h(
    h: int,
    num_episodes: int = 100,

    # PPO config (Pedestrian 기반)
    ppo_n_steps=512,
    ppo_lr=1e-4,
    ppo_batch_size=32,
    ppo_n_epochs=8,
    ppo_ent_coef=0.01,

    # AIRL config
    gen_train_timesteps=512,
    airl_train_n_rounds=10_000,  # Pedestrian: 20K
    demo_batch_size=256,
    n_disc_updates_per_round=1,

    # Reward network
    reward_net_hid_sizes=[8, 8],  # Pedestrian: [8, 8]
    reward_net_activation=nn.LeakyReLU,

    seed=42
):
    """
    Per-h AIRL training (Pedestrian style)

    Key features:
    - Fixed horizon environment
    - Large dataset (100+ episodes)
    - Long training (10K+ rounds)
    - Small reward net (interpretability)
    """

    # 1. Load expert data
    import pickle
    with open(f'data/training_trajectories/trajectories_h{h}_fixed_horizon.pkl', 'rb') as f:
        expert_trajs = pickle.load(f)

    print(f"Loaded {len(expert_trajs)} expert trajectories for h={h}")

    # 2. Convert to imitation format
    from airl_utils import convert_to_imitation_format_fixed_horizon
    demonstrations = convert_to_imitation_format_fixed_horizon(expert_trajs)

    # 3. Create environment with Fixed Horizon
    from env import FourInARowEnv
    from fixed_horizon_wrapper import FixedHorizonWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        env = FourInARowEnv()
        env = FixedHorizonWrapper(env, max_episode_length=36)
        return env

    venv = DummyVecEnv([make_env])

    # 4. Generator PPO
    gen_algo = PPO(
        "MlpPolicy",
        venv,
        learning_rate=ppo_lr,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        n_epochs=ppo_n_epochs,
        ent_coef=ppo_ent_coef,
        seed=seed,
        verbose=1
    )

    # 5. Reward network
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        hid_sizes=reward_net_hid_sizes,
        activation=reward_net_activation
    )

    # 6. AIRL trainer
    trainer = airl.AIRL(
        demonstrations=demonstrations,
        demo_batch_size=demo_batch_size,
        venv=venv,
        gen_algo=gen_algo,
        reward_net=reward_net,
        n_disc_updates_per_round=n_disc_updates_per_round,
        gen_train_timesteps=gen_train_timesteps,
    )

    # 7. Train
    print(f"\nTraining AIRL for h={h}...")
    print(f"  Expert episodes: {len(demonstrations)}")
    print(f"  Training rounds: {airl_train_n_rounds}")
    print(f"  Gen timesteps/round: {gen_train_timesteps}")

    trainer.train(total_timesteps=gen_train_timesteps * airl_train_n_rounds)

    # 8. Save
    os.makedirs(f'models/airl_fixed_horizon/h{h}', exist_ok=True)
    torch.save(reward_net.state_dict(), f'models/airl_fixed_horizon/h{h}/reward_net.pt')
    gen_algo.save(f'models/airl_fixed_horizon/h{h}/generator_ppo.zip')

    print(f"\n✅ Training complete for h={h}")
    print(f"   Saved to models/airl_fixed_horizon/h{h}/")

# Usage
if __name__ == '__main__':
    # Train h=1
    train_AIRL_per_h(h=1, num_episodes=100)

    # Train h=4
    train_AIRL_per_h(h=4, num_episodes=100)
```

### C. Phase 2.5: 사람 데이터 분석

**구조** (Pedestrian 기반):
```python
# analyze_human_data.py

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 1. Load trained discriminators
D_1 = load_discriminator('models/airl_fixed_horizon/h1/reward_net.pt')
D_4 = load_discriminator('models/airl_fixed_horizon/h4/reward_net.pt')

# 2. Load human trajectories (from raw_data.csv)
# TODO: Extract human trajectories with Fixed Horizon format
# - 40 participants
# - ~100-500 games per participant
# - Convert to (37, 90) format

# 3. Per-participant analysis
results = []
for pid in range(1, 41):
    human_trajs = load_human_trajectories(pid)  # TODO: implement

    # Score with both discriminators
    scores_h1 = [D_1(traj) for traj in human_trajs]
    scores_h4 = [D_4(traj) for traj in human_trajs]

    # Aggregate
    mean_h1 = np.mean(scores_h1)
    mean_h4 = np.mean(scores_h4)

    # Infer h
    inferred_h = 1 if mean_h1 > mean_h4 else 4
    confidence = abs(mean_h1 - mean_h4)

    results.append({
        'participant': pid,
        'inferred_h': inferred_h,
        'score_h1': mean_h1,
        'score_h4': mean_h4,
        'confidence': confidence,
        'n_games': len(human_trajs)
    })

# 4. Correlation with Elo
results_df = pd.DataFrame(results)
elo_df = pd.read_csv('data/human_elo_ratings.csv')

merged = results_df.merge(elo_df, on='participant')

# H1: h ~ Elo correlation
corr, p = spearmanr(merged['inferred_h'], merged['elo'])
print(f"h-Elo correlation: r={corr:.3f}, p={p:.4f}")

# H2: Expert vs Novice
from scipy.stats import mannwhitneyu
experts = merged[merged['expertise'] == 'expert']['inferred_h']
novices = merged[merged['expertise'] == 'novice']['inferred_h']
stat, p = mannwhitneyu(experts, novices)
print(f"Expert vs Novice: U={stat}, p={p:.4f}")
```

---

## 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: 데이터 준비 (완료)                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Fixed Horizon Wrapper 구현 ✅                            │
│ 2. 데이터 생성 파이프라인 ✅                                │
│ 3. 대량 데이터 생성 (100 episodes × 2 h values)            │
│    → h=1: data/training_trajectories/trajectories_h1_...    │
│    → h=4: data/training_trajectories/trajectories_h4_...    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: AIRL 학습                                          │
├─────────────────────────────────────────────────────────────┤
│ 1. airl_utils.py 업데이트 (90-dim support)                 │
│ 2. train_airl_fixed_horizon.py 작성                        │
│ 3. Per-h AIRL 학습 (Pedestrian 스타일)                     │
│    → h=1: 100 expert eps, 10K training rounds               │
│    → h=4: 100 expert eps, 10K training rounds               │
│ 4. 모델 저장: models/airl_fixed_horizon/h{1,4}/            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2.5: 사람 데이터 분석                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. 사람 trajectory 추출 (opendata/raw_data.csv)            │
│ 2. Fixed Horizon 포맷 변환 (37, 90)                        │
│ 3. Discriminator scoring (D_1, D_4)                         │
│ 4. Per-participant h 추론                                   │
│ 5. 통계 분석:                                               │
│    - h ~ Elo correlation                                    │
│    - Expert vs Novice comparison                            │
│    - Individual differences                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Pedestrian vs 현재 비교

| 측면 | Pedestrian (참고) | 4-in-a-row (현재) | 상태 |
|------|-------------------|-------------------|------|
| **Fixed Horizon** | FixedHorizonEnvWrapper | FixedHorizonWrapper | ✅ 구현 완료 |
| **Absorbing State** | obs + flag (114-dim) | obs + flag (90-dim) | ✅ 구현 완료 |
| **데이터 규모** | 46 subjects × 100 eps | 2 h_values × 100 eps | ✅ 충분 |
| **Episode 길이** | Variable → Fixed (max) | Variable → Fixed (36) | ✅ 해결 |
| **AIRL 구조** | Per-subject | Per-h | ✅ 적절 |
| **Reward Net** | [8, 8] LeakyReLU | [8, 8] LeakyReLU | 📋 계획됨 |
| **Training Rounds** | 20,000 | 10,000 (계획) | 📋 계획됨 |
| **통계 분석 단위** | Subject-level (n=46) | Participant-level (n=40) | 📋 계획됨 |

---

## 다음 실행 명령

### 지금 바로 실행:

```bash
cd fourinarow_airl

# 1. h=1 데이터 생성
python3 generate_training_data_fixed_horizon.py --h 1 --num_episodes 100

# 2. h=4 데이터 생성
python3 generate_training_data_fixed_horizon.py --h 4 --num_episodes 100

# 3. 결과 확인
ls -lh data/training_trajectories/
cat data/training_trajectories/summary_h1_fixed_horizon.txt
cat data/training_trajectories/summary_h4_fixed_horizon.txt
```

### 다음 단계:

1. **airl_utils.py 업데이트** (필수)
2. **train_airl_fixed_horizon.py 작성** (필수)
3. **AIRL 학습 실행** (시간 소요: ~수 시간)
4. **Phase 2.5 준비** (사람 데이터 추출)

---

**작성자**: Claude Code
**날짜**: 2025-12-26
**버전**: 1.0
