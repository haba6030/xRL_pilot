# AIRL 완전 가이드: 전체 구조와 실행 방법

**Last Updated**: 2025-12-26
**Main Approach**: **Option A (Pure NN)** ⭐
**Status**: Option B baseline 완료 (71%), Option A 진행 예정

---

## ⚠️ 중요: 연구 방향

**이 프로젝트는 Option A (Pure Neural Network)를 main approach로 진행합니다.**

### 왜 Option A를 선택하는가?

| 이유 | 설명 |
|------|------|
| **순수한 IRL 검증** | Domain knowledge 없이 순수 학습으로 planning depth 효과 측정 |
| **이론적 정합성** | Planning depth가 행동에 미치는 순수한 영향 분리 |
| **Pedestrian 프로젝트 일관성** | 기존 연구와 동일한 접근법 |
| **연구 질문에 직접 답변** | "Planning depth만으로 행동이 달라지는가?" |

### 두 가지 옵션

- **Option A (Pure NN)**: Random 초기화 → 순수 AIRL 학습 - **Main Approach** ⭐
- **Option B (BC Distillation)**: BFS → BC → AIRL fine-tuning - **Baseline/비교군**

### 현재 상태

- ✅ **Option B (Baseline)**: Steps A-E 완료 (71%) - 비교 기준 확보
- 🔄 **Option A (Main)**: 코드 준비 완료, 실험 진행 예정

---

## 📋 목차
1. [AIRL이란 무엇인가?](#1-airl이란-무엇인가)
2. [전체 파이프라인 구조](#2-전체-파이프라인-구조)
3. [각 단계 상세 설명](#3-각-단계-상세-설명)
4. [옵션별 차이점](#4-옵션별-차이점)
5. [현재 구현 현황](#5-현재-구현-현황)
6. [실행 방법](#6-실행-방법)

---

## 1. AIRL이란 무엇인가?

### 핵심 아이디어

**AIRL (Adversarial Inverse Reinforcement Learning)**는:
- **입력**: Expert의 행동 데이터 (trajectories)
- **출력**: Expert를 만들어낸 "Reward function"
- **방법**: GAN처럼 Discriminator(reward)와 Generator(policy)를 번갈아 학습

### 왜 필요한가?

```
기존 방식:
  사람이 reward 설계 → RL로 학습 → 정책
  ⚠️ 문제: reward 설계가 어렵고 실제 의도와 다를 수 있음

AIRL 방식:
  Expert 행동 관찰 → AIRL → Reward 복원 → 정책
  ✅ 장점: Expert가 실제로 최적화한 reward를 역추론
```

### 우리 프로젝트의 목표

**"Planning depth h가 행동에 어떻게 영향을 주는가?"**
- Expert는 depth h=4로 계획함
- 다른 h (1, 2, 8)로 학습하면 어떻게 되는가?
- Reward는 depth와 독립적이어야 함!

---

## 2. 전체 파이프라인 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIRL PIPELINE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Step 0] Expert Data                                           │
│      ↓                                                           │
│      └─→ Expert trajectories (state, action, next_state)        │
│          (NO depth information!)                                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [Step 1] Environment Setup                             │  │
│  │  - FourInARowEnv (6x6 board)                            │  │
│  │  - Observation: 89-dim (board + features, NO h)         │  │
│  │  - Action: 36 positions                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [Step 2] Generator (Policy) - 옵션 선택!               │  │
│  │                                                          │  │
│  │  ⭐ Option A: Pure NN (Main Approach)                   │  │
│  │  ├─ Random initialization                                │  │
│  │  └─ 순수 AIRL 학습 (50K-100K steps)                    │  │
│  │                                                          │  │
│  │  Option B: BFS Distillation (Baseline)                   │  │
│  │  ├─ Step 2a: BFS(h) 데이터 생성                         │  │
│  │  ├─ Step 2b: BC로 BFS 모방                              │  │
│  │  ├─ Step 2c: PPO로 래핑                                  │  │
│  │  └─ AIRL fine-tuning (10K steps)                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [Step 3] Discriminator (Reward Network)                │  │
│  │  - BasicRewardNet                                        │  │
│  │  - Input: (state, action, next_state)                    │  │
│  │  - Output: reward (scalar)                               │  │
│  │  - ✅ NO depth parameter!                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [Step 4] AIRL Training Loop                            │  │
│  │                                                          │  │
│  │  for iteration in range(num_iterations):                │  │
│  │    1. Generator rollout                                  │  │
│  │       └─ Generate trajectories using current policy      │  │
│  │                                                          │  │
│  │    2. Discriminator update                               │  │
│  │       ├─ Sample expert trajectories                      │  │
│  │       ├─ Sample generated trajectories                   │  │
│  │       └─ Train to distinguish expert vs generated        │  │
│  │                                                          │  │
│  │    3. Generator update                                   │  │
│  │       └─ Improve policy using discriminator feedback     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  [Step 5] Evaluation                                            │
│      ├─ Compare with expert behavior                            │
│      ├─ Measure KL divergence                                   │
│      └─ Evaluate win rate                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 각 단계 상세 설명

### Step 0: Expert Data 준비

**목적**: AIRL이 모방할 expert 행동 데이터 준비

**옵션 1: 실제 human data 사용**
```python
# opendata/raw_data.csv 로드
expert_trajectories = load_expert_trajectories(
    csv_path='opendata/raw_data.csv',
    player_filter=0,  # Black player
    max_trajectories=100
)
```

**옵션 2: Synthetic data (BFS 생성)**
```python
# BFS(h=4)로 데이터 생성 (expert 대용)
from generate_training_data import generate_depth_limited_trajectories

expert_trajs = generate_depth_limited_trajectories(
    h=4,  # "Expert"는 h=4로 계획한다고 가정
    num_episodes=100
)
```

**데이터 형식**:
```python
GameTrajectory:
  - observations: (T+1, 89)  # 상태 시퀀스 (최종 상태 포함)
  - actions: (T,)             # 행동 시퀀스
  - rewards: (T,)             # 보상 시퀀스 (terminal만 ±1)
  - ✅ NO depth label!        # depth 정보 없음!
```

**중요**: Expert 데이터에는 **depth 정보가 없습니다**! 단지 (state, action) 쌍만 있습니다.

---

### Step 1: Environment Setup

**목적**: 4-in-a-row 환경 구축

**파일**: `fourinarow_airl/env.py`

**주요 특징**:
```python
env = FourInARowEnv()

# Observation space
env.observation_space
# Box(89,)
# - 0-35: Black pieces (6x6 board)
# - 36-71: White pieces (6x6 board)
# - 72-88: Van Opheusden features (17 features)
# ✅ NO depth encoding!

# Action space
env.action_space
# Discrete(36) - 6x6 board positions
```

**왜 89-dim?**
- Board state (72) + Van Opheusden heuristic features (17)
- Depth h는 **포함되지 않음** (관찰 불가능)

---

### Step 2: Generator (Policy) 생성

**목적**: Expert를 모방할 정책 학습

이 단계에서 **Option A vs Option B 선택!**

#### **Option A: Pure Neural Network**

**철학**: Domain knowledge 없이 순수 학습

**단계**:
```python
# 1. Random 초기화
from create_ppo_generator_pure_nn import create_pure_ppo_generator

gen_algo, venv = create_pure_ppo_generator(
    env=env,
    h=4,  # Naming only (NOT in network!)
    learning_rate=3e-4
)

# 2. AIRL에서 바로 사용
# (no pretraining)
```

**특징**:
- ✅ 순수 neural network
- ✅ Random weights
- ⚠️ 느린 학습 (50K-100K steps)
- ⚠️ 불안정할 수 있음

**파일**: `fourinarow_airl/create_ppo_generator_pure_nn.py`

---

#### **Option B: BFS Distillation**

**철학**: Van Opheusden BFS를 warm start로 활용

**단계**:

**2a. BFS 데이터 생성**
```python
from generate_training_data import generate_depth_limited_trajectories

# BFS(h=4)로 게임 플레이
training_trajs = generate_depth_limited_trajectories(
    h=4,
    num_episodes=100,
    seed=42
)
```

**파일**: `fourinarow_airl/generate_training_data.py`

**2b. Behavior Cloning (BC) 학습**
```python
from train_bc import train_bc_policy

# BFS 행동을 신경망이 모방
bc_trainer = train_bc_policy(
    trajectories=training_trajs,
    env=env,
    h=4,
    n_epochs=50
)
```

**목적**: BFS의 (state → action) 매핑을 신경망으로 학습

**파일**: `fourinarow_airl/train_bc.py`

**2c. PPO로 래핑**
```python
from create_ppo_generator import create_ppo_from_bc

# BC policy를 PPO로 래핑 (AIRL에서 fine-tune 가능)
gen_algo, venv = create_ppo_from_bc(
    bc_trainer=bc_trainer,
    env=env,
    h=4
)
```

**파일**: `fourinarow_airl/create_ppo_generator.py`

**특징**:
- ✅ BFS 지식 활용
- ✅ 빠른 학습 (10K steps)
- ✅ 안정적
- ⚠️ BC와 IRL 효과 구분 어려움

---

### Step 3: Discriminator (Reward Network) 생성

**목적**: Expert vs Generated를 구분하는 reward function 학습

```python
from create_reward_net import create_reward_network

reward_net = create_reward_network(env)
# ✅ NO h parameter!
```

**아키텍처**:
```python
BasicRewardNet:
  Input: (state, action, next_state, done)
    - state: (batch, 89)
    - action: (batch,) - discrete indices
    - next_state: (batch, 89)
    - done: (batch,) - boolean

  Internal:
    - Preprocessing (one-hot encoding for actions)
    - MLP [64, 64]
    - Tanh activation

  Output: (batch, 1) - reward
```

**중요 원칙**:
- ✅ **NO depth parameter** - reward는 관찰 가능한 정보만 사용
- ✅ 모든 h 실험에서 **동일한 아키텍처** 사용
- ✅ 각 h마다 **별도로 학습** (fresh instance)

**파일**: `fourinarow_airl/create_reward_net.py`

---

### Step 4: AIRL Training Loop

**목적**: Discriminator와 Generator를 번갈아 학습

```python
from train_airl import train_airl_single_depth  # Option B
# OR
from train_airl_pure_nn import train_airl_pure_nn  # Option A

trainer = train_airl_single_depth(
    h=4,
    expert_trajectories=expert_trajectories,
    env=env,
    total_timesteps=10000  # Option B
    # total_timesteps=50000  # Option A
)
```

**내부 동작** (imitation library가 자동 처리):

```python
# Pseudocode
for iteration in range(total_timesteps // gen_train_timesteps):

    # 1. Generator rollout
    gen_trajectories = []
    for _ in range(gen_train_timesteps):
        trajectory = gen_algo.rollout(env)
        gen_trajectories.append(trajectory)

    # 2. Discriminator update (n_disc_updates_per_round 번)
    for _ in range(n_disc_updates_per_round):
        # Sample expert batch
        expert_batch = sample(expert_trajectories, demo_batch_size)

        # Sample generated batch
        gen_batch = sample(gen_trajectories, demo_batch_size)

        # Binary classification loss
        # Expert = 1, Generated = 0
        loss = BCE(
            reward_net(expert_batch), ones
        ) + BCE(
            reward_net(gen_batch), zeros
        )

        # Update discriminator
        optimizer.step()

    # 3. Generator update (PPO)
    # Use discriminator output as reward
    gen_algo.learn(
        rollouts=gen_trajectories,
        reward_function=reward_net
    )
```

**하이퍼파라미터**:
- `demo_batch_size`: Discriminator 학습 배치 크기 (기본: 64)
- `n_disc_updates_per_round`: Discriminator 업데이트 횟수 (기본: 4)
- `gen_train_timesteps`: Generator 롤아웃 길이 (기본: 2048)

**파일**:
- `fourinarow_airl/train_airl.py` (Option B)
- `fourinarow_airl/train_airl_pure_nn.py` (Option A)

---

### Step 5: Evaluation

**목적**: 학습된 policy가 expert를 얼마나 잘 모방하는지 평가

```python
# 1. Trajectory 생성
from compare_option_a_vs_b import generate_trajectories

test_trajs = generate_trajectories(
    gen_algo=trained_gen,
    env=env,
    num_episodes=50
)

# 2. Metrics 계산
# - Action distribution similarity (KL divergence)
kl_div = compute_kl_divergence(expert_dist, test_dist)

# - Win rate
win_rate = compute_win_rate(test_trajs)

# - Trajectory length distribution
lengths = [len(t.actions) for t in test_trajs]
```

**파일**: `compare_option_a_vs_b.py`

---

## 4. 옵션별 차이점

### 전체 비교표

| 컴포넌트 | **Option A** | **Option B** | **차이점** |
|---------|-------------|-------------|-----------|
| **Expert Data** | Human or BFS(h=4) | Human or BFS(h=4) | ✅ 동일 |
| **Environment** | FourInARowEnv (89-dim) | FourInARowEnv (89-dim) | ✅ 동일 |
| **Generator 초기화** | Random | BC(BFS) | ⚠️ **핵심 차이!** |
| **Generator 학습** | 순수 AIRL | BC → AIRL fine-tune | ⚠️ 다름 |
| **Timesteps** | 50K-100K | 10K | ⚠️ 다름 |
| **Reward Network** | BasicRewardNet (NO h) | BasicRewardNet (NO h) | ✅ 동일 |
| **AIRL Algorithm** | imitation.airl.AIRL | imitation.airl.AIRL | ✅ 동일 |
| **Output** | Learned policy + reward | Learned policy + reward | ✅ 동일 |

### 핵심 포인트

1. **Reward network는 항상 동일**
   - 둘 다 depth-agnostic
   - 둘 다 (state, action, next_state)만 사용
   - 각 h마다 별도 학습하지만 아키텍처는 같음

2. **차이는 Generator 초기화뿐**
   - Option A: 백지 상태
   - Option B: BFS 지식 사전 학습

3. **h의 역할**
   - Option A: Naming only (파일 저장용)
   - Option B: BC 데이터 생성 시 BFS(h) 사용, 이후 naming
   - 둘 다: Network에는 h parameter 없음!

---

## 5. 현재 구현 현황

### ✅ 완료된 파일들

#### **환경 & 데이터**
- ✅ `fourinarow_airl/env.py` - 4-in-a-row 환경
- ✅ `fourinarow_airl/features.py` - Van Opheusden features
- ✅ `fourinarow_airl/data_loader.py` - Expert 데이터 로더
- ✅ `fourinarow_airl/bfs_wrapper.py` - BFS C++ wrapper

#### **Option B 구현** (BFS Distillation)
- ✅ `fourinarow_airl/generate_training_data.py` - BFS 데이터 생성
- ✅ `fourinarow_airl/depth_limited_policy.py` - BFS policy wrapper
- ✅ `fourinarow_airl/train_bc.py` - Behavior Cloning
- ✅ `fourinarow_airl/create_ppo_generator.py` - BC → PPO
- ✅ `fourinarow_airl/create_reward_net.py` - Reward network
- ✅ `fourinarow_airl/airl_utils.py` - Utility functions
- ✅ `fourinarow_airl/train_airl.py` - AIRL 학습 (Option B)

#### **Option A 구현** (Pure NN)
- ✅ `fourinarow_airl/create_ppo_generator_pure_nn.py` - Pure NN generator
- ✅ `fourinarow_airl/train_airl_pure_nn.py` - AIRL 학습 (Option A)

#### **비교 & 분석**
- ✅ `compare_option_a_vs_b.py` - Option A vs B 비교
- ✅ `visualize_option_difference.py` - 시각화 생성

#### **테스트 & 검증**
- ✅ `verify_depth_utility.py` - Depth utility 검증
- ✅ `test_phase1_integration.py` - Phase 1 통합 테스트

#### **문서**
- ✅ `AIRL_DESIGN.md` - 초기 설계 문서
- ✅ `OPTION_A_VS_B.md` - 옵션 비교 상세 문서
- ✅ `OPTION_DIFFERENCE_SIMPLE.md` - 간단 설명
- ✅ `AIRL_COMPLETE_GUIDE.md` - 이 문서!

### 🔧 구현 상태 체크리스트

#### Phase 1: 기본 Infrastructure ✅
- [x] Environment 구현
- [x] Feature extraction
- [x] Data loading
- [x] BFS wrapper

#### Phase 2: Option B Pipeline (Baseline) ✅
- [x] BFS data generation (Step A)
- [x] BC training (Step B)
- [x] PPO wrapping (Step C)
- [x] Reward network (Step D)
- [x] AIRL training (Step E)
- **Status**: 5/7 steps complete (71%)

#### Phase 3: Option A Pipeline (Main) ✅
- [x] Pure NN generator 코드 준비
- [x] AIRL training 코드 준비
- **Status**: 코드 완료, 실험 진행 예정

#### Phase 4: Evaluation & Comparison ✅
- [x] Comparison script
- [x] Visualization
- [x] Documentation

#### Phase 5: Experiments (🔄 진행 예정)

**Main Experiments (Option A - Priority 1)** ⭐:
- [ ] Option A 학습: h=1 (50K-100K steps)
- [ ] Option A 학습: h=2 (50K-100K steps)
- [ ] Option A 학습: h=4 (50K-100K steps)
- [ ] Option A 학습: h=8 (50K-100K steps)
- [ ] Option A 평가 및 분석

**Baseline Experiments (Option B - Priority 2)**:
- [x] Option B 학습: Steps A-E 완료
- [ ] Option B 추가 학습 (더 많은 timesteps)
- [ ] Option B 평가 및 분석

**Comparison Analysis**:
- [ ] Option A vs B 성능 비교
- [ ] Depth discrimination 테스트
- [ ] Best h 식별

---

## 6. 실행 방법

### 6.1 환경 설정

```bash
# Conda environment 활성화
conda activate pedestrian_analysis

# 필수 패키지 확인
pip list | grep -E "imitation|stable-baselines3|torch|gymnasium"
```

### 6.2 Option A 실행 (Pure NN) ⭐ Main Approach

**권장**: Option A를 main approach로 사용하세요.

#### 단계별 실행

```bash
# Step 1: 테스트 (선택사항)
python3 fourinarow_airl/create_ppo_generator_pure_nn.py --test

# Step 2: AIRL 학습
python3 fourinarow_airl/train_airl_pure_nn.py --h 4 --total_timesteps 50000

# 출력:
# - models/airl_pure_nn_results/airl_pure_generator_h4.zip
# - models/airl_pure_nn_results/airl_pure_reward_h4.pt
# - models/airl_pure_nn_results/airl_pure_metadata_h4.pkl
```

#### 모든 depth 학습

```bash
for h in 1 2 4 8; do
    python3 fourinarow_airl/train_airl_pure_nn.py \
        --h $h \
        --total_timesteps 50000 \
        --output_dir models/airl_pure_nn_results
done
```

### 6.3 Option B 실행 (BFS Distillation) - Baseline

**용도**: 빠른 baseline 구축 또는 비교군

#### 단계별 실행

```bash
# Step 1: BFS 데이터 생성
python3 fourinarow_airl/generate_training_data.py \
    --h 4 \
    --num_episodes 100 \
    --output training_data/depth_h4.pkl

# Step 2: BC 학습
python3 fourinarow_airl/train_bc.py \
    --h 4 \
    --training_data training_data/depth_h4.pkl \
    --n_epochs 50

# Step 3: PPO generator 생성
python3 fourinarow_airl/create_ppo_generator.py --h 4

# Step 4: AIRL 학습
python3 fourinarow_airl/train_airl.py \
    --h 4 \
    --total_timesteps 10000 \
    --output_dir models/airl_results

# 출력:
# - models/airl_results/airl_generator_h4.zip
# - models/airl_results/airl_reward_h4.pt
# - models/airl_results/airl_metadata_h4.pkl
```

#### 모든 단계 자동화

```bash
# 모든 depth에 대해 Option B 파이프라인 실행
for h in 1 2 4 8; do
    echo "Processing h=$h..."

    # BFS 데이터
    python3 fourinarow_airl/generate_training_data.py \
        --h $h --num_episodes 100

    # BC 학습
    python3 fourinarow_airl/train_bc.py --h $h --n_epochs 50

    # PPO 생성
    python3 fourinarow_airl/create_ppo_generator.py --h $h

    # AIRL 학습
    python3 fourinarow_airl/train_airl.py \
        --h $h --total_timesteps 10000
done
```

### 6.4 비교 및 평가

```bash
# Option A vs B 비교 (h=4)
python3 compare_option_a_vs_b.py --h 4 --num_episodes 50

# 출력:
# - figures/option_a_vs_b_h4.png
# - 콘솔에 metrics 출력 (KL divergence, win rate, etc.)

# 시각화 생성
python3 visualize_option_difference.py

# 출력:
# - figures/option_a_vs_b_diagram.png
# - figures/option_a_vs_b_training_curves.png
# - figures/reward_network_same.png
```

### 6.5 Quick Start (테스트용)

```bash
# Option A 빠른 테스트
python3 fourinarow_airl/train_airl_pure_nn.py --test

# Option B 빠른 테스트
python3 fourinarow_airl/train_airl.py --test

# 두 테스트 모두 minimal timesteps로 전체 파이프라인 검증
```

---

## 7. 핵심 개념 정리

### Q1: depth h는 어디에 사용되나?

**A: h는 세 가지 다른 의미로 사용됨**

1. **Experiment design**: 각 h={1,2,4,8}마다 별도 AIRL 학습
2. **Option B - BFS 데이터 생성**: BFS(h=4)로 trajectory 생성
3. **Naming/Metadata**: 파일 저장 시 `h4` 태그

**h가 사용되지 않는 곳**:
- ❌ Reward network architecture
- ❌ Observation space
- ❌ AIRL algorithm

### Q2: Reward network는 h마다 다른가?

**A: 아키텍처는 같고, weights는 다름**

```python
# 모든 h에 대해 동일한 아키텍처
for h in [1, 2, 4, 8]:
    reward_net = create_reward_network(env)  # Same architecture

    # 하지만 각각 별도로 학습 (different weights)
    train_airl(h=h, reward_net=reward_net, ...)
```

### Q3: Option A와 B 중 어느 것을 선택?

**A: 이 프로젝트는 Option A를 main approach로 진행합니다** ⭐

| 상황 | 선택 | 이유 |
|------|------|------|
| **Main Experiments** | **Option A** ⭐ | 순수한 IRL 능력 검증, Planning depth 순수 효과 |
| **Baseline 구축** | **Option B** | 빠른 비교 기준, 안정적 결과 |
| **빠른 프로토타이핑** | Option B | BFS 지식 활용으로 빠른 수렴 |
| **Pedestrian 프로젝트 일관성** | Option A | 기존 연구와 동일 접근법 |
| **논문 작성** | 둘 다 | Option A (main) + Option B (baseline) 비교 |

**현재 상태**:
- Option B: 이미 구현됨 (Steps A-E 완료, 71%) → Baseline 확보 ✅
- Option A: 코드 준비 완료 → Main experiments 진행 예정 🔄

### Q4: 왜 Reward에 h를 넣지 않나?

**A: 이론적/실험적 이유**

1. **AIRL 이론**: Reward는 관찰 가능한 정보만 사용해야 함
2. **Identifiability**: h를 넣으면 reward와 planning이 confound됨
3. **연구 질문**: "같은 reward에서 다른 h가 다른 행동을 만드는가?"
   - 이를 답하려면 reward는 h-agnostic해야 함

---

## 8. 트러블슈팅

### 문제 1: "imitation library not found"

```bash
# 해결
conda activate pedestrian_analysis
pip install imitation stable-baselines3 torch
```

### 문제 2: "PPO generator not found"

```bash
# Option B의 경우 순서대로 실행 필요
python3 fourinarow_airl/generate_training_data.py --h 4
python3 fourinarow_airl/train_bc.py --h 4
python3 fourinarow_airl/create_ppo_generator.py --h 4
python3 fourinarow_airl/train_airl.py --h 4
```

### 문제 3: "Expert trajectories too few"

```python
# generate_training_data.py에서 num_episodes 증가
python3 fourinarow_airl/generate_training_data.py \
    --h 4 \
    --num_episodes 200  # 기본 100 → 200
```

### 문제 4: "Training too slow"

```bash
# Option A는 원래 느림 → timesteps 줄이기 (테스트용)
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 4 \
    --total_timesteps 10000  # 50000 → 10000

# 또는 Option B 사용
```

---

## 9. 다음 단계

### 실험 계획

1. **Baseline 구축**
   ```bash
   # Option B로 모든 h 학습 (빠르고 안정적)
   for h in 1 2 4 8; do
       # Option B pipeline
   done
   ```

2. **Pure Learning 검증**
   ```bash
   # Option A로 학습 (시간 오래 걸림)
   for h in 1 2 4 8; do
       python3 fourinarow_airl/train_airl_pure_nn.py \
           --h $h --total_timesteps 100000
   done
   ```

3. **비교 분석**
   ```bash
   # 각 h에 대해 Option A vs B 비교
   for h in 1 2 4 8; do
       python3 compare_option_a_vs_b.py --h $h
   done
   ```

4. **Depth Discrimination**
   - 학습된 h별 policy가 실제로 다른 행동을 하는가?
   - Expert depth를 예측할 수 있는가?

---

## 요약

### 전체 파이프라인 (한눈에)

```
Expert Data → Environment Setup → Generator Choice
                                      ↓
                            ┌─────────┴─────────┐
                            │                   │
                       Option A              Option B
                       (Pure NN)         (BFS Distillation)
                            │                   │
                            └─────────┬─────────┘
                                      ↓
                              Reward Network
                              (depth-agnostic)
                                      ↓
                              AIRL Training
                            (Discriminator ↔ Generator)
                                      ↓
                         Learned Policy + Reward
                                      ↓
                                 Evaluation
```

### 핵심 기억할 점

1. **Reward는 항상 depth-agnostic** (h parameter 없음)
2. **차이는 Generator 초기화 방법** (Random vs BC)
3. **각 h마다 별도 학습** (하지만 같은 아키텍처)
4. **Expert 데이터에 depth 정보 없음** (관찰 불가능)
5. **둘 다 구현됨** (Option A와 B 모두 준비 완료!)

---

**이제 실험을 시작할 준비가 되었습니다!** 🚀
