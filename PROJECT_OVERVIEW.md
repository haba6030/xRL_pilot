# Planning-Aware AIRL: 연구 개요

**Planning Depth를 명시적으로 모델링하여 인간 의사결정을 이해하고 설명하는 연구**

**Last Updated**: 2025-12-26
**Current Phase**: Phase 2 (71% complete)

---

## 📌 핵심 아이디어

### 전통적인 IRL의 문제점

**Inverse Reinforcement Learning (IRL)**은 행동 데이터로부터 보상 함수를 추론합니다:

```
관찰: 사람들의 행동 데이터 (observations, actions)
질문: "이 사람들은 어떤 보상을 최적화하고 있는가?"
결과: 추론된 보상 함수 r(s, a)
```

**하지만**, 전통적 IRL은 중요한 가정을 합니다:
- ❌ 모든 사람이 같은 방식으로 계획한다
- ❌ Planning depth는 고정되어 있거나 무한하다
- ❌ 행동 차이는 오직 보상 차이에서만 온다

### Planning-Aware AIRL의 해결책

우리는 **Planning Depth (h)**를 명시적으로 모델링합니다:

```
관찰: 사람들의 행동 데이터 (observations, actions)
질문: "이 사람들은 얼마나 깊이 생각하며, 어떤 보상을 최적화하는가?"
결과: Planning depth h + 보상 함수 r(s, a)
```

**핵심 통찰** (Yao et al., 2024):
> Planning horizon은 IRL에서 latent confounder 역할을 합니다.
> 이를 무시하면 **reward identifiability**가 깨집니다.

---

## 🎯 연구 목표

### Objective 1: Planning Depth를 명시적으로 정의

**Planning Depth (h)**: 얼마나 많은 미래 단계를 미리 생각하는가?

```python
h = 1  # 1 step ahead  - 즉각적 반응
h = 2  # 2 steps ahead - 기본 계획
h = 4  # 4 steps ahead - 중급 계획
h = 8  # 8 steps ahead - 전문가 수준
```

**다른 파라미터**:
- **β (inverse temperature)**: 얼마나 결정적으로 선택하는가?
- **lapse rate**: 무작위로 선택할 확률은 얼마나 되는가?

### Objective 2: 초보자 vs 전문가 구분

**연구 질문**: Planning depth h로 전문가를 구분할 수 있는가?

**가설**:
- **H1 (Brute-force)**: 전문가 = 더 깊은 planning (h ↑)
- **H2 (Efficiency)**: 전문가 = 효율적 pruning (h 비슷하지만 더 나은 heuristics)

**Phase 1 발견** (van Opheusden et al., 2023 데이터 재분석):
- ✅ 전문가가 오히려 **더 얕은** planning depth를 보임 (p=0.01)
- ✅ Depth와 성능 간 **부적 상관관계** (r=-0.50)
- ✅ **Efficiency hypothesis 지지**: 전문가 = 효율적 pruning

### Objective 3: IRL 설명력 향상

**표준 AIRL**:
```python
# 모든 행동 차이를 reward로 설명
reward_net = f(observation, action)  # 모든 것을 reward로!
```

**Planning-Aware AIRL**:
```python
# 행동 차이를 planning + reward로 분해
policy = DepthLimitedPolicy(h=h)              # Planning mechanism
reward_net = f(observation, action)           # Reward (NO h!)
```

**기대 효과**:
- ✅ 더 나은 reward identifiability
- ✅ 더 해석 가능한 개인차 설명
- ✅ Out-of-distribution (OOD) generalization 향상

### Objective 4: 임상 특성 설명

**전통적 접근**:
```
불안 장애 → 다른 보상 함수 (예: 위험 회피 ↑)
```

**Planning-Aware 접근**:
```
불안 장애 → Planning mechanism 차이
  - 더 짧은 planning depth? (myopic)
  - 더 높은 lapse rate? (distraction)
  - 다른 feature weighting? (threat bias)
  + 보상 함수 차이
```

**이점**: 메커니즘 기반 설명 → 더 나은 intervention 가능성

### Objective 5: 신경 메커니즘 연결 (탐색적)

**두 가지 접근**:

1. **Model-based fMRI**:
```python
# Trial-wise regressors
value_t = Q(s_t, a_t)              # Value signal
uncertainty_t = H(π(·|s_t))        # Uncertainty
conflict_t = max(Q) - second_max(Q) # Conflict
planning_proxy_t = f(h, depth_t)   # Planning proxy
```

2. **Individual differences**:
```python
# Subject-level parameters → brain activity
h_subject → dmPFC activity?
β_subject → striatum activity?
lapse_subject → attention network?
```

---

## 🧪 방법론: 4-in-a-Row Game

### 왜 4-in-a-Row인가?

**장점**:
- ✅ 충분히 복잡 (planning 필요)
- ✅ 계산 가능 (h=1~8 실현 가능)
- ✅ 잘 정의된 heuristics (van Opheusden et al., 2023)
- ✅ 풍부한 데이터 (67,331 trials, 40 participants)

**게임 설명**:
- 6×6 보드
- 2명의 플레이어 (Black/White)
- 목표: 4개를 연속으로 놓기
- 행동 공간: 36 positions (0-35)

### Model Components

#### 1. Board State

```python
# 89-dimensional observation
board_state = {
    'board': 72,      # 6×6×2 (black/white bitboards)
    'features': 17,   # heuristic features
}
# ⚠️ NO h information in observations!
```

#### 2. Heuristic Evaluation

**17 features** (van Opheusden et al., 2023):
- Center control
- Connected/unconnected 2-in-a-row
- 3-in-a-row
- 4-in-a-row (win)
- Orientation variants (horizontal, vertical, diagonal)

```python
def heuristic_value(state, weights):
    features = extract_features(state)  # 17-dim
    return dot(weights, features)
```

#### 3. Depth-Limited Search

**Best-First Search (BFS)** with fixed depth h:

```python
def depth_limited_search(state, h, weights):
    """
    Search up to depth h
    Returns Q-values for all legal actions
    """
    frontier = PriorityQueue()
    Q = {}

    for action in legal_actions(state):
        next_state = transition(state, action)
        value = heuristic_value(next_state, weights)
        frontier.push(next_state, depth=1, root_action=action)
        Q[action] = value

    while frontier and depth < h:
        node = frontier.pop()
        # Expand and update Q-values
        # ...

    return Q
```

#### 4. Policy

**Softmax policy** with temperature β and lapse rate:

```python
def policy(state, h, beta, lapse, weights):
    Q = depth_limited_search(state, h, weights)

    # Softmax
    pi_soft = softmax(beta * Q)

    # Lapse (random choice)
    pi_uniform = uniform(len(Q))
    pi = (1 - lapse) * pi_soft + lapse * pi_uniform

    return pi
```

---

## 🏗️ Planning-Aware AIRL 구조

### 핵심 원칙 ⭐

```python
# ✅ CRITICAL PRINCIPLE
# Planning depth h는 POLICY에만 존재
# Reward network는 DEPTH-AGNOSTIC

# ✅ CORRECT
policy = DepthLimitedPolicy(h=h)              # h HERE!
reward_net = create_reward_network(env)       # NO h!
observations.shape == (T+1, 89)               # NO h!
```

**왜?**

1. **Theoretical**: Reward는 환경의 속성, Planning은 agent의 속성
2. **Identifiability**: h와 reward를 분리해야 각각 추론 가능
3. **Generalization**: 같은 reward로 다른 h policy 생성 가능

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AIRL Training                          │
│                                                          │
│  ┌──────────────────┐         ┌────────────────────┐   │
│  │  Generator       │         │  Discriminator     │   │
│  │  (h-specific)    │         │  (h-AGNOSTIC)      │   │
│  │                  │         │                    │   │
│  │  Policy π_h      │────────>│  Reward Net r_φ    │   │
│  │  (PPO)           │         │  (NO h param!)     │   │
│  │                  │         │                    │   │
│  │  Input: s (89)   │         │  Input: (s,a,s')   │   │
│  │  Output: a       │         │  Output: reward    │   │
│  └──────────────────┘         └────────────────────┘   │
│          ↑                              ↑               │
│          │                              │               │
│  ┌───────┴──────────┐         ┌────────┴────────────┐  │
│  │ h-specific       │         │ Expert trajectories │  │
│  │ training data    │         │ (NO h labels!)      │  │
│  │ (h=1,2,4,8)      │         │ (s, a, s', done)    │  │
│  └──────────────────┘         └─────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Two Approaches: Option A vs Option B

**이 프로젝트는 Option A (Pure NN)를 main approach로 진행합니다** ⭐

| | Option A (Main) ⭐ | Option B (Baseline) |
|---|---|---|
| **Generator 초기화** | Random weights | BC from BFS |
| **학습 방식** | 순수 AIRL | BC → AIRL fine-tune |
| **Timesteps** | 50K-100K | 10K |
| **장점** | 순수한 planning depth 효과 측정 | 빠르고 안정적 |
| **현재 상태** | 코드 준비 완료 → 실험 예정 | Steps A-E 완료 (71%) |
| **용도** | Main experiments | Baseline/비교군 |

**상세 설명**: [AIRL_COMPLETE_GUIDE.md](docs/AIRL_COMPLETE_GUIDE.md) 참조

### Implementation Pipeline (Option B - Baseline)

**Phase 2 구현** (현재 71% 완료 - Baseline 확보):

#### Step A: h-specific Training Data 생성

```python
# fourinarow_airl/generate_training_data.py

for h in [1, 2, 4, 8]:
    policy = DepthLimitedPolicy(h=h)

    trajectories = []
    for episode in range(num_episodes):
        traj = play_episode(env, policy)
        trajectories.append({
            'observations': traj.obs,   # (T+1, 89) - NO h!
            'actions': traj.acts,       # (T,)
            'h': h,                     # metadata only
        })

    save(trajectories, f'trajectories_h{h}.pkl')
```

**✅ Checkpoint 1**: Observations are 89-dim (NO h)
**✅ Checkpoint 2**: Actions in range [0, 35]
**✅ Checkpoint 3**: 'h' is metadata only

#### Step B: Behavior Cloning (BC)

```python
# fourinarow_airl/train_bc.py

for h in [1, 2, 4, 8]:
    # Load h-specific trajectories
    trajectories = load(f'trajectories_h{h}.pkl')

    # Convert to imitation format (WITHOUT h!)
    imitation_trajs = convert_to_imitation_format(trajectories)

    # Train BC policy (depth-agnostic neural network)
    bc_trainer = BC(
        observation_space=Box(89,),   # NO h!
        action_space=Discrete(36),
        demonstrations=imitation_trajs,
    )
    bc_trainer.train(n_epochs=50)

    save(bc_trainer, f'bc_trainer_h{h}.pkl')
```

**✅ Checkpoint 3**: Convert WITHOUT h
**✅ Checkpoint 4**: BC policy has NO depth-related attributes

#### Step C: BC를 PPO로 래핑

```python
# fourinarow_airl/create_ppo_generator.py

for h in [1, 2, 4, 8]:
    # Load BC policy
    bc_trainer = load(f'bc_trainer_h{h}.pkl')

    # Create PPO with BC initialization
    ppo_model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
    )

    # Copy BC weights to PPO
    ppo_model.policy.load_state_dict(
        bc_trainer.policy.state_dict()
    )

    save(ppo_model, f'ppo_generator_h{h}.zip')
```

**✅ Checkpoint 5**: PPO uses BC policy (depth-agnostic)

#### Step D: Depth-AGNOSTIC Reward Network

```python
# fourinarow_airl/create_reward_net.py

def create_reward_network(env):
    """
    ⚠️ CRITICAL: NO h parameter!
    """
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,  # Box(89,)
        action_space=env.action_space,            # Discrete(36)
        hid_sizes=[64, 64],
    )
    return reward_net  # NO h anywhere!
```

**✅ Checkpoint 6a**: NO h in function signature
**✅ Checkpoint 6b**: NO h in reward network
**✅ Checkpoint 6c**: No depth-related attributes
**✅ Checkpoint 6d**: Forward pass verified

**Technical Note**: `BasicRewardNet` requires two-stage processing:

```python
# Preprocess (handles action one-hot encoding)
state_th, action_th, next_state_th, done_th = reward_net.preprocess(
    obs_tensor,       # (batch, 89) FloatTensor
    action_tensor,    # (batch,) LongTensor - indices!
    next_obs_tensor,  # (batch, 89) FloatTensor
    done_tensor       # (batch,) BoolTensor
)

# Forward pass
reward = reward_net(state_th, action_th, next_state_th, done_th)
```

#### Step E: AIRL Training

```python
# fourinarow_airl/train_airl.py

for h in [1, 2, 4, 8]:
    # Load h-specific generator (BC → PPO)
    gen_algo = PPO.load(f'ppo_generator_h{h}.zip', env=env)

    # Create depth-AGNOSTIC reward network
    reward_net = create_reward_network(env)  # NO h!

    # Load expert trajectories (NO h labels!)
    expert_trajectories = load('expert_trajectories.pkl')

    # AIRL trainer
    trainer = airl.AIRL(
        demonstrations=expert_trajectories,  # NO h labels
        gen_algo=gen_algo,                   # h-specific
        reward_net=reward_net,               # h-AGNOSTIC!
        allow_variable_horizon=True,         # 4-in-a-row games vary
    )

    trainer.train(total_timesteps=50000)

    save(trainer, f'airl_results_h{h}/')
```

**✅ Checkpoint 7a**: Expert trajectories have NO h labels
**✅ Checkpoint 7b**: Generator learned from h-specific policy
**✅ Checkpoint 7c**: Discriminator has NO h parameter

#### Step F: Multi-Depth Comparison (🔄 진행 중)

**목표**: 어떤 h가 expert behavior를 가장 잘 설명하는가?

```python
results = {}
for h in [1, 2, 4, 8]:
    trainer = load(f'airl_results_h{h}/')

    # Evaluation metrics
    results[h] = {
        'disc_acc': trainer.disc_acc,
        'disc_acc_expert': trainer.disc_acc_expert,
        'disc_acc_gen': trainer.disc_acc_gen,
        'imitation_quality': evaluate_imitation(trainer),
        'kl_divergence': compute_kl(trainer, expert_policy),
    }

# Best h = most balanced discriminator
best_h = argmin(abs(results[h]['disc_acc'] - 0.5) for h in [1,2,4,8])
```

**평가 기준**:
- **Discriminator accuracy ≈ 0.5** (generator fools discriminator)
- **Trajectory similarity** (Euclidean distance, DTW)
- **Action distribution KL** (behavioral realism)

#### Step G: 평가 및 분석 (📋 계획)

**분석**:
1. Best h 식별
2. Learned reward 시각화
3. Policy comparison (h=1 vs h=8)
4. Generalization test (OOD states)

---

## 📊 AIRL Training Metrics 이해하기

### Discriminator Metrics

**3가지 주요 지표**:
```python
disc_acc         # Overall discriminator accuracy
disc_acc_expert  # Accuracy on expert data (should be ~0.5)
disc_acc_gen     # Accuracy on generated data (should be ~0.5)
```

### Training Progression

```
┌─────────────────────────────────────────────────────────────┐
│ Training Stage 1: Discriminator Too Strong (Undertrained)  │
├─────────────────────────────────────────────────────────────┤
│ disc_acc = 0.5            # Overall looks OK...             │
│ disc_acc_expert = 1.0     # BUT: Discriminator too strong!  │
│ disc_acc_gen = 0.0        # Generator can't fool it         │
│                                                             │
│ Interpretation: Need more training!                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Training Stage 2: Balanced (Well-Trained) ✅                │
├─────────────────────────────────────────────────────────────┤
│ disc_acc ≈ 0.5            # Overall balanced                │
│ disc_acc_expert ≈ 0.5     # Generator fools discriminator!  │
│ disc_acc_gen ≈ 0.5        # Good imitation quality          │
│                                                             │
│ Interpretation: AIRL converged! ✅                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Training Stage 3: Generator Too Strong (Overtrained)        │
├─────────────────────────────────────────────────────────────┤
│ disc_acc = 0.5            # Overall balanced                │
│ disc_acc_expert = 0.0     # Generator dominates             │
│ disc_acc_gen = 1.0        # Discriminator too weak          │
│                                                             │
│ Interpretation: Possible mode collapse                      │
└─────────────────────────────────────────────────────────────┘
```

**⚠️ Common Mistake**:
- ❌ "disc_acc_expert=1.0 means good!" (NO!)
- ✅ "All metrics ≈ 0.5 means good!" (YES!)

---

## 🔬 기대 결과 및 Contributions

### Expected Results

**RQ1**: Planning depth로 expertise 구분 가능한가?
- **예상**: Moderate correlation (r=0.3-0.5)
- **Phase 1 발견**: Negative correlation (expertise = shallower depth)

**RQ2**: Planning-Aware AIRL이 standard AIRL보다 나은가?
- **예상**: ✅ 더 나은 reward identifiability
- **예상**: ✅ 더 나은 OOD generalization
- **예상**: ✅ 더 해석 가능한 individual differences

**RQ3**: 최적 h는 무엇인가?
- **예상**: h=2~4 (intermediate planning)
- **검증 방법**: Discriminator balance, imitation quality

### Contributions

**1. Theoretical**:
- Planning을 latent confounder로 명시적 모델링
- Reward identifiability 향상 방법 제시

**2. Methodological**:
- Planning-Aware AIRL framework 구현
- Multi-depth comparison protocol

**3. Empirical**:
- 4-in-a-row 데이터에서 검증
- Expertise-planning relationship 규명

**4. Clinical**:
- Planning mechanism 기반 individual differences 설명
- Future: 임상 특성 예측 가능성

---

## 📚 주요 참고문헌

### Core Papers

1. **van Opheusden, B., Acerbi, L., & Ma, W. J. (2023)**. "Expertise increases planning depth in human gameplay". *Nature*, 618, 1000-1005.
   - https://www.nature.com/articles/s41586-023-06124-2
   - **기여**: Planning depth와 expertise 관계, 4-in-a-row 데이터/모델

2. **Yao, W., Chen, B., & Dragan, A. D. (2024)**. "Planning horizon as a latent confounder in inverse reinforcement learning". *arXiv preprint arXiv:2409.18051*.
   - https://arxiv.org/abs/2409.18051
   - **기여**: Planning horizon이 IRL에서 confounder 역할 증명

3. **Mhammedi, Z., Helou, D., & Gretton, A. (2023)**. "Reinforcement learning for multi-step inverse kinematics". *arXiv preprint arXiv:2304.05889*.
   - https://arxiv.org/abs/2304.05889
   - **기여**: Multi-step factor를 explicit하게 모델링

4. **Fu, J., Luo, K., & Levine, S. (2018)**. "Learning robust rewards with adversarial inverse reinforcement learning". *ICLR 2018*.
   - **기여**: AIRL 알고리즘 (MaxEnt IRL + GAN)

### Related Work

- **Ng, A. Y., & Russell, S. J. (2000)**. "Algorithms for inverse reinforcement learning". *ICML 2000*.
- **Ziebart, B. D., et al. (2008)**. "Maximum entropy inverse reinforcement learning". *AAAI 2008*.
- **Ho, J., & Ermon, S. (2016)**. "Generative adversarial imitation learning". *NeurIPS 2016*.

---

## 🛠️ 기술 스택

| 카테고리 | 도구/라이브러리 | 버전 | 용도 |
|----------|----------------|------|------|
| **언어** | Python | 3.9.7 | 메인 구현 |
| **환경** | Conda | - | 패키지 관리 (pedestrian_analysis) |
| **RL** | stable-baselines3 | latest | PPO 구현 |
| **IRL** | imitation | 1.0.1 | BC, AIRL 구현 |
| **DL** | PyTorch | latest | 신경망 학습 |
| **Data** | NumPy, Pandas | latest | 데이터 처리 |
| **Viz** | Matplotlib, Seaborn | latest | 시각화 |
| **Game** | Custom dm_env | - | 4-in-a-row 환경 |

### 환경 설정

```bash
# Conda 환경
conda activate pedestrian_analysis

# OpenMP 충돌 해결 (macOS)
export KMP_DUPLICATE_LIB_OK=TRUE

# Working directory
cd /Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/fourinarow_airl
```

---

## 🚀 빠른 시작 (Lab Members)

### 1. 환경 설정

```bash
# Repository clone
git clone [repository-url]
cd xRL_pilot

# Conda 환경 활성화
conda activate pedestrian_analysis

# OpenMP workaround (macOS only)
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 2. 문서 읽기 순서

**시작**:
1. [README.md](README.md) - 프로젝트 소개
2. [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (이 파일) - 연구 배경

**Phase 2 이해**:
3. [AIRL_DESIGN.md](docs/AIRL_DESIGN.md) - 설계 문서
4. [AIRL_COMPLETE_GUIDE.md](docs/AIRL_COMPLETE_GUIDE.md) - 실행 가이드 ⭐
5. [PHASE2_PROGRESS.md](progress/PHASE2_PROGRESS.md) - 현재 진행 상황

**구현 세부사항**:
6. [IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) - 기술 참고사항
7. [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) - 구현 요약

### 3. 코드 실행

```bash
cd fourinarow_airl

# Step A: Training data 생성
python3 generate_training_data.py --num_episodes 100

# Step B: BC 학습
python3 train_bc.py --n_epochs 50

# Step C: PPO generator 생성
python3 create_ppo_generator.py

# Step D: Reward network 테스트
python3 create_reward_net.py --test

# Step E: AIRL 학습
python3 train_airl.py --total_timesteps 50000
```

### 4. 결과 확인

```bash
# Training trajectories
ls data/training_trajectories/

# BC policies
ls models/bc_policies/

# PPO generators
ls models/ppo_generators/

# AIRL results
ls models/airl_results/
```

---

## 📝 다음 단계 (Step F)

**현재 상태**: Steps A-E 완료 (71%)

**다음 작업**: Multi-Depth Comparison

```bash
# Train AIRL for all depths with sufficient timesteps
python3 train_airl.py --total_timesteps 100000
```

**분석 계획**:
1. Compare discriminator metrics across h ∈ {1, 2, 4, 8}
2. Evaluate imitation quality
3. Identify best h for expert behavior
4. Visualize learned rewards

**연구 질문**: "어떤 planning depth가 expert behavior를 가장 잘 설명하는가?"

---

## 🤝 연구진 및 기여

**소속**: [Lab Name]
**PI**: [PI Name]
**연구원**: [Researcher Names]

**기여 방법**:
- 이슈 제기: GitHub Issues
- 코드 리뷰: Pull Requests
- 문서 개선: [DOCUMENTATION_QUALITY_REVIEW.md](progress/DOCUMENTATION_QUALITY_REVIEW.md) 참조

---

## 📧 문의

- **이메일**: [contact email]
- **GitHub**: [repository URL]
- **Lab Website**: [lab website]

---

**Last Updated**: 2025-12-26
**Document Version**: 1.0
**Status**: Phase 2 (71% complete)
