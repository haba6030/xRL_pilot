# Option A에서 depth h는 어떻게 사용되나?

## TL;DR (핵심 답변)

**Option A에서 depth h는 주로 "실험 설계 변수"입니다.**

```python
# h는 네트워크에 들어가지 않음!
# 단지 어떤 expert data를 사용할지 결정

for h in [1, 2, 4, 8]:
    # 1. Expert data: BFS(h)로 생성한 데이터 선택
    expert_data = load_expert_trajectories_from_BFS(h=h)

    # 2. Generator: 랜덤 초기화 (h 사용 안 함!)
    gen = PPO("MlpPolicy", env)  # NO h parameter!

    # 3. Reward network: depth-agnostic (h 사용 안 함!)
    reward = BasicRewardNet(obs_space, action_space)  # NO h!

    # 4. AIRL 학습
    train_airl(expert_data, gen, reward)

    # 5. 저장: h는 파일명에만 사용
    save(gen, f"airl_pure_generator_h{h}.zip")
```

**핵심**: h는 **어떤 expert를 모방할지** 결정하는 레이블일 뿐, 네트워크 구조와는 무관!

---

## 상세 설명

### 시나리오 1: 각 h마다 다른 Expert 사용

**가장 일반적인 사용법**

```python
# Synthetic expert data 생성
for h in [1, 2, 4, 8]:
    # BFS(h)로 "expert" 생성
    expert_trajs = generate_depth_limited_trajectories(h=h, num_episodes=100)

    # 이 expert를 모방하는 pure NN policy 학습
    train_airl_pure_nn(h=h, expert_trajs=expert_trajs)
```

**h의 역할**:
- BFS(h=1)로 생성한 expert와 BFS(h=8)로 생성한 expert는 **행동이 다름**
- Option A는 각각의 expert를 모방하려고 시도
- 네트워크는 h를 모르지만, **학습 데이터가 다르므로** 결과가 달라짐

**실험 질문**:
- "Pure NN이 h=1 expert vs h=8 expert를 구분할 수 있나?"
- "같은 아키텍처로 다른 planning depth expert를 모방할 수 있나?"

---

### 시나리오 2: 같은 Expert, 다른 해석

**고급 사용법** (이론적)

```python
# Human expert data (depth 알 수 없음)
human_expert_trajs = load_expert_trajectories('opendata/raw_data.csv')

for h in [1, 2, 4, 8]:
    # 가설: "이 human expert가 depth h로 플레이했다"
    # Option A: 같은 데이터로 학습, h는 단지 실험 ID
    train_airl_pure_nn(
        h=h,  # Experiment ID
        expert_trajs=human_expert_trajs  # Same data!
    )
```

**h의 역할**:
- 각 h는 **독립적인 실험**을 의미
- 같은 expert를 모방하지만, **다른 랜덤 seed**로 초기화
- 여러 번 학습해서 **분산(variance)** 측정

**실험 질문**:
- "같은 expert를 모방할 때 초기화에 따라 얼마나 다른 policy가 나오나?"
- "Pure NN의 안정성은?"

---

## Option A vs Option B 비교 (h 사용)

### Option A: h는 "어떤 expert를 모방할지" 레이블

```python
# Option A 구조
for h in [1, 2, 4, 8]:
    # h에 따라 다른 expert data
    expert_data = generate_BFS_data(h=h)  # ← h 사용!

    # Generator: h 모름
    gen = PPO("MlpPolicy", env)  # NO h

    # Reward: h 모름
    reward = BasicRewardNet(...)  # NO h

    # 학습
    AIRL(expert_data, gen, reward)

    # 결과: h별로 다른 policy (다른 expert 때문)
```

### Option B: h는 "BFS 초기화 + 어떤 expert" 레이블

```python
# Option B 구조
for h in [1, 2, 4, 8]:
    # 1. h에 따라 BFS(h) 데이터 생성
    bfs_data = generate_BFS_data(h=h)  # ← h 사용!

    # 2. BC로 BFS(h) 모방
    bc_policy = BC(bfs_data)  # ← h의 영향 받음

    # 3. PPO로 래핑 (BC policy 상속)
    gen = PPO.from_bc(bc_policy)  # ← 간접적으로 h 영향

    # 4. Reward: h 모름
    reward = BasicRewardNet(...)  # NO h

    # 5. Expert data (보통 BFS(h)와 같음)
    expert_data = bfs_data

    # 6. AIRL fine-tuning
    AIRL(expert_data, gen, reward)

    # 결과: h별로 다른 policy (BFS 초기화 + 다른 expert)
```

---

## 구체적 예시

### 예시 1: h=2 vs h=8 expert 모방 (Option A)

```python
# h=2 experiment
expert_h2 = generate_BFS_trajectories(h=2, episodes=100)
# → 얕은 계획, 단기적 이익 추구, 빠른 결정

gen_h2 = PPO("MlpPolicy", env)  # Random init
reward_h2 = BasicRewardNet(...)

train_airl(expert_h2, gen_h2, reward_h2)
# 결과: gen_h2는 얕은 계획 행동 학습

# h=8 experiment
expert_h8 = generate_BFS_trajectories(h=8, episodes=100)
# → 깊은 계획, 장기적 전략, 느린 결정

gen_h8 = PPO("MlpPolicy", env)  # Random init (다른 seed)
reward_h8 = BasicRewardNet(...)  # Fresh instance

train_airl(expert_h8, gen_h8, reward_h8)
# 결과: gen_h8는 깊은 계획 행동 학습
```

**차이점**:
- `gen_h2`와 `gen_h8`는 **같은 아키텍처**
- 하지만 **다른 expert data**로 학습
- 네트워크는 h를 모르지만, **행동 패턴이 달라짐**

---

### 예시 2: 실제 코드 흐름

```python
# fourinarow_airl/train_airl_pure_nn.py의 실제 사용

# 사용자가 h=4를 선택
h = 4

# Step 1: Expert data (h=4 BFS로 생성)
from generate_training_data import generate_depth_limited_trajectories

expert_trajs = generate_depth_limited_trajectories(
    h=4,  # ← h 사용! (expert 생성)
    num_episodes=100
)
# 이 데이터는 "depth 4로 계획한 행동"

# Step 2: Generator (h 모름!)
from create_ppo_generator_pure_nn import create_pure_ppo_generator

gen_algo, venv = create_pure_ppo_generator(
    env=env,
    h=4,  # ← Naming only! (네트워크는 h 모름)
    learning_rate=3e-4
)
# gen_algo는 랜덤 초기화, h 정보 없음

# Step 3: Reward (h 모름!)
from create_reward_net import create_reward_network

reward_net = create_reward_network(env)  # NO h parameter!

# Step 4: AIRL 학습
train_airl(
    h=4,              # ← Metadata/logging
    expert_trajs,     # ← h=4 BFS data
    gen_algo,         # ← h 모름
    reward_net        # ← h 모름
)

# Step 5: 저장
gen_algo.save('models/airl_pure_nn_results/airl_pure_generator_h4.zip')
#                                                                  ^^
#                                                              h는 파일명에만!
```

---

## h의 세 가지 의미 정리

### 1. **Expert 생성 시 사용** (Option A & B 공통)

```python
# h는 BFS 알고리즘의 실제 파라미터
expert_data = generate_BFS_trajectories(h=4)
# BFS가 depth=4까지 탐색 → 특정 행동 패턴 생성
```

### 2. **실험 조직화** (Option A & B 공통)

```python
# h는 실험 버전 관리
experiments = {
    'h1': train_airl(h=1, expert_h1, ...),
    'h2': train_airl(h=2, expert_h2, ...),
    'h4': train_airl(h=4, expert_h4, ...),
    'h8': train_airl(h=8, expert_h8, ...),
}

# 나중에 비교
compare(experiments['h1'], experiments['h8'])
```

### 3. **파일 naming** (Option A & B 공통)

```python
# h는 파일 저장/로드 시 구분자
models/
├── airl_pure_nn_results/
│   ├── airl_pure_generator_h1.zip  # h=1 실험 결과
│   ├── airl_pure_generator_h2.zip  # h=2 실험 결과
│   ├── airl_pure_generator_h4.zip  # h=4 실험 결과
│   └── airl_pure_generator_h8.zip  # h=8 실험 결과
```

---

## 왜 각 h마다 따로 학습하나?

### 연구 질문

**"Planning depth가 행동에 어떤 영향을 주는가?"**

```python
# 가설
# h=1 expert: 근시안적 (myopic) - 즉각적 이득만 고려
# h=8 expert: 전략적 (strategic) - 장기적 결과 고려

# 실험
# 1. 각 h별로 expert 생성
expert_h1 = BFS(h=1).play_games(100)
expert_h8 = BFS(h=8).play_games(100)

# 2. Pure NN이 각각 모방 가능한가?
policy_h1 = train_airl_pure(expert_h1)  # NN이 h=1 행동 학습
policy_h8 = train_airl_pure(expert_h8)  # NN이 h=8 행동 학습

# 3. 결과 비교
# - policy_h1과 policy_h8가 다른 행동을 하나?
# - 어떤 차이가 있나? (win rate, action distribution, etc.)
# - Pure NN이 planning depth를 "간접적으로" 학습했나?
```

### 예상 결과

```python
# Policy h1 행동 패턴
policy_h1.play_game():
    # 짧은 trajectory (빨리 끝남)
    # 공격적 (즉각적 승리 추구)
    # 수비 약함 (미래 위협 무시)

# Policy h8 행동 패턴
policy_h8.play_game():
    # 긴 trajectory (신중하게 플레이)
    # 전략적 (함정 설치)
    # 수비 강함 (장기 전략)
```

**핵심 인사이트**:
- 네트워크는 h를 모름
- 하지만 **다른 expert의 행동 패턴**을 학습
- 결과적으로 **간접적으로 planning depth를 표현**

---

## Option A에서 h를 사용하는 실제 워크플로우

### 워크플로우 1: 모든 h 실험

```bash
# h=1 실험
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 1 \
    --expert_data synthetic \  # BFS(h=1) 생성
    --total_timesteps 50000

# h=2 실험
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 2 \
    --expert_data synthetic \  # BFS(h=2) 생성
    --total_timesteps 50000

# h=4 실험
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 4 \
    --expert_data synthetic \  # BFS(h=4) 생성
    --total_timesteps 50000

# h=8 실험
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 8 \
    --expert_data synthetic \  # BFS(h=8) 생성
    --total_timesteps 50000
```

### 워크플로우 2: Human expert에 대한 가설 검증

```bash
# 같은 human expert data로 여러 실험
# (각 h는 독립적 실험 ID)

for h in 1 2 4 8; do
    python3 fourinarow_airl/train_airl_pure_nn.py \
        --h $h \
        --expert_data opendata/raw_data.csv \  # Same data!
        --total_timesteps 50000 \
        --seed $((42 + h))  # Different random seed
done
```

---

## 핵심 요약

### Option A에서 depth h는:

1. ✅ **Expert 데이터 생성 시 사용**
   - `BFS(h=4)`로 trajectory 생성
   - 각 h마다 다른 행동 패턴

2. ✅ **실험 조직화**
   - 각 h는 독립적인 AIRL 실험
   - 나중에 비교 분석

3. ✅ **파일 naming**
   - `airl_pure_generator_h4.zip`
   - 구분 및 추적 용이

4. ❌ **네트워크 입력으로 사용 안 함**
   - Generator (MLP): NO h
   - Reward network: NO h
   - Observation: 89-dim (NO h)

### 핵심 원칙

> **"h는 어떤 expert를 모방할지 결정하는 레이블이지, 네트워크가 보는 정보가 아니다."**

```python
# 이렇게 이해하면 됨
Option A:
    Expert(h=4) → [Pure NN learns] → Policy(mimics h=4 behavior)
                   ↑
                   h는 여기에만!
                   네트워크는 h 모름!
```

---

## 실험 설계 팁

### 좋은 실험 설계

```python
# 각 h마다 충분한 expert data
for h in [1, 2, 4, 8]:
    expert_data[h] = generate_BFS_trajectories(h=h, episodes=200)

    # Pure NN 학습
    policy[h] = train_airl_pure(expert_data[h], timesteps=100000)

# 비교 분석
for h1, h2 in [(1,2), (2,4), (4,8)]:
    compare_policies(policy[h1], policy[h2])
    # → "h 증가에 따라 행동이 어떻게 변하는가?"
```

### 나쁜 실험 설계

```python
# ❌ 잘못된 예: h를 네트워크에 넣으려고 시도
policy = train_airl_pure(
    expert_data,
    network_with_h_input=True  # WRONG!
)
# 이러면 reward가 h-dependent가 되어 이론적 문제 발생
```

---

이제 명확하신가요? Option A에서 h는 **실험 설계 변수**이지, **네트워크 파라미터가 아닙니다**! 🎯
