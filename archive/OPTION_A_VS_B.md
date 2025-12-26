# Option A vs Option B: AIRL Implementation Comparison

## Overview

두 가지 AIRL 구현 방식의 비교 문서입니다.

---

## Option A: Pure Neural Network (Pedestrian 방식)

### 특징
- **Generator**: 순수 neural network, random initialization
- **NO domain knowledge**: van Opheusden BFS 사용 안 함
- **Pure learning**: AIRL reward signal로만 학습
- **Pedestrian 프로젝트**와 유사한 접근

### 장점
✅ **이론적 순수성**: Domain knowledge 없이 reward만으로 학습
✅ **Generalizability**: 특정 domain에 덜 의존적
✅ **Research clarity**: "IRL이 정말 reward를 복원하는가?" 테스트 가능

### 단점
⚠️ **느린 학습**: Warm start 없이 처음부터 학습
⚠️ **더 많은 timesteps 필요**: 50K-100K 권장 (vs 10K for Option B)
⚠️ **불안정할 수 있음**: Random init → 수렴 보장 없음

### 구현 파일
- `fourinarow_airl/create_ppo_generator_pure_nn.py`: Pure NN generator 생성
- `fourinarow_airl/train_airl_pure_nn.py`: Option A AIRL 학습

### 사용법
```bash
# Generator 생성 (테스트)
python3 fourinarow_airl/create_ppo_generator_pure_nn.py --test

# AIRL 학습
python3 fourinarow_airl/train_airl_pure_nn.py --h 4 --total_timesteps 50000
```

---

## Option B: BFS Distillation (현재 기본 구현)

### 특징
- **Generator**: BC로 BFS 행동 모방 → PPO fine-tuning
- **Domain knowledge 활용**: van Opheusden BFS를 warm start로 사용
- **Faster convergence**: BC initialization 덕분에 빠른 수렴

### 장점
✅ **빠른 학습**: 10K timesteps로 충분
✅ **안정적**: BC로 reasonable policy부터 시작
✅ **Van Opheusden 연구 활용**: 기존 domain knowledge 보존

### 단점
⚠️ **Domain-specific**: 4-in-a-row BFS에 의존적
⚠️ **이론적 불순**: IRL이 reward를 학습하는지, BC가 학습한 것인지 ambiguous

### 구현 파일
- `fourinarow_airl/generate_training_data.py`: BFS 데이터 생성
- `fourinarow_airl/train_bc.py`: BC 학습
- `fourinarow_airl/create_ppo_generator.py`: BC → PPO
- `fourinarow_airl/train_airl.py`: Option B AIRL 학습

### 사용법
```bash
# 1. BFS 데이터 생성
python3 fourinarow_airl/generate_training_data.py --h 4 --num_episodes 100

# 2. BC 학습
python3 fourinarow_airl/train_bc.py --h 4

# 3. PPO generator 생성
python3 fourinarow_airl/create_ppo_generator.py --h 4

# 4. AIRL 학습
python3 fourinarow_airl/train_airl.py --h 4 --total_timesteps 10000
```

---

## 공통점

두 옵션 모두:
- **Reward network**: Pure NN, depth-agnostic (h parameter 없음)
- **Observation**: 89-dim (board + features, NO depth encoding)
- **depth h**: 각 h마다 별도 학습 (h는 naming/metadata일 뿐)

---

## depth h를 Reward Network에 포함해야 하나?

### ❌ NO - 포함하지 않는 것이 맞습니다!

#### 이유

1. **AIRL 이론적 원칙**
   - Reward는 (state, action)의 함수여야 함
   - Planning process와 독립적이어야 함

2. **Identifiability**
   - h를 reward에 넣으면 reward와 planning이 confound됨
   - "이 행동이 좋은 이유가 reward 때문인가, planning depth 때문인가?" 구분 불가

3. **Research Question**
   - "같은 reward에서 다른 planning depth가 어떻게 다른 행동을 만드는가?"
   - 이를 테스트하려면 reward는 depth-agnostic해야 함

#### 현재 구현

```python
# ✅ 올바른 방식 (현재 구현)
reward_net = create_reward_network(env)  # NO h parameter!
# Input: (state, action, next_state) → Output: reward

# ❌ 잘못된 방식
reward_net = create_reward_network(env, h=4)  # WRONG!
```

#### depth h의 역할

- **NOT in reward network**: Reward는 h를 모름
- **IN generator**: 각 h마다 별도 generator 학습
  - Option A: h는 naming용 (network는 h 모름)
  - Option B: h로 BFS 생성 → BC로 distill (network는 h 모름)
- **Experiment design**: 각 h={1,2,4,8}마다 별도 AIRL 학습

---

## 비교표

| 항목 | **Option A** | **Option B** | **공통** |
|-----|-------------|-------------|---------|
| **Generator 초기화** | Random | BC(BFS) | - |
| **Domain knowledge** | 없음 | van Opheusden BFS | - |
| **학습 속도** | 느림 (50K-100K) | 빠름 (10K) | - |
| **안정성** | 낮음 | 높음 | - |
| **이론적 순수성** | 높음 | 낮음 | - |
| **Reward network** | Pure NN (depth-agnostic) | Pure NN (depth-agnostic) | ✅ 동일 |
| **Observation** | 89-dim (NO h) | 89-dim (NO h) | ✅ 동일 |
| **depth h 역할** | Naming only | BC 학습 시 사용, 이후 naming | ✅ 둘 다 network에 NO h |

---

## 권장 사항

### 연구 목적에 따라 선택

**Option A를 선택하는 경우:**
- IRL의 순수한 성능을 테스트하고 싶을 때
- Domain knowledge 없이 학습 가능성을 검증할 때
- Pedestrian 프로젝트와 유사한 설정이 필요할 때

**Option B를 선택하는 경우:**
- 빠른 프로토타이핑이 필요할 때
- Van Opheusden 연구 결과를 활용하고 싶을 때
- 안정적인 baseline이 필요할 때

### 실험 설계

**권장: 둘 다 실험하고 비교**

```bash
# Option A 학습
python3 fourinarow_airl/train_airl_pure_nn.py --h 4 --total_timesteps 50000

# Option B 학습
python3 fourinarow_airl/train_airl.py --h 4 --total_timesteps 10000

# 비교: 어느 것이 expert behavior를 더 잘 모방하는가?
```

**비교 metrics:**
1. Expert trajectory와의 KL divergence
2. Win rate
3. Action distribution similarity
4. Training time
5. Sample efficiency

---

## 요약

### depth h를 reward network에 포함?
**❌ NO** - 둘 다 depth-agnostic reward 사용

### Option A vs B 차이점?
**Generator 초기화만 다름**
- Option A: Random init (순수 학습)
- Option B: BC init (warm start)

### 어떤 것을 선택?
**연구 목적에 따라:**
- Pure learning test → Option A
- Practical performance → Option B
- **권장: 둘 다 실험하고 비교**
