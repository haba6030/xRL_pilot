# Planning-Aware IRL 연구 논의 기록

**작성일:** 2024-12-23
**목적:** 연구 설계 과정에서의 핵심 논의 및 결정사항 기록

---

## 1. Depth_by_session.txt 정체 확인

### 결론
- **BFS PV depth의 평균값**
- 각 값 = (모든 trials의 PV depth 합) / (N × trial 수), N=10
- 생성: `compute_planning_depth.cpp` → `get_depth_of_pv()`

### 통계
```
Shape: (30 participants, 5 sessions)
Raw mean: 6.73 ± 1.52
Range: 2.74 ~ 10.37

After "-2" correction (learning.ipynb):
Corrected mean: 4.73 ± 1.52
Range: 0.74 ~ 8.37
```

### "-2" 보정의 미해결 문제
**Learning.ipynb Cell 36:**
```python
pv_array = np.maximum(np.loadtxt(...) - 2, 0)
```

**가능한 설명:**
1. 경험적 조정 (empirical calibration)
2. Root node + 첫 move 제외 (이론적 근거 불명확)
3. 초기 코드 버그의 잔존 보정

**⚠️ TODO:** C++ 바이너리로 재계산하여 검증 필요

---

## 2. 순환 논리 문제 및 해결 방안

### 문제: 현재 Expertise 분류의 순환성

**Current approach (data_reanalysis.py):**
```python
# 1. Parameters로 expertise 분류
expertise_features = ['log-likelihood', 'lapse rate', 'pruning threshold']
expertise_score = Z-score(features).mean()
expertise_label = (expertise_score > median)

# 2. 같은 parameters를 다시 비교
expert_ll = data[expertise_label==1]['log-likelihood']
novice_ll = data[expertise_label==0]['log-likelihood']
# → 당연히 expert_ll > novice_ll (이걸로 분류했으니!)
```

**Circular logic:**
```
Parameters → Expertise Label → Compare Parameters (순환!)
```

### 해결: Elo 기반 독립적 분류 ✅

**독립적인 성과 지표 사용:**
```python
# 1. Elo rating 계산 (게임 승패만 사용, parameters 무관)
elo_ratings = calculate_bayesian_elo(games)

# 2. Elo 기준 expertise 분류
expertise_label = (elo_ratings > median)

# 3. 이제 parameters 비교가 의미 있음
expert_params = model_fits[expertise_label == 1]
novice_params = model_fits[expertise_label == 0]
# → "높은 Elo를 가진 사람이 낮은 lapse rate를 가진다" (인과적 해석 가능)
```

**장점:**
- **독립성**: Elo는 게임 결과만으로 계산
- **객관성**: Bayesian Elo는 통계적으로 robust
- **인과 추론**: Elo → Parameters 방향의 관계 주장 가능

**구현:** `elo_based_classification.py` 생성
**TODO:** learning.ipynb의 실제 Bayesian Elo ratings 로드 필요

---

## 3. Fixed Depth의 필요성: Post-hoc vs Generative

### 핵심 질문
> "그냥 지금의 BFS를 한 다음 pv_depth로 분석하는 것과 무엇이 다를까?"

### 답변: 근본적으로 다름 ✅

#### A. Post-hoc Analysis (Van Opheusden 2023)

**현재 접근:**
```python
# 1. BFS 실행 (gamma로 간접 제어)
game_tree = BFS(board, gamma=0.05, pruning_thresh=50, lapse=0.1)

# 2. PV depth 측정 (결과)
pv_depth = game_tree.get_depth_of_pv()  # 예: 5.2

# 3. 통계적 비교
t_test(expert_pv_depth, novice_pv_depth)
# Result: Expert < Novice (p=0.011)

# 문제: 왜?
# - gamma가 다르기 때문?
# - pruning_thresh가 다르기 때문?
# - 진짜 "planning depth"가 다르기 때문?
# → 분리 불가능 (confounded)
```

**특징:**
- PV depth는 **결과물 (output)**
- Planning은 **관찰 (measurement)**
- 인과 관계 불명확

#### B. Generative Model (제안)

**제안 접근:**
```python
# 1. h를 explicit parameter로 설정
for h in [2, 4, 6, 8, 10]:
    # 2. h 고정, 나머지 fitting
    params_h = fit_model(data, h)  # optimize [beta, lapse] only
    loglik_h[h] = compute_loglik(data, h, params_h)

# 3. Model selection (h 추론!)
h_optimal = argmax(loglik_h - AIC_penalty)

# 4. 이제 인과 관계 명확
# "이 참가자의 planning depth는 h=4"
# → h가 행동의 원인

# 5. 예측 가능
new_behavior = model_h4.generate(new_board, params_h4)
```

**특징:**
- h는 **원인 (latent variable)**
- Planning은 **과정 (generative process)**
- 추론 및 예측 가능

#### C. 비교표

| 측면 | Post-hoc (현재) | Generative (제안) |
|------|----------------|-------------------|
| **Planning depth** | 결과물 (측정) | 원인 (파라미터) |
| **모델링** | PV depth = f(gamma, pruning, ...) | Behavior = f(h, beta, lapse) |
| **추론** | 불가능 | h를 데이터로부터 추론 |
| **예측** | 불가능 | h로 새 행동 예측 |
| **해석** | Correlation | Causation (주장 가능) |
| **통계** | t-test on measurements | Model selection on h |

### Mhammedi (2023) 논문 주장

**핵심:**
> "Planning은 단순한 결과 통계가 아니라,
> 정책을 생성하는 과정(process-level latent)으로 모델링되어야 한다."

**적용:**
- Van Opheusden: PV depth는 "결과 통계" (post-hoc)
- 제안: h는 "process-level latent" (generative)

---

## 4. Fixed Depth 범위 수정 ⚠️

### 원래 제안
```python
h ∈ {1, 2, 3, 4, 5}
```

### 수정된 제안 ✅
```python
h ∈ {2, 4, 6, 8, 10}
```

**근거:**
- depth_by_session.txt 평균: **6.73 ± 1.52**
- 범위: 2.74 ~ 10.37
- "-2" 보정 후: 4.73 ± 1.52 (0.74 ~ 8.37)

**이유:**
1. **데이터 범위 반영**: 실제 PV depth가 6~8 수준
2. **해상도**: {1,2,3,4,5}는 너무 낮음 → 변별력 부족
3. **균등 간격**: 2 단위 증가 (분석 용이)

**비교:**
```
Original proposal: {1, 2, 3, 4, 5}
  - Range: 1~5
  - Mean: 3
  - Issue: 실제 depth (6~8)보다 훨씬 낮음

Revised proposal: {2, 4, 6, 8, 10}
  - Range: 2~10
  - Mean: 6
  - Better: 실제 depth 범위 포괄
```

---

## 5. AIRL의 문제점 및 미해결 이슈 ⚠️⚠️

### A. AIRL Discriminator 활용 방법 미제시

**AIRL 구조:**
```python
# Discriminator (reward estimator)
D(s, a) = exp(f(s,a)) / (exp(f(s,a)) + π(a|s))
# f(s,a) = r(s,a) + γV(s') - V(s)  (reward + shaping)

# Generator (policy)
π(a|s) = BFS policy

# Training
# - Discriminator: 구분 (expert vs generated)
# - Generator: Discriminator 속이기
```

**문제:**
> "AIRL의 핵심은 discriminator인데 이를 활용하는 방법이 미제시"

**미해결:**
1. **Discriminator architecture**: 어떻게 설계?
   - Input: Board state features?
   - Output: Reward estimate?
   - Van Opheusden의 17 features 사용?

2. **Training procedure**: 어떻게 학습?
   - Alternating training (D, G)?
   - Fixed-h policy를 어떻게 update?
   - BFS parameters (beta, lapse) 학습 방법?

3. **Evaluation**: 어떻게 평가?
   - Discriminator accuracy?
   - Policy performance?
   - Reward recovery?

### B. AIRL과 Van Opheusden 구조의 부적합성

**근본적 충돌:**

**AIRL 가정:**
```python
# Reward를 명시적으로 주지 않음
# Network가 reward 학습
reward_network = NeuralNetwork(state, action)
reward = reward_network(s, a)  # Learned!
```

**Van Opheusden 구조:**
```python
# Reward를 명시적으로 정의
heuristic_value = sum([w_i * feature_i for i in range(17)])
# 17 feature weights는 고정 (BADS로 fitting)
# Feature는 domain-specific (2-in-a-row, 3-in-a-row, ...)
```

**충돌:**
> "AIRL은 reward를 명시적으로 안 주는데,
> van Opheusden 구조를 활용하는 것의 적절함 의문"

### C. 가능한 해결 방안 (미검증)

#### **Option 1: Hybrid Approach**
```python
# Van Opheusden heuristic을 reward network initialization으로
reward_network = NeuralNetwork(state, action)
# Initialize with 17 features
reward_network.initialize_with_heuristic(w_act, w_pass)

# AIRL training으로 fine-tune
for epoch in range(epochs):
    # Discriminator update
    loss_D = train_discriminator(expert_data, generated_data, reward_network)

    # Policy (BFS) update
    loss_G = train_policy(reward_network)  # How?
```

**문제:**
- BFS policy를 어떻게 update? (gradient-based가 아님)
- Heuristic initialization이 AIRL에 도움이 되는가?

#### **Option 2: Feature-based Discriminator**
```python
# Van Opheusden features를 discriminator input으로
discriminator = Discriminator(
    input_features=17,  # Van Opheusden features
    output='reward'
)

# Training
for (s, a) in expert_data:
    features = extract_features(s, a)  # 17-dim
    reward = discriminator(features)
```

**문제:**
- Features는 hand-crafted (AIRL의 철학과 맞지 않음)
- Feature selection bias

#### **Option 3: Pure AIRL (Van Opheusden 버림)**
```python
# 완전히 새로운 learned reward
reward_network = NeuralNetwork(state_representation, action)
# No heuristic, no features

# AIRL training
train_airl(expert_data, reward_network, policy_network)
```

**문제:**
- Van Opheusden의 기여 (17 features, BFS) 활용 못함
- 4-in-a-row domain knowledge 버림
- 데이터 부족 문제 (40명 참가자)

### D. 미해결 질문

**Q1: AIRL을 왜 써야 하는가?**
- Reward identifiability? → 이미 heuristic이 있는데?
- Generalization? → BFS도 generalize 가능
- Counterfactual? → Fixed-h BFS만으로도 가능

**Q2: Planning-aware AIRL의 구체적 알고리즘은?**
```python
# Pseudo-code 필요:
# 1. Discriminator 설계
# 2. h-constrained policy 학습 방법
# 3. Alternating training procedure
# 4. Evaluation metric
```

**Q3: 데이터 규모가 충분한가?**
- Expert data: 40명 × ~1000 trials = 40K trials
- AIRL은 data-hungry (특히 high-dim state)
- 4-in-a-row state space: 2^36 (huge!)

---

## 6. 연구 기여의 재정의

### Van Opheusden (2023) 기여
- **발견**: Expertise ↔ Planning depth 관계
- **방법**: BFS + 17 heuristic features
- **측정**: PV depth (post-hoc)
- **한계**: Planning이 결과물 (confounded)

### 제안 연구의 차별점 (수정)

#### Phase 1: Fixed-h BFS (명확함) ✅

**목표:**
- Planning depth를 explicit parameter (h)로 모델링
- h를 데이터로부터 추론 (model selection)
- Expertise와 h의 관계 (generative model)

**방법:**
```cpp
// C++ 구현
class heuristic_fixed_h : public heuristic {
    int fixed_depth;  // h ∈ {2, 4, 6, 8, 10}

    zet makemove_bfs(board b, bool player) override {
        while(n->depth <= fixed_depth && ...) {
            // BFS with depth limit
        }
    }
};
```

```python
# Python fitting
for h in [2, 4, 6, 8, 10]:
    params_h = fit_model(data, h)  # optimize [beta, lapse]
    loglik_h[h] = compute_loglik(data, h, params_h)

h_optimal = argmax(loglik_h - AIC_penalty)
```

**기여:**
- **Post-hoc → Generative**: PV depth 측정 → h로 행동 생성
- **Correlation → Model selection**: h 추론 가능
- **Mhammedi 주장 실현**: Planning as process-level latent

**차별점:**
- Van Opheusden: PV depth는 gamma, pruning의 부산물
- 제안: h는 독립적인 generative parameter

#### Phase 2: Planning-aware AIRL (불명확함) ⚠️

**의도된 목표:**
- Reward와 Planning 분리
- h를 latent confounder로 모델링 (Yao 2024)
- Counterfactual analysis

**현재 문제:**
1. **Discriminator 활용 방법 미제시**
2. **AIRL과 Van Opheusden 구조 충돌**
3. **구체적 알고리즘 부재**
4. **데이터 규모 의문**

**⚠️ 결정 필요:**
- Phase 2를 진행할 것인가?
- 진행한다면 어떤 방식으로?
  - Option 1: Hybrid (heuristic initialization)
  - Option 2: Feature-based discriminator
  - Option 3: Pure AIRL (heuristic 버림)
- 아니면 Phase 1만으로도 충분한 기여인가?

---

## 7. 제안: 단계적 접근

### 우선순위 1: Fixed-h BFS (즉시 실행) ✅

**명확한 기여:**
- Planning depth를 generative parameter로
- h 추론 및 model selection
- Expertise와 h의 관계 (Elo 기반)

**구현:**
1. C++ 코드 수정 (`heuristic_fixed_h.cpp`)
2. MATLAB/Python fitting pipeline
3. Elo 기반 expertise 분류
4. h 분포 비교 (Expert vs Novice)

**예상 결과:**
```python
Expert h distribution: {2: 5, 4: 12, 6: 3}  # 주로 h=4
Novice h distribution: {4: 3, 6: 7, 8: 8, 10: 2}  # 분산됨

# Chi-square test: p < 0.001
# Expert는 낮은 h 선호 (efficient planning)
```

**논문:**
- Title: "Planning Depth as Generative Parameter: A Model Selection Approach to Expertise in Strategic Games"
- Contribution: Post-hoc → Generative modeling
- Impact: Behavioral modeling, cognitive science

### 우선순위 2: AIRL 재검토 (보류) ⚠️

**질문:**
1. AIRL이 정말 필요한가?
2. Fixed-h BFS만으로 충분하지 않은가?
3. Van Opheusden 구조와 어떻게 결합?

**대안:**
- **Behavioral cloning**: Fixed-h BFS로 충분
- **Model comparison**: h selection으로 충분
- **IRL 대신**: Model-based RL (reward는 heuristic 사용)

**진행 조건:**
- AIRL discriminator 설계 구체화
- Van Opheusden 통합 방법 해결
- 데이터 규모 충분성 확인

---

## 8. 핵심 결정사항

### 확정 사항 ✅
1. **Elo 기반 expertise 분류** (순환 논리 해결)
2. **Fixed-h 범위**: {2, 4, 6, 8, 10} (데이터 반영)
3. **Phase 1 진행**: Fixed-h BFS 구현 및 분석

### 미결 사항 ⚠️
1. **"-2" 보정 근거** (C++ 재계산 필요)
2. **AIRL 진행 여부** (재검토 필요)
3. **AIRL 구체적 설계** (discriminator, training)

### TODO
- [ ] `elo_based_classification.py` 완성 (learning.ipynb Elo 연동)
- [ ] `compute_planning_depth` 재실행하여 "-2" 검증
- [ ] `heuristic_fixed_h.cpp` 구현
- [ ] MATLAB/Python fitting pipeline 구축
- [ ] AIRL 필요성 재평가 및 설계 구체화

---

**마지막 업데이트:** 2024-12-23
**다음 논의:** AIRL discriminator 설계 및 Van Opheusden 통합 방법
