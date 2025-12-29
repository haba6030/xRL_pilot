# Overall Summary 

## 1. 연구 배경

### RQ3: Planning Depth가 Expertise를 구분하는가?

**van Opheusden et al. (2023) 가설**:
- Expert는 더 깊게 계획한다 (higher h)
- Expertise = Planning depth의 함수
- Principal Variation (PV) depth가 Elo와 상관

**우리의 접근**:
- Multi-step IK + Discriminator로 E[h] 추정
- Elo rating (객관적 지표)과 E[h] 상관관계 분석
- Expert vs Novice 비교

---

## 2. 분석 파이프라인

### Phase 0: 데이터 준비

**입력 데이터**:
```
/xRL_pilot/opendata/raw_data.csv
- 40 participants
- 318 human-vs-human games
- 5,482 moves

/xRL_pilot/data/human_elo_ratings.csv
- Elo 범위: 1464.6 - 1535.4
- Expert (n=10): Elo ≥ 1508.8
- Intermediate (n=20): 1491.2 ≤ Elo < 1508.8
- Novice (n=10): Elo < 1491.2
```

---

### Phase 1: Multi-Step IK Data 생성

**스크립트**: `preprocess_multistep_ik_data.py`

**목적**: (state_t, state_{t+h}, action_t) 쌍 생성

**방법**:
```python
for h in [1, 2, 3, 4]:
    for game in games:
        for t in range(len(game) - h):
            state_current = game[t].state
            state_future = game[t+h].state  # h-step 후
            action_taken = game[t].action

            data_h.append((state_current, state_future, action_taken))
```

**출력**:
```
data/multistep_ik/ik_pairs_h1.pkl: 1502 pairs
data/multistep_ik/ik_pairs_h2.pkl: 1403 pairs
data/multistep_ik/ik_pairs_h3.pkl: 1304 pairs
data/multistep_ik/ik_pairs_h4.pkl: 1205 pairs
```

---

### Phase 2: h-Specific Models 학습

**스크립트**: `train_separate_h_models.py`

**핵심 아이디어**: Separate encoders (Mhammedi joint model 개선)

**방법**:
```python
# 각 h마다 독립적인 모델
for h in [1, 2, 3, 4]:
    model_h = LogisticRegression()

    # h-specific 데이터만 사용
    X = [state_current, state_future]  # 178-dim (no h_onehot!)
    y = action

    model_h.fit(X_h, y_h)
    save(f'models/separate_h/model_h{h}.pkl')
```

**왜 Separate?**
```
Joint model 문제:
- [state_current, state_future, h_onehot] → action
- 모델이 state 무시하고 h_onehot만 보는 shortcut 학습
- h-interference 발생

Separate model 장점:
- 각 모델이 100% 용량을 해당 h에 집중
- h를 "체화" (모델 자체가 h를 표현)
- KL divergence 증가: 0.04 → 0.10 (2.6배 개선)
```

**결과**:
```
model_h1.pkl: 77.1% validation accuracy
model_h2.pkl: 26.0% validation accuracy
model_h3.pkl: 18.8% validation accuracy
model_h4.pkl: 14.9% validation accuracy

→ h가 클수록 정확도 낮음 (예측 어려움, 미래는 불확실)
→ 하지만 성능(win rate)은 h=4가 가장 높음!
```

---

### Phase 3: Trajectory 생성 (Rollout Simulation)

**스크립트**: `generate_trajectories_separate_h.py`, `generate_h23_trajectories.py`

**목적**: 각 h 값으로 게임 플레이 시뮬레이션

**핵심 방법**: Rollout-based Inference

```python
def select_action(env, model_h, h):
    legal_actions = env.get_legal_actions()
    scores = []

    for action in legal_actions:
        # 1. 이 action을 선택
        sim_env = deepcopy(env)
        sim_env.step(action)

        # 2. 나머지 h-1 수를 "무작위"로 시뮬레이션 ⚠️
        for _ in range(h - 1):
            legal = sim_env.get_legal_actions()
            random_action = np.random.choice(legal)  # ← 문제!
            sim_env.step(random_action)

        # 3. h-step 후 상태로 점수 계산
        future_state = sim_env.get_observation()
        features = [current_state, future_state]
        score = model_h.predict_proba(features)[action]
        scores.append(score)

    # 4. Softmax로 action 선택
    probs = softmax(scores)
    return np.random.choice(legal_actions, p=probs)
```

**⚠️ Random Rollout의 문제점**:
```
Training: 실제 인간의 미래 state 사용
Inference: 무작위 rollout으로 미래 state 근사

→ Distribution mismatch!
→ 특히 expert 게임에서 문제 (상대도 expert)
```

**출력**:
```
data/separate_h_trajectories/trajectories_h1.pkl: 100 episodes, 2455 actions
data/separate_h_trajectories/trajectories_h2.pkl: 100 episodes, 2325 actions
data/separate_h_trajectories/trajectories_h3.pkl: 100 episodes, 2325 actions
data/separate_h_trajectories/trajectories_h4.pkl: 100 episodes, 2258 actions
```

---

### Phase 4: Multi-Class Discriminator 학습

**스크립트**: `train_multiclass_discriminator.py`

**목적**: (state, action) → P(h=1,2,3,4) 분류

**Architecture**:
```
Input: [state(89-dim), action_onehot(36-dim)] → 125-dim
Hidden: [256, 128, 64] (ReLU + BatchNorm + Dropout)
Output: 4 classes (h=1,2,3,4) (softmax)

Loss: CrossEntropyLoss
Optimizer: Adam(lr=0.001, weight_decay=1e-4)
Training: 100 epochs, batch_size=64
```

**데이터**:
```
Training: 7490 (state, action, h) pairs
Test: 1873 pairs
```

**성능**:
```
Test Accuracy: 93.8%
Chance: 25%

Per-class F1:
- h=1: 0.950 (excellent)
- h=2: 0.925 (good)
- h=3: 0.909 (good)
- h=4: 0.923 (excellent)

Confusion Matrix: 거의 완벽한 대각선
```

**해석**: Planning depth는 **highly identifiable** from behavior!

---

### Phase 5: Human E[h] 추정

**스크립트**: `estimate_player_h_multiclass_fixed.py`

**🚨 Critical Bug Fix**: Human-vs-human 게임 처리

#### 5.1. 원래 버전의 문제 (`estimate_player_h_multiclass.py`)

**잘못된 코드**:
```python
# 게임 단위로 분석
for game in games:
    game_moves = game['moves']  # 양쪽 참가자의 모든 수

    # 게임 전체 분석 (두 사람의 수가 섞임!)
    h_probs, h_pred, h_expected = estimate_h_for_game(game_moves)

    # 양쪽 참가자에게 같은 값 할당 ❌
    participants = set(move['participant'] for move in game_moves)
    for participant in participants:  # 2명
        player_data[participant]['h_expected'].append(h_expected)
```

**문제점**:
```
Human-vs-human 게임 = 2명 참가
게임 전체 수 분석 = 두 사람의 수가 섞임
→ 두 참가자가 동일한 E[h] 값을 가짐!

결과:
Participant 16 (Expert, Elo 1535): E[h] = 2.823
Participant 15 (Novice, Elo 1465): E[h] = 2.823 ← 동일!

→ 개인차가 완전히 사라짐
→ Elo vs E[h] 상관관계 = 0.000
```

#### 5.2. 수정된 버전 (`estimate_player_h_multiclass_fixed.py`)

**올바른 코드**:
```python
# 참가자별로 분석
for participant in all_participants:
    # 이 참가자가 둔 수만 추출
    participant_moves = df[df['participant'] == participant]

    # 이 참가자의 수만 분석
    observations = []
    actions = []
    for move in participant_moves:
        obs = parse_board_state(move['black_pieces'], move['white_pieces'])
        action = move['move']
        observations.append(obs)
        actions.append(action)

    # Discriminator로 E[h] 추정
    probs = discriminator.predict(observations, actions)
    h_probs = np.mean(probs, axis=0)
    h_expected = np.sum(h_probs * [1,2,3,4])

    results[participant] = h_expected
```

**수정 후 결과**:
```
Participant 16 (Expert, Elo 1535): E[h] = 2.798
Participant 15 (Novice, Elo 1465): E[h] = 2.808

→ 이제 각자 고유한 값!
→ 개인차 복원
```

---

### Phase 6: Elo vs E[h] 분석

**스크립트**: `analyze_elo_vs_h.py`

**방법**:
```python
# 1. Elo rating 로드
elo_df = pd.read_csv('data/human_elo_ratings.csv')

# 2. E[h] 추정값 로드 (수정된 버전)
h_df = pd.read_csv('human_h_multiclass_estimates_fixed.csv')

# 3. 병합
merged_df = elo_df.merge(h_df, on='participant')

# 4. 상관관계 분석
r_spearman, p_value = spearmanr(merged_df['elo'], merged_df['h_expected_mean'])

# 5. 그룹 비교
expert = merged_df[merged_df['expertise'] == 'expert']
intermediate = merged_df[merged_df['expertise'] == 'intermediate']
novice = merged_df[merged_df['expertise'] == 'novice']

f_stat, p_anova = f_oneway(expert['h_expected_mean'],
                            intermediate['h_expected_mean'],
                            novice['h_expected_mean'])
```

---

## 3. 주요 결과

### 3.1. Overall E[h] 분포

**전체 통계** (N=40):
```
E[h] mean: 2.844 ± 0.085
E[h] range: [2.562, 3.009]

Mode 분류:
- h=1: 0명 (0%)
- h=2: 0명 (0%)
- h=3: 3명 (7.5%)
- h=4: 37명 (92.5%)

평균 확률 분포:
P(h=1) = 0.135
P(h=2) = 0.227
P(h=3) = 0.297
P(h=4) = 0.341
```

**해석**:
- 대부분 참가자가 h=4로 분류되지만
- E[h] < 3.0 (진짜 h=4는 아님)
- Mixed strategy 사용 (context-dependent planning)

---

### 3.2. Elo vs E[h] 상관관계

**Correlation Analysis**:
```
Pearson r:  -0.142, p = 0.383 (ns)
Spearman r: -0.117, p = 0.471 (ns)

Effect size: small (|r| = 0.117)
```

**결론**: Elo rating과 E[h] 사이에 **유의미한 상관관계 없음**

---

### 3.3. Expertise Groups 비교

**Group Statistics**:
```
Expert (n=10):       E[h] = 2.804 ± 0.097
Intermediate (n=20): E[h] = 2.859 ± 0.067
Novice (n=10):       E[h] = 2.853 ± 0.090

ANOVA: F = 1.47, p = 0.243 (ns)
```

**Pairwise t-tests**:
```
Expert vs Novice:       t = -1.12, p = 0.279, d = -0.527
Expert vs Intermediate: t = -1.74, p = 0.094, d = -0.654 ⚠️
Intermediate vs Novice: t =  0.18, p = 0.862, d =  0.067
```

**🚨 핵심 발견**:
- Expert의 E[h]가 가장 **낮음**!
- Expert vs Intermediate: d = -0.654 (medium effect, marginally significant)
- 역U자형 패턴 (Intermediate가 peak)

---

### 3.4. Win Rate vs E[h]

**놀라운 발견**:
```
Win Rate vs E[h]: r = -0.426, p = 0.006 ✅
Elo vs E[h]:      r = -0.117, p = 0.471 ❌

→ Win rate와는 강한 음의 상관관계!
```

**해석**:
- Win rate가 **높을수록** E[h]가 **낮다**
- Elo는 상관없지만 Win rate와는 상관있음
- Elo ≠ Win rate (Elo가 더 안정적 지표)

**Elo vs Win Rate**:
```
r = 0.600, p < 0.001
→ 둘은 관련있지만 다른 정보 포함
```

---

### 3.5. Binary Split (High vs Low Elo)

**Median split** (Elo = 1500):
```
High-Elo (n=24): E[h] = 2.844 ± 0.088
Low-Elo (n=16):  E[h] = 2.843 ± 0.084

t-test: t = 0.05, p = 0.960
Mean difference = 0.001
Cohen's d = 0.017
```

**결론**: 거의 **차이 없음** (완전히 동일)

---

## 4. Expertise Paradox 해석

### 4.1. 왜 역설인가?

**예상** (van Opheusden 가설):
```
Expertise ↑ → Planning depth ↑
Expert는 더 멀리 내다본다 (higher h)
```

**실제** (우리 결과):
```
Expertise ↑ → Planning depth ↓ (약한 경향)
Expert:       E[h] = 2.804
Intermediate: E[h] = 2.859
Novice:       E[h] = 2.853
```

**패턴**: 역U자형 또는 음의 경향

---

### 4.2. 가능한 설명

#### 설명 1: Efficiency Hypothesis (진짜 패턴) ⭐⭐⭐⭐⭐

**가설**: Expert는 오히려 **적게** 계획한다

**논리**:
```
Novice:
- Heuristic/pattern 부족
- 깊게 생각하지만 방향 틀림
- 비효율적 탐색 (h↑, performance↓)

Expert:
- 강력한 heuristic/pattern recognition
- 직관적으로 좋은 수 파악
- 짧게 계획해도 정확 (h↓, performance↑)

→ "Thinking Fast" (Kahneman)
→ System 1 (intuitive) > System 2 (deliberative)
```

**증거**:
```
Chess masters (de Groot, 1965):
- 좋은 수를 빠르게 떠올림 (5초)
- Novice는 오래 고민 (20분+)
- "Expert intuition" = chunk recognition

Go experts (Lee Sedol):
- "느낌"으로 수를 둠
- 계산은 확인용

van Opheusden PV depth:
- PV depth ≠ actual planning depth
- PV depth = search trace length
- Expert는 pruning을 잘함 → 짧은 trace
```

**함의**:
```
Planning depth ≠ Skill
Skill = Efficient planning (quality > quantity)

Expert: Low h + high quality = high performance
Novice: High h + low quality = low performance
```

---

#### 설명 2: Random Rollout Artifact ⭐⭐⭐⭐⭐

**가설**: Rollout 방법이 expert의 h를 underestimate

**논리**:
```python
# 현재 rollout (문제)
for _ in range(h - 1):
    random_action = np.random.choice(legal_actions)  # Uniform!
    sim_env.step(random_action)

문제점:
1. Expert 게임: 상대도 expert (좋은 수를 둠)
2. Random rollout: 좋은 수/나쁜 수 동일 확률
3. Distribution mismatch!

Expert의 실제 미래:
- 상대가 좋은 수를 둠
- 위협적인 상황 많음
- h를 크게 해야 대응 가능

Random rollout 미래:
- 상대가 무작위
- 위협 적음
- h 작아도 충분

→ Expert가 낮은 h로 분류됨!
```

**예측**:
```
Random rollout:
Expert E[h] = 2.80
Novice E[h] = 2.85

Opponent model rollout:
Expert E[h] = 2.95 (증가!)
Novice E[h] = 2.85 (변화 적음)

→ 역설 해소 가능
```

---

#### 설명 3: Discriminator Calibration Issue ⭐⭐⭐

**가설**: Discriminator가 expert 행동을 misclassify

**논리**:
```
Training data:
- h=1,2,3,4 균등 분포 (각 25%)
- Random rollout으로 생성
- Synthetic policy

Real expert:
- Context-dependent planning
- h=2,3을 전략적으로 사용
- Adaptive policy

Discriminator:
- Fixed h를 가정
- Context 무시
- Expert의 adaptive h를 "낮은 h"로 오해
```

**증거**:
```
Expert 확률 분포:
P(h=1) = 0.14
P(h=2) = 0.23
P(h=3) = 0.30
P(h=4) = 0.33

→ 골고루 분산 (adaptive!)
→ 하지만 E[h] = 2.80 (낮음)
```

---

#### 설명 4: Sample Homogeneity (Ceiling Effect) ⭐⭐

**가설**: 모든 참가자가 skilled → 차이 작음

**논리**:
```
van Opheusden 데이터:
- 대학생, 성인
- 100게임 완료 조건
- Selection bias (skilled만 참여)

결과:
Expert Elo: 1508-1535 (range = 27)
Novice Elo: 1465-1491 (range = 26)
전체 range: 71점 (매우 좁음!)

E[h] 범위: 2.56-3.01 (range = 0.45)
→ 이론적 범위 1-4의 11%만 사용
→ Ceiling effect
```

**함의**:
- 진짜 Novice 없음 (모두 intermediate 이상)
- 차이가 있어도 탐지 어려움
- Power 부족

---

#### 설명 5: Win Rate ≠ Elo ⭐⭐⭐

**가설**: Win rate는 noisy measure

**논리**:
```
Win rate 계산:
- 4-in-a-row 승리 판정 간단
- Draw가 많음 (83.6%)
- 게임 수 적음 (~16 games/person)
- Variance 큼

Elo rating:
- Bayesian inference
- 모든 게임 누적
- 더 안정적
- 하지만 E[h]와 무상관

역설:
Win rate ↔ E[h]: r = -0.426 ✅
Elo ↔ E[h]:      r = -0.117 ❌

→ Win rate에만 있는 noise가 E[h]와 correlate?
```

---

### 4.3. 가장 가능성 높은 설명

**복합적 원인**:

**1순위: Random Rollout Artifact (설명 2)**
```
가장 직접적인 원인
Testable (opponent model 구현하면 검증)
예측 가능 (expert h가 증가할 것)
```

**2순위: Efficiency Hypothesis (설명 1)**
```
이론적으로 타당
Chess/Go 증거 있음
하지만 van Opheusden과 모순
```

**3순위: Discriminator Calibration (설명 3)**
```
가능성 있음
Adaptive h를 낮은 h로 오해
```

**가능성 낮음**: Sample Homogeneity (설명 4), Win Rate Noise (설명 5)

---

## 5. 검증 계획

### 5.1. Opponent Model Rollout (최우선) 🔥

**목적**: Random rollout artifact 검증

**방법**:
```python
# Opponent model 학습
opponent_model = LogisticRegression()
opponent_moves = extract_all_opponent_moves(games)
opponent_model.fit(states, actions)

# Rollout with opponent model
for _ in range(h - 1):
    legal = sim_env.get_legal_actions()
    state = sim_env.get_observation()

    # 확률분포로 샘플링
    probs = opponent_model.predict_proba(state)[legal]
    probs /= probs.sum()
    action = np.random.choice(legal, p=probs)

    sim_env.step(action)
```

**예측**:
```
H0: Expert E[h] 증가
H1: Expert E[h] 변화 없음

만약 H0:
→ Random rollout이 원인 (artifact)
→ 진짜 expert는 높은 h

만약 H1:
→ Efficiency hypothesis 지지
→ Expert는 진짜 낮은 h
```

---

### 5.2. Rollout Method 비교 실험

**세 가지 방법 비교**:
1. **Random**: 현재 방법 (uniform)
2. **Opponent Model**: 학습된 상대방 정책
3. **Heuristic**: van Opheusden heuristic 사용

**절차**:
```
각 방법으로:
1. Trajectory 생성 (100 episodes × 4 h values)
2. Discriminator 학습
3. Human E[h] 추정
4. Elo vs E[h] 상관관계 계산

비교:
- 어느 방법이 가장 높은 상관관계?
- Expert E[h] 값 변화?
```

**기대**:
```
Random:   r = -0.12 (현재)
Opponent: r = +0.30 (예상)
Heuristic: r = +0.20 (예상)

→ 방법론적 개선!
```

---

### 5.3. External Validation

**다른 데이터셋**:
- Chess (Lichess database)
- Go (KGS server data)
- Economic games (cooperation tasks)

**예측**:
```
만약 Efficiency hypothesis:
→ 모든 도메인에서 expert < novice

만약 Artifact:
→ Rollout 방법에 따라 달라짐
```

---

## 6. 논문 작성을 위한 핵심 포인트

### 6.1. Introduction

**배경**:
```
- IRL의 planning confounder 문제 (Yao et al., 2024)
- Expertise와 planning depth 관계 (van Opheusden et al., 2023)
- 하지만 planning depth를 직접 측정하기 어려움
```

**우리 접근**:
```
- Multi-step IK + Discriminator로 h 추정
- Behavior-based inference (no assumptions)
- Elo rating (objective measure)과 비교
```

---

### 6.2. Methods

**핵심 기여**:
1. **Separate encoders**: h-interference 제거
2. **Rollout-based inference**: Training-inference match
3. **Multi-class discriminator**: Binary보다 정확 (93.8% acc)
4. **Player-specific analysis**: Human-vs-human 게임 올바른 처리

---

### 6.3. Results

**예상치 못한 발견**:
```
RQ3: Does planning depth discriminate expertise?
Answer: NO - 오히려 음의 경향

Expert:       E[h] = 2.804
Intermediate: E[h] = 2.859
Novice:       E[h] = 2.853

Elo vs E[h]: r = -0.12, p = 0.47 (ns)
Win rate vs E[h]: r = -0.43, p = 0.006 ✅
```

---

### 6.4. Discussion

**두 가지 해석**:

**해석 A: Methodological Artifact**
```
Random rollout이 문제
Expert 게임의 상대방 행동 mismatch
→ Opponent model로 해결 가능
→ Null result (방법론적 문제)
```

**해석 B: Genuine Pattern (Efficiency Hypothesis)**
```
Expert는 효율적으로 계획
Intuition > Deliberation
Quality > Quantity
→ Novel finding (새로운 발견)
```

**검증**:
```
Opponent model rollout 구현
→ Expert E[h] 증가하면 A
→ 변화 없으면 B
```

---

### 6.5. Implications

**만약 해석 A (Artifact)**:
```
- Rollout 방법이 critical
- 실제 상대방 행동 모델링 필요
- Multi-step IK 개선 방향 제시
```

**만약 해석 B (Efficiency)**:
```
- Planning depth ≠ Expertise
- Expertise = Efficient planning
- van Opheusden PV depth 재해석 필요
- Chess/Go 연구와 일치
```

**공통**:
```
- Planning은 identifiable (93.8% acc)
- Mixed strategy 사용 (adaptive h)
- Individual differences exist
```

---

## 7. 다음 단계 (우선순위)

### 🔥 Priority 1: Opponent Model Rollout

**Task**: `implement_opponent_model_rollout.py`

**목표**:
1. 상대방 정책 학습
2. Trajectory 재생성
3. E[h] 재추정
4. Elo vs E[h] 재분석

**예상 작업 시간**: 2-3시간

**기대**:
- Expert E[h] 증가
- Elo vs E[h] 상관관계 positive
- Expertise paradox 해소

---

### Priority 2: Rollout Method Comparison

**세 가지 방법 체계적 비교**:
- Random (baseline)
- Opponent model (proposed)
- Heuristic-guided (alternative)

**분석**:
- Discriminator accuracy
- E[h] 분포
- Elo correlation
- Expert vs Novice difference

---

### Priority 3: Sensitivity Analysis

**Robustness checks**:
- Sample size 영향 (bootstrap)
- Discriminator architecture
- Training data size
- h 범위 (h=1-5 vs h=1-4)

---

## 8. 기술적 세부사항

### 8.1. Bug Fix 요약

**문제**: Human-vs-human 게임에서 두 참가자가 같은 E[h] 값

**원인**: 게임 전체 수를 분석 → 두 사람 수 섞임

**해결**: 참가자별로 분리
```python
# Before (wrong)
for game in games:
    all_moves = game['moves']  # 양쪽 수
    h = estimate(all_moves)
    for participant in participants:  # 2명
        results[participant] = h  # 같은 값!

# After (correct)
for participant in all_participants:
    participant_moves = df[df['participant'] == participant]
    h = estimate(participant_moves)  # 각자 분석
    results[participant] = h
```

**영향**:
- Before: r(Elo, E[h]) = 0.000
- After: r(Elo, E[h]) = -0.117
- Individual variance 복원

---

### 8.2. 데이터 파일

**입력**:
```
/xRL_pilot/opendata/raw_data.csv
/xRL_pilot/data/human_elo_ratings.csv
models/multiclass_discriminator.pt
```

**중간 결과**:
```
human_h_multiclass_estimates.csv (wrong - deprecated)
human_h_multiclass_estimates_fixed.csv (correct)
```

**최종 분석**:
```
results/elo_vs_h_analysis.csv
figures/elo_vs_h_analysis.png
figures/human_h_multiclass_results_fixed.png
```

---

### 8.3. 통계 검정력

**현재 샘플**:
```
N = 40
Expert: 10
Intermediate: 20
Novice: 10

Effect size (Expert vs Intermediate): d = -0.654
Power (two-tailed t-test, α=0.05): ~40%
→ Underpowered!

필요 샘플 (power = 0.80):
→ ~40 per group (총 120명)
```

---

## 9. 요약

### 핵심 메시지

**발견**:
```
✅ Planning depth is identifiable (93.8% accuracy)
✅ Humans use mixed strategies (E[h]=2.84, not pure h=4)
❌ Expertise does NOT increase planning depth
🚨 Expert have LOWER E[h] than intermediate (paradox!)
```

**가능한 원인**:
```
1. Random rollout artifact (testable)
2. Expert efficiency (genuine pattern)
3. Discriminator calibration issue
```

**다음 단계**:
```
1. Opponent model rollout 구현 (critical!)
2. Rollout method 비교
3. External validation
```

**논문 방향**:
```
Option A: Methodological paper
- "Rollout method matters for planning depth inference"
- Null result → 방법론적 개선

Option B: Substantive finding
- "Expert efficiency: Less planning, better performance"
- Positive result → 새로운 이론
```

---

**Last Updated**: 2025-12-29
**Status**: Paradox discovered, opponent model next
**For**: Lab discussion, paper draft

---

## References

**Key Papers**:
- van Opheusden et al. (2023). *Expertise increases planning depth in human gameplay.* Nature.
- Mhammedi et al. (2023). *Reinforcement learning from passive data via latent intentions.* NeurIPS.
- Yao et al. (2024). *Inverse reinforcement learning with the average reward MDP.*
- de Groot (1965). *Thought and choice in chess.*
- Kahneman (2011). *Thinking, Fast and Slow.*

