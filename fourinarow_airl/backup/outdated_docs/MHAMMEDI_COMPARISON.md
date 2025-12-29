# Mhammedi(2023) Multi-Step IK vs Our Implementation

## Mhammedi(2023) 원본 방법

### Paper: "Reinforcement Learning from Passive Data via Latent Intentions"

**핵심 아이디어**: Multi-step Inverse Kinematics
```
Given: Trajectory τ = (x_0, a_0, x_1, a_1, ..., x_T)
Goal: Learn policy π that predicts actions from observations

Key insight: Use future observations to infer current action
- One-step IK: (x_t, x_{t+1}) → a_t
- Multi-step IK: (x_t, x_{t+h}) → a_t  (for all h > 0)
```

### Algorithm 2: IKDP (Inverse Kinematics for Dynamic Programming)

**Line 7 (핵심)**:
```
max_{f,φ} Σ_{(x_h, a_h, x_h')} log f((a_h, j) | φ(x_h), φ(x_h'))
```

Where:
- `f`: Inverse model (action predictor)
- `φ`: Encoder (observation → latent)
- `j = h' - h`: Number of steps ahead
- `(x_h, x_h')`: Current and future observations

**h의 역할**:
- h는 "어느 미래를 볼 것인가"를 결정
- Multiple h values를 함께 학습 (joint training)
- 모든 h ∈ {1, ..., H}에 대해 같은 model f 사용

---

## 우리 구현

### Phase 1: Training (Separate h-Specific Models)

**Data generation**:
```python
# h=1 training data
for traj in human_games:
    for t in range(len(traj) - 1):
        pair = {
            'state_current': traj[t],      # x_t
            'state_future': traj[t + 1],   # x_{t+1}
            'action': action[t]             # a_t
        }
        h1_data.append(pair)

# h=4 training data
for traj in human_games:
    for t in range(len(traj) - 4):
        pair = {
            'state_current': traj[t],      # x_t
            'state_future': traj[t + 4],   # x_{t+4}
            'action': action[t]             # a_t
        }
        h4_data.append(pair)
```

**Model training**:
```python
# Separate models (NOT joint like Mhammedi)
model_h1 = MLP()
model_h1.fit(
    X = [state_current, state_future],  # (178-dim)
    y = action                           # (36 classes)
)

model_h4 = MLP()  # Completely separate!
model_h4.fit(
    X = [state_current, state_future],
    y = action
)
```

**Key difference**:
- Mhammedi: ONE model for all h (joint)
- Ours: SEPARATE models per h

---

### Phase 2: Inference (Planning via Rollout)

**At test time, h determines rollout depth**:

```python
def select_action(env, h_model, h):
    current_state = env.get_state()

    # For each legal action
    for action in legal_actions:
        # Simulate h-step rollout
        future_state = simulate_h_steps(env, action, h)

        # Score with h-specific model
        score = h_model.predict_proba([current_state, future_state])[action]

    # Select action via softmax
    return softmax_sample(scores)
```

**h의 이중 역할**:
1. **Training**: 어느 future state를 사용할지 (t+h)
2. **Inference**: 몇 step을 시뮬레이션할지 (rollout depth)

---

## 핵심 차이점 비교표

| Aspect | Mhammedi(2023) | Our Implementation |
|--------|----------------|-------------------|
| **Model architecture** | Single joint model | Separate per h |
| **h encoding** | h as input feature | No encoding (model IS h) |
| **Training data** | All h jointly | h-specific data only |
| **h at inference** | Part of input (x, x', h) | Rollout depth parameter |
| **Capacity** | Shared across h | Full capacity per h |
| **Interference** | Possible | Eliminated |
| **Scalability** | Better (one model) | Worse (h models) |

---

## 왜 우리가 Separate models를 선택했나?

### Mhammedi의 Joint Model 실패 (우리의 경험)

**시도했던 것** (`train_multistep_ik_sklearn.py`):
```python
# Joint model with h_onehot
for h in [1,2,3,4]:
    features = [state_current, state_future, h_onehot]  # 182-dim

model = MLP()
model.fit(all_h_data)  # All h together
```

**결과**: KL = 0.0399 (실패)

**문제 진단**: h-interference
- Model이 h를 무시하고 "average" behavior 학습
- h_onehot이 충분히 강한 signal이 아님
- Shared capacity → h-specific patterns 학습 실패

### Separate Models 성공

**변경**:
```python
# h=1 model: ONLY h=1 data
model_h1.fit(h1_data)  # 178-dim, no h_onehot

# h=4 model: ONLY h=4 data
model_h4.fit(h4_data)  # 178-dim, no h_onehot
```

**결과**: KL = 0.1049 (성공!) ✅

**Why it worked**:
- No interference
- Full capacity per h
- Model architecture IS the h encoding

---

## Rollout의 역할 (Mhammedi에는 없음)

### Mhammedi: Passive Learning
```
Training: (x_t, x_{t+h}) from real trajectory
Inference: (x_t, x_{t+h}) ... wait, 미래를 어떻게 아나?

→ Mhammedi는 주로 offline RL context
→ Future observation을 알 수 있다고 가정
```

### Ours: Active Planning
```
Training: (x_t, x_{t+h}) from real trajectory
Inference: (x_t, SIMULATED x_{t+h}) via rollout

→ Rollout으로 future를 예측
→ 각 action의 h-step future를 simulate
→ Model로 평가 → best action 선택
```

**This is planning**!
```python
# h=1: 1-step lookahead
future_1 = simulate(action, steps=1)
score_1 = model_h1([current, future_1])[action]

# h=4: 4-step lookahead
future_4 = simulate(action, steps=4)
score_4 = model_h4([current, future_4])[action]

→ h=1은 근시안적, h=4는 장기적
→ 다른 actions 선택 → KL divergence 발생!
```

---

## 이론적 관점

### Mhammedi의 기여
- Multi-step IK가 one-step IK보다 sample-efficient
- Latent intentions (h) 를 학습 가능
- Block MDP에서 provably works

### 우리의 확장
- **Planning-aware IRL**에 적용
- h를 explicit parameter로 조작
- Separate encoders로 h-interference 해결
- Rollout으로 active planning 구현

### Combined View
```
Mhammedi: Multi-step IK for better representation learning
Ours: Multi-step IK for h-dependent behavior generation

Mhammedi: One model, many h (representation)
Ours: Many models, one h each (behavior)

Mhammedi: Passive (offline RL)
Ours: Active (planning)
```

---

## 수식 비교

### Mhammedi's Objective
```
max_{f,φ} E_{τ} [ Σ_t Σ_h log f(a_t, h | φ(x_t), φ(x_{t+h})) ]
```
- Single f for all h
- h as input variable

### Our Objective (Separate Models)
```
For h=1:
  max_{f_1,φ_1} E_{τ} [ Σ_t log f_1(a_t | φ_1(x_t), φ_1(x_{t+1})) ]

For h=4:
  max_{f_4,φ_4} E_{τ} [ Σ_t log f_4(a_t | φ_4(x_t), φ_4(x_{t+4})) ]
```
- Separate f_h for each h
- h is implicit (model index)

### Our Inference (Planning via Rollout)
```
At test time:
  For each a ∈ A:
    x̂_{t+h} = Rollout(x_t, a, h)  ← NEW!
    score[a] = f_h(a | x_t, x̂_{t+h})

  π(a|x_t) = softmax(scores)
```

---

## 언제 Mhammedi vs Separate Models?

### Mhammedi (Joint) 추천:
- Offline RL setting (future known)
- Want compact representation
- Limited data per h
- Need generalization across h

### Separate Models 추천:
- Online planning (future simulated)
- Want maximally distinct behaviors
- Have enough data per h
- h is explicit experimental variable (우리 경우!)

---

## 결론

**Mhammedi(2023)**: Multi-step IK for representation learning
**Our work**: Multi-step IK for planning-aware behavior generation

**Key innovation**:
1. Separate encoders eliminate h-interference
2. Rollout enables active planning
3. h becomes explicit, manipulable parameter

**Trade-off**:
- More models to train (h separate models)
- But: Clearer h-dependent behaviors
- Result: KL = 0.1049 ✅

This is **planning-aware IRL**, not just representation learning!
