# Code Walkthrough: Multi-Step IK Implementation

완전한 코드 플로우 설명 및 h의 역할 추적

---

## 전체 파이프라인 개요

```
[1] Human Data
    ↓
[2] Multi-step IK Pairs (h=1, h=4)
    ↓
[3] Train Separate Models
    ↓
[4] Generate Trajectories (with Rollout)
    ↓
[5] Compare Distributions (KL divergence)
```

---

## Step 1: Data Preprocessing

### File: `preprocess_multistep_ik_data.py`

**Input**: `opendata/raw_data.csv` (67,331 moves, 101 games)

**h의 역할**: 시간 offset (t → t+h)

```python
def load_human_trajectories():
    """
    Human game data → trajectories

    Returns:
        trajectories: List of games
            Each game: {
                'observations': [(89,), (89,), ...],  # States
                'actions': [24, 11, 14, ...],          # Actions
                'game_id': 0
            }
    """
    # Parse CSV row by row
    for row in df.iterrows():
        black_pieces = row['black_pieces']  # "000...100" (36-char)
        white_pieces = row['white_pieces']
        color = row['color']

        # Convert to 89-dim observation
        state = parse_board_state(black_pieces, white_pieces, color)
        # state = [black (36), white (36), van_opheusden_features (17)]

        current_game['observations'].append(state)
        current_game['actions'].append(row['move'])
```

**h를 이용한 pair 생성**:

```python
def create_multistep_ik_pairs(trajectories, h):
    """
    h의 역할: 몇 step 뒤의 state를 볼 것인가?

    h=1: (state_t, state_{t+1}, action_t)
    h=4: (state_t, state_{t+4}, action_t)
    """
    pairs = []

    for traj in trajectories:
        observations = traj['observations']  # [(89,), (89,), ...]
        actions = traj['actions']            # [a_0, a_1, ...]

        T = len(actions)

        # **h가 여기서 사용됨!**
        for t in range(T - h):  # t+h가 valid해야 함
            pair = {
                'state_current': observations[t],      # x_t
                'state_future': observations[t + h],   # x_{t+h} ← h!
                'action': actions[t],                   # a_t
                'h': h,
                'game_id': traj['game_id'],
                't': t
            }
            pairs.append(pair)

    return pairs
```

**Output**:
- `data/multistep_ik/ik_pairs_h1.pkl`: 1502 pairs (state_t, state_{t+1}, a_t)
- `data/multistep_ik/ik_pairs_h4.pkl`: 1205 pairs (state_t, state_{t+4}, a_t)

**h의 첫 번째 역할**: Time offset for creating training pairs

---

## Step 2: Train Separate Models

### File: `train_separate_h_models.py`

**h의 역할**: Model index (어느 model을 학습할지)

```python
def load_h_specific_data(h):
    """
    h-specific data만 로드

    h=1 → ik_pairs_h1.pkl (1502 pairs)
    h=4 → ik_pairs_h4.pkl (1205 pairs)
    """
    filepath = f'data/multistep_ik/ik_pairs_h{h}.pkl'
    pairs = pickle.load(filepath)

    X = []
    y = []

    for pair in pairs:
        # **h_onehot 없음!**
        # Features: ONLY current + future states
        features = np.concatenate([
            pair['state_current'],  # (89,)
            pair['state_future']    # (89,)
        ])  # Total: (178,)

        X.append(features)
        y.append(pair['action'])

    return X, y  # (N, 178), (N,)
```

**Model training**:

```python
def train_h_specific_model(h):
    """
    h-specific model 학습

    h=1 → model_h1: (state_t, state_{t+1}) → a_t
    h=4 → model_h4: (state_t, state_{t+4}) → a_t
    """
    X, y = load_h_specific_data(h)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        ...
    )

    model.fit(X, y)

    # Save h-specific model
    joblib.dump({
        'model': model,
        'h': h,  # ← h는 metadata로 저장
        ...
    }, f'models/separate_h/model_h{h}.pkl')
```

**Output**:
- `models/separate_h/model_h1.pkl`: h=1 전용 model (val acc: 77.1%)
- `models/separate_h/model_h4.pkl`: h=4 전용 model (val acc: 14.9%)

**h의 두 번째 역할**: Model specialization (각 h마다 독립적 model)

---

## Step 3: Generate Trajectories with Rollout

### File: `generate_trajectories_separate_h.py`

**h의 역할**: Rollout depth (planning horizon)

```python
class SeparateHAgent:
    def __init__(self, model_path, h):
        self.model = load_model(model_path)
        self.h = h  # ← Rollout depth로 사용됨

    def rollout_future_state(self, env, action, rng):
        """
        **h의 핵심 역할: 몇 step을 시뮬레이션할지**

        h=1: 1-step rollout
        h=4: 4-step rollout
        """
        # Deep copy env (원본 보존)
        sim_env = copy.deepcopy(env)

        # Initial action
        sim_env.step(action)

        # **h-1번 더 시뮬레이션** ← h가 여기서 사용됨!
        for step in range(self.h - 1):
            if terminated or truncated:
                break

            # Random action (rollout policy)
            legal = sim_env.get_legal_actions()
            random_action = rng.choice(legal)
            sim_env.step(random_action)

        # h-step 후의 state
        future_state = sim_env._get_observation()  # (89,)

        return future_state
```

**Action selection (planning)**:

```python
def select_action(self, env, rng):
    """
    Planning via rollout simulation

    Each legal action → h-step simulation → score
    """
    current_state = env._get_observation()  # (89,)

    legal_actions = env.get_legal_actions()
    action_scores = {}

    # **각 action에 대해 h-step rollout**
    for action in legal_actions:
        # 1. Simulate h-step future
        future_state = self.rollout_future_state(env, action, rng)

        # 2. Score with h-specific model
        features = np.concatenate([current_state, future_state])  # (178,)

        # model_h predicts: (x_t, x_{t+h}) → P(a_t)
        probs = self.model.predict_proba([features])[0]  # (36,)
        score = probs[action]

        action_scores[action] = score

    # 3. Softmax and sample
    probs = softmax(action_scores)
    selected_action = sample(probs)

    return selected_action
```

**실제 동작 예시**:

```python
# h=1 agent
agent_h1 = SeparateHAgent(model_h1, h=1)

# 현재 state에서 action 선택
# Action 12를 고려:
future_1step = rollout(env, action=12, steps=1)  # 1-step ahead
score_12 = model_h1([current, future_1step])[12]

# Action 24를 고려:
future_1step = rollout(env, action=24, steps=1)
score_24 = model_h1([current, future_1step])[24]

# → scores를 비교해서 best action 선택
```

```python
# h=4 agent
agent_h4 = SeparateHAgent(model_h4, h=4)

# Action 12를 고려:
future_4step = rollout(env, action=12, steps=4)  # **4-step ahead!**
score_12 = model_h4([current, future_4step])[12]

# → h=1보다 더 먼 미래를 본다
# → 다른 actions를 선호할 수 있음
```

**Output**:
- `data/separate_h_trajectories/trajectories_h1.pkl`: 100 episodes (h=1 planning)
- `data/separate_h_trajectories/trajectories_h4.pkl`: 100 episodes (h=4 planning)
- `data/separate_h_trajectories/actions_h1.pkl`: 2455 actions
- `data/separate_h_trajectories/actions_h4.pkl`: 2258 actions

**h의 세 번째 역할**: Planning horizon (how far to look ahead)

---

## Step 4: Compare Distributions

### File: `compare_separate_h_distributions.py`

**h의 역할**: Distribution label

```python
def load_actions(h):
    """h로 구분된 action 분포 로드"""
    actions = pickle.load(f'actions_h{h}.pkl')
    return actions

# h=1 distribution
actions_h1 = load_actions(h=1)  # 2455 actions
dist_h1 = compute_distribution(actions_h1)

# h=4 distribution
actions_h4 = load_actions(h=4)  # 2258 actions
dist_h4 = compute_distribution(actions_h4)

# Compare
kl_divergence = KL(dist_h1 || dist_h4)  # 0.1049 ✅
```

**h의 네 번째 역할**: Experimental condition label

---

## h의 전체 역할 요약

| Stage | h의 의미 | 예시 |
|-------|----------|------|
| **1. Data** | Time offset | h=1: t→t+1, h=4: t→t+4 |
| **2. Training** | Model index | model_h1 vs model_h4 |
| **3. Rollout** | Simulation depth | 1-step vs 4-step lookahead |
| **4. Generation** | Planning horizon | myopic vs far-sighted |
| **5. Analysis** | Condition label | Distribution h=1 vs h=4 |

---

## 코드 플로우 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Data Preparation                                   │
└─────────────────────────────────────────────────────────────┘
    Human Games (raw_data.csv)
           ↓ preprocess_multistep_ik_data.py
    ┌──────────────┐         ┌──────────────┐
    │ h=1 pairs    │         │ h=4 pairs    │
    │ (t, t+1, a)  │         │ (t, t+4, a)  │
    │ N=1502       │         │ N=1205       │
    └──────────────┘         └──────────────┘
           ↓                        ↓

┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Training Separate Models                           │
└─────────────────────────────────────────────────────────────┘
    train_separate_h_models.py
    ┌──────────────┐         ┌──────────────┐
    │ model_h1     │         │ model_h4     │
    │ (178→36)     │         │ (178→36)     │
    │ val: 77.1%   │         │ val: 14.9%   │
    └──────────────┘         └──────────────┘
           ↓                        ↓

┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Trajectory Generation with Rollout                 │
└─────────────────────────────────────────────────────────────┘
    generate_trajectories_separate_h.py

    Agent h=1:                Agent h=4:
    ┌────────────┐           ┌────────────┐
    │ For action │           │ For action │
    │ ↓          │           │ ↓          │
    │ Rollout    │           │ Rollout    │
    │ 1 step     │           │ 4 steps    │  ← h determines this!
    │ ↓          │           │ ↓          │
    │ Predict    │           │ Predict    │
    │ with       │           │ with       │
    │ model_h1   │           │ model_h4   │
    └────────────┘           └────────────┘
           ↓                        ↓
    trajectories_h1.pkl      trajectories_h4.pkl
    actions_h1.pkl           actions_h4.pkl
           ↓                        ↓

┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Distribution Comparison                            │
└─────────────────────────────────────────────────────────────┘
    compare_separate_h_distributions.py

    dist_h1 (2455 actions)   dist_h4 (2258 actions)
           ↓                        ↓
              KL divergence = 0.1049 ✅
```

---

## 핵심 코드 스니펫 (h 추적)

### 1. Data: h as time offset

```python
# preprocess_multistep_ik_data.py:140
for t in range(T - h):
    pair = {
        'state_current': observations[t],
        'state_future': observations[t + h],  # ← h determines offset
        'action': actions[t]
    }
```

### 2. Training: h as model index

```python
# train_separate_h_models.py:95
model_path = f'models/separate_h/model_h{h}.pkl'  # ← h in filename
joblib.dump({'model': model, 'h': h}, model_path)
```

### 3. Rollout: h as simulation depth

```python
# generate_trajectories_separate_h.py:86
for step in range(self.h - 1):  # ← h controls loop
    sim_env.step(random_action)
```

### 4. Inference: h-specific model selection

```python
# generate_trajectories_separate_h.py:145
future_state = self.rollout_future_state(env, action, rng)
features = np.concatenate([current_state, future_state])
score = self.model.predict_proba([features])[0][action]
# ← self.model is h-specific!
```

---

## 왜 이 구조가 작동하는가?

### Training과 Inference의 일치

**Training**:
```
h=1 data: (real_state_t, real_state_{t+1}) → a_t
h=4 data: (real_state_t, real_state_{t+4}) → a_t
```

**Inference**:
```
h=1: (current_state, simulated_1step_future) → action
h=4: (current_state, simulated_4step_future) → action
```

**Match**: Both use h-step ahead states!

### h-Dependent Behavior Emergence

```
h=1 model learned: "1-step future → action" patterns
h=4 model learned: "4-step future → action" patterns

→ Different futures → Different actions
→ KL divergence = 0.1049 ✅
```

---

## 중요 설계 결정

### 1. Why separate models?
- **Tried**: Joint model with h_onehot → Failed (KL=0.0399)
- **Success**: Separate models → KL=0.1049
- **Reason**: Eliminate h-interference

### 2. Why deepcopy for rollout?
```python
sim_env = copy.deepcopy(env)  # ← Critical!
```
- Without deepcopy: original env gets modified
- Each action needs independent simulation
- Tried shallow copy: Failed (shared state)

### 3. Why random rollout policy?
- Simple, no heuristics
- Adds variance (actually helps h=4)
- Could use learned policy (future work)

### 4. Why temperature=1.0?
- Balanced exploration/exploitation
- Could increase for more diversity
- Tested: Works well

---

## 파일별 h 사용 요약

| File | h 사용처 | 의미 |
|------|----------|------|
| `preprocess_multistep_ik_data.py` | `t + h` | Time offset |
| `train_separate_h_models.py` | `model_h{h}` | Model index |
| `generate_trajectories_separate_h.py` | `range(self.h - 1)` | Rollout depth |
| `compare_separate_h_distributions.py` | `actions_h{h}` | Condition label |

---

## 다음 단계에서 h는?

### Step 0.3: AIRL

**h will be**:
- Latent variable to infer from behavior
- Input to discriminator: D(s, a, h)
- Test: Can we recover h from trajectories?

```python
# AIRL discriminator
def discriminator(state, action, h):
    """
    h를 명시적으로 conditioning

    Goal: Learn r(s,a,h) instead of r(s,a)
    """
    pass
```

This is where h becomes truly powerful for IRL!
