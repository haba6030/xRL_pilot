# AIRL 적용 설계: 4-in-a-row Planning-Aware IRL

**작성일:** 2024-12-23
**목적:** Pedestrian 프로젝트 AIRL 구조를 참고하여 4-in-a-row에 적용 가능한 설계 도출

---

## 1. Pedestrian 프로젝트 AIRL 구조 분석

### A. 핵심 컴포넌트

```python
# 1. Environment (Gymnasium)
env = PedestrianEnv()  # Custom environment
env = InfiniteHorizonEnvWrapper(env)  # or FixedHorizonEnvWrapper

# 2. Expert Demonstrations
demonstrations = load_traj(subjId)  # List[TrajectoryWithRew]
# TrajectoryWithRew = (obs, acts, rews, infos, terminal)

# 3. Reward Network (Discriminator)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,  # Box(113,)
    action_space=env.action_space,            # Discrete(9)
    hid_sizes=[32, 32],                       # [32, 32] or [256, 256, 256]
    activation=nn.Tanh                        # or nn.LeakyReLU
)
# Input: (state, action) → Output: reward

# 4. Generator (Policy)
gen_algo = PPO(
    "MlpPolicy",
    env,
    n_steps=512,
    batch_size=32,
    ...
)
# Neural network policy

# 5. AIRL Trainer
trainer = airl.AIRL(
    demonstrations=demonstrations,
    demo_batch_size=256,
    venv=env,
    gen_algo=gen_algo,
    reward_net=reward_net,
    n_disc_updates_per_round=2,
    gen_train_timesteps=512,
    ...
)

# 6. Training
trainer.train(total_timesteps=4000 * 512)
```

### B. 데이터 형식

**Expert trajectories:**
```python
TrajectoryWithRew(
    obs=np.array([[obs_0], [obs_1], ..., [obs_T]]),     # shape: (T+1, obs_dim)
    acts=np.array([act_0, act_1, ..., act_{T-1}]),      # shape: (T,)
    rews=np.array([rew_0, rew_1, ..., rew_{T-1}]),      # shape: (T,)
    infos=[info_0, info_1, ..., info_{T-1}],            # shape: (T,)
    terminal=True
)
```

### C. 핵심 통찰

1. **Imitation library 사용**: `from imitation.algorithms.adversarial import airl`
2. **Discriminator = BasicRewardNet**: Neural network (state, action) → reward
3. **Generator = PPO**: Neural network policy (학습 가능)
4. **Environment 필수**: Gymnasium interface 필요
5. **Variable horizon 지원**: `allow_variable_horizon=True`

---

## 2. 4-in-a-row AIRL 적용 가능성 판단

### ✅ 적용 가능! (조건부)

**근거:**
1. **State 정의 가능**: Board state (6×6 bitboard)
2. **Action 정의 가능**: Move (0-35)
3. **Expert data 있음**: opendata/raw_data.csv (40명, 67K trials)
4. **Trajectory 구성 가능**: 게임별 (state, action) 시퀀스

### ⚠️ 주요 도전 과제

#### **Challenge 1: BFS Policy의 비미분성**

**문제:**
```python
# Pedestrian: PPO (neural network) → Gradient-based update
policy_net = MLP(obs) → action_logits
# Back-propagation 가능

# 4-in-a-row: BFS (symbolic algorithm) → No gradients
policy_bfs = BFS(board, h, beta, lapse) → action
# Back-propagation 불가능!
```

**해결 방안:**
1. **Option A**: BFS → Neural network distillation
2. **Option B**: Hybrid (neural + BFS)
3. **Option C**: Direct parameter optimization (CEM, BADS)

#### **Challenge 2: Van Opheusden Heuristic 활용**

**문제:**
```python
# Van Opheusden: 17 hand-crafted features
heuristic_value = sum([w_i * feature_i for i in range(17)])

# AIRL: Learned reward (neural network)
reward = reward_net(state, action)

# 충돌: 어떻게 통합?
```

**해결 방안:**
1. **Option A**: Heuristic으로 reward_net initialization
2. **Option B**: Heuristic를 auxiliary loss로
3. **Option C**: Pure AIRL (heuristic 버림)

#### **Challenge 3: Environment 구현**

**문제:**
- 4-in-a-row는 2-player game
- 현재 C++ 구현만 있음
- Gymnasium interface 필요

**해결 방안:**
- Python Gymnasium environment 구현 필요

---

## 3. 제안 설계: Planning-Aware AIRL

### A. 전체 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                 Planning-Aware AIRL                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Expert Data]                                       │
│    ↓                                                 │
│  Board states + Actions (from opendata)              │
│    ↓                                                 │
│  ┌──────────────────────────────────────┐           │
│  │   Discriminator (Reward Network)      │           │
│  │   Input: (board_state, action, h)     │           │
│  │   Output: reward estimate             │           │
│  └──────────────────────────────────────┘           │
│    ↓                                                 │
│  ┌──────────────────────────────────────┐           │
│  │   Generator (h-constrained policy)    │           │
│  │   Option A: PPO (neural)              │           │
│  │   Option B: BFS (fixed-h) + distill   │           │
│  └──────────────────────────────────────┘           │
│    ↓                                                 │
│  Adversarial Training                                │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### B. State Representation

**Option 1: Board Encoding (간단)**
```python
# 72-dimensional vector
state = np.concatenate([
    black_pieces,  # 36-dim (6×6 binary)
    white_pieces,  # 36-dim (6×6 binary)
])
# Total: 72-dim
```

**Option 2: Feature-based (Van Opheusden)**
```python
# 17 + 72 = 89-dimensional
state = np.concatenate([
    board_encoding,       # 72-dim (raw board)
    heuristic_features,   # 17-dim (center, 2/3/4-in-a-row, ...)
])
# Total: 89-dim

# Features:
# - center_control: count pieces in center
# - connected_2_in_a_row: count connected pairs
# - unconnected_2_in_a_row: count unconnected pairs
# - 3_in_a_row: count triplets
# - 4_in_a_row: count winning states
# ... (17 features total)
```

**Option 3: CNN Embedding (학습)**
```python
# Raw board as 2-channel image
state = np.stack([
    black_pieces.reshape(6, 6),  # Channel 0
    white_pieces.reshape(6, 6),  # Channel 1
], axis=0)
# Shape: (2, 6, 6)

# CNN encoder
cnn_encoder = nn.Sequential(
    nn.Conv2d(2, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 6 * 6, 128)
)
state_embedding = cnn_encoder(state)  # 128-dim
```

**추천: Option 2 (Feature-based)**
- Van Opheusden의 domain knowledge 활용
- Reward network initialization 가능
- 해석 가능성 유지

### C. Action Space

```python
action_space = gym.spaces.Discrete(36)
# 0-35: board positions (6×6)
```

**Legal action masking:**
```python
def get_legal_actions(board_state):
    """Return mask of legal actions"""
    black_pieces, white_pieces = board_state[:36], board_state[36:]
    occupied = (black_pieces | white_pieces)
    legal_mask = 1 - occupied  # 1=legal, 0=illegal
    return legal_mask  # shape: (36,)
```

### D. Environment 구현

```python
import gymnasium as gym
import numpy as np

class FourInARowEnv(gym.Env):
    """4-in-a-row Gymnasium Environment"""

    def __init__(self):
        super().__init__()

        # State: board encoding (72-dim) + features (17-dim)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(89,),  # 72 + 17
            dtype=np.float32
        )

        # Action: move position (0-35)
        self.action_space = gym.spaces.Discrete(36)

        # Internal state
        self.black_pieces = np.zeros(36, dtype=np.float32)
        self.white_pieces = np.zeros(36, dtype=np.float32)
        self.current_player = 0  # 0=black, 1=white

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.black_pieces = np.zeros(36, dtype=np.float32)
        self.white_pieces = np.zeros(36, dtype=np.float32)
        self.current_player = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # 1. Apply action
        if self.current_player == 0:  # Black
            self.black_pieces[action] = 1.0
        else:  # White
            self.white_pieces[action] = 1.0

        # 2. Check win/draw
        terminated = self._check_win() or self._check_draw()

        # 3. Compute reward (placeholder)
        reward = 1.0 if self._check_win() else 0.0

        # 4. Switch player
        self.current_player = 1 - self.current_player

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, False, info

    def _get_obs(self):
        """Construct observation: board + features"""
        # Board encoding
        board = np.concatenate([self.black_pieces, self.white_pieces])

        # Heuristic features (17-dim)
        features = self._extract_features()

        obs = np.concatenate([board, features])
        return obs.astype(np.float32)

    def _extract_features(self):
        """Extract 17 Van Opheusden features"""
        # TODO: Implement feature extraction
        # - center_control
        # - connected_2/3/4_in_a_row
        # - ...
        return np.zeros(17, dtype=np.float32)

    def _check_win(self):
        """Check 4-in-a-row for current player"""
        # TODO: Implement win check
        return False

    def _check_draw(self):
        """Check if board is full"""
        return np.sum(self.black_pieces + self.white_pieces) >= 36

    def get_legal_actions(self):
        """Return mask of legal actions"""
        occupied = self.black_pieces + self.white_pieces
        return 1 - occupied  # 1=legal, 0=illegal
```

### E. Expert Demonstrations 변환

```python
def load_expert_trajectories(participant_id):
    """
    Load expert data from opendata/raw_data.csv
    Convert to imitation library format
    """
    import pandas as pd
    from imitation.data.types import TrajectoryWithRew

    # Load raw data
    raw_data = pd.read_csv('opendata/raw_data.csv')
    participant_data = raw_data[raw_data['participant'] == participant_id]

    # Group by game (need to reconstruct game sequences)
    # TODO: 게임 단위로 grouping 필요 (현재 raw_data에는 게임 ID 없음)

    trajectories = []

    # For each game:
    for game in games:
        observations = []
        actions = []
        rewards = []
        infos = []

        # Initial state
        black_pieces = np.zeros(36, dtype=np.float32)
        white_pieces = np.zeros(36, dtype=np.float32)
        current_player = 0

        for trial in game:
            # Current observation
            obs = construct_observation(black_pieces, white_pieces)
            observations.append(obs)

            # Action
            action = trial['move']
            actions.append(action)

            # Update board
            if current_player == 0:
                black_pieces[action] = 1.0
            else:
                white_pieces[action] = 1.0

            # Reward (placeholder)
            reward = 0.0
            rewards.append(reward)

            # Info
            info = {'response_time': trial['response_time']}
            infos.append(info)

            # Switch player
            current_player = 1 - current_player

        # Final observation
        final_obs = construct_observation(black_pieces, white_pieces)
        observations.append(final_obs)

        # Create trajectory
        traj = TrajectoryWithRew(
            obs=np.array(observations),
            acts=np.array(actions),
            rews=np.array(rewards),
            infos=infos,
            terminal=True
        )
        trajectories.append(traj)

    return trajectories

def construct_observation(black_pieces, white_pieces):
    """Construct observation from board state"""
    board = np.concatenate([black_pieces, white_pieces])
    features = extract_features(black_pieces, white_pieces)  # 17-dim
    obs = np.concatenate([board, features])
    return obs.astype(np.float32)
```

### F. Discriminator (Reward Network)

**Option 1: Pure Neural Network**
```python
reward_net = BasicRewardNet(
    observation_space=env.observation_space,  # Box(89,)
    action_space=env.action_space,            # Discrete(36)
    hid_sizes=[128, 128, 128],
    activation=nn.Tanh
)
# Input: (state, action) → Output: reward
```

**Option 2: Van Opheusden Initialization**
```python
class HeuristicInitializedRewardNet(nn.Module):
    """
    Reward network initialized with Van Opheusden heuristic
    """
    def __init__(self, obs_dim, action_dim, hid_sizes, heuristic_weights):
        super().__init__()

        # Feature extractor (board → 17 features)
        self.feature_extractor = nn.Linear(72, 17, bias=False)
        # Initialize with Van Opheusden feature weights
        with torch.no_grad():
            self.feature_extractor.weight.data = torch.tensor(
                heuristic_weights, dtype=torch.float32
            )

        # MLP (features + action → reward)
        self.mlp = nn.Sequential(
            nn.Linear(17 + action_dim, hid_sizes[0]),
            nn.Tanh(),
            nn.Linear(hid_sizes[0], hid_sizes[1]),
            nn.Tanh(),
            nn.Linear(hid_sizes[1], 1)
        )

    def forward(self, state, action):
        # Extract board
        board = state[:, :72]  # (batch, 72)

        # Extract features (initialized with heuristic)
        features = self.feature_extractor(board)  # (batch, 17)

        # One-hot action
        action_onehot = F.one_hot(action, num_classes=36)  # (batch, 36)

        # Concatenate
        x = torch.cat([features, action_onehot], dim=-1)

        # MLP
        reward = self.mlp(x)
        return reward
```

**추천: Option 2 (Heuristic Initialization)**
- Van Opheusden의 domain knowledge 활용
- Warm start (학습 속도 향상)
- Interpretability

### G. Generator (Policy)

**핵심 문제: BFS는 미분 불가능!**

**Option A: Pure Neural Network (Pedestrian 방식)**
```python
gen_algo = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=32,
    ...
)
# 완전히 학습 가능
# BFS 버림 (Van Opheusden 기여 무시)
```

**Option B: BFS Distillation (추천) ⭐**
```python
class BFSDistilledPolicy(nn.Module):
    """
    Neural network trained to mimic BFS with fixed h
    """
    def __init__(self, obs_dim, action_dim, h, hid_sizes):
        super().__init__()
        self.h = h  # Fixed planning depth

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hid_sizes[0]),
            nn.Tanh(),
            nn.Linear(hid_sizes[0], hid_sizes[1]),
            nn.Tanh(),
            nn.Linear(hid_sizes[1], action_dim)
        )

    def forward(self, obs):
        logits = self.policy_net(obs)
        return logits

    @staticmethod
    def pretrain_from_bfs(env, h, beta, lapse, heuristic_params):
        """
        Pre-train policy by imitating BFS outputs
        """
        # 1. Generate BFS rollouts
        bfs_policy = BFS_fixed_h(h=h, beta=beta, lapse=lapse,
                                  heuristic=heuristic_params)

        # 2. Collect (state, action) pairs
        dataset = []
        for _ in range(10000):  # 10K episodes
            obs = env.reset()
            done = False
            while not done:
                # BFS action
                action = bfs_policy.predict(obs)
                dataset.append((obs, action))

                obs, _, done, _, _ = env.step(action)

        # 3. Train neural network via behavior cloning
        policy = BFSDistilledPolicy(obs_dim, action_dim, h, hid_sizes)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        for epoch in range(100):
            for obs, action in dataset:
                logits = policy(obs)
                loss = F.cross_entropy(logits, action)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return policy

# Usage in AIRL
policy = BFSDistilledPolicy.pretrain_from_bfs(env, h=6, ...)
gen_algo = CustomPPO(policy, env, ...)  # Fine-tune with PPO
```

**Option C: Hybrid (Neural + BFS) - 실험적**
```python
class HybridPolicy(nn.Module):
    """
    Combine neural network and BFS
    """
    def __init__(self, obs_dim, action_dim, h):
        super().__init__()
        self.h = h

        # Neural component
        self.neural_net = nn.Sequential(...)

        # BFS component (non-differentiable)
        self.bfs_policy = BFS_fixed_h(h=h)

        # Mixing parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, obs):
        # Neural logits
        neural_logits = self.neural_net(obs)

        # BFS action (detached, no gradients)
        with torch.no_grad():
            bfs_action = self.bfs_policy.predict(obs)
            bfs_logits = torch.zeros_like(neural_logits)
            bfs_logits[bfs_action] = 10.0  # High logit for BFS action

        # Mix
        logits = self.alpha * neural_logits + (1 - self.alpha) * bfs_logits
        return logits
```

**추천: Option B (BFS Distillation)**
- Van Opheusden BFS 활용
- Gradient-based 학습 가능
- Fixed h 명시적으로 표현

---

## 4. Planning-Aware AIRL: h를 Latent Variable로

### A. 핵심 아이디어

**Standard AIRL:**
```python
# h 무시
reward_net = RewardNet(state, action) → reward
```

**Planning-Aware AIRL:**
```python
# h를 condition으로
reward_net = RewardNet(state, action, h) → reward

# h별로 별도 학습
for h in [2, 4, 6, 8, 10]:
    reward_net_h = RewardNet_h(state, action, h=h)
    policy_h = BFSDistilledPolicy(h=h)

    trainer = AIRL(demonstrations, policy_h, reward_net_h)
    trainer.train()
```

### B. Reward Network with h

```python
class PlanningAwareRewardNet(nn.Module):
    """
    Reward network conditioned on planning depth h
    """
    def __init__(self, obs_dim, action_dim, hid_sizes, num_h_values=5):
        super().__init__()

        # h embedding
        self.h_embedding = nn.Embedding(
            num_embeddings=num_h_values,  # h ∈ {2,4,6,8,10}
            embedding_dim=8
        )

        # MLP (state + action + h_embedding → reward)
        input_dim = obs_dim + action_dim + 8
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hid_sizes[0]),
            nn.Tanh(),
            nn.Linear(hid_sizes[0], hid_sizes[1]),
            nn.Tanh(),
            nn.Linear(hid_sizes[1], 1)
        )

    def forward(self, state, action, h_idx):
        """
        Args:
            state: (batch, obs_dim)
            action: (batch,) or (batch, action_dim)
            h_idx: (batch,) - index of h value (0-4 for {2,4,6,8,10})
        """
        # Embed h
        h_emb = self.h_embedding(h_idx)  # (batch, 8)

        # One-hot action (if discrete)
        if action.dim() == 1:
            action_onehot = F.one_hot(action, num_classes=36)
        else:
            action_onehot = action

        # Concatenate
        x = torch.cat([state, action_onehot, h_emb], dim=-1)

        # Predict reward
        reward = self.mlp(x)
        return reward
```

### C. 학습 절차 (Actual Implementation)

**Implementation**: Steps A-E (see IMPLEMENTATION_NOTES.md for details)

```python
def train_planning_aware_airl():
    """
    Planning-Aware AIRL for 4-in-a-row (Implemented)
    
    Uses BC (Behavior Cloning) approach instead of BFS distillation.
    See fourinarow_airl/train_airl.py for complete implementation.
    """
    # 1. Load or generate expert data
    expert_trajectories = load_expert_trajectories(participant_id)
    # OR generate synthetic expert data
    expert_game_trajs = generate_depth_limited_trajectories(h=4, num_episodes=100)
    expert_trajectories = convert_to_imitation_format(expert_game_trajs)
    
    # 2. Setup environment
    env = FourInARowEnv()
    
    results = {}
    
    # 3. Train for each h
    for h in [1, 2, 4, 8]:
        print(f"\n=== Training with h={h} ===")
        
        # 3.1. Load BC-initialized PPO generator (Steps A-C)
        # Step A: generate_depth_limited_trajectories(h) 
        # Step B: train_bc_policy()
        # Step C: create_ppo_from_bc()
        gen_algo = load_ppo_generator(h=h)
        
        # 3.2. Create depth-AGNOSTIC reward network (Step D)
        # CRITICAL: NO h parameter!
        reward_net = create_reward_network(env)
        
        # 3.3. AIRL trainer (Step E)
        venv = DummyVecEnv([lambda: env])
        trainer = airl.AIRL(
            demonstrations=expert_trajectories,  # NO h labels!
            demo_batch_size=64,
            venv=venv,
            gen_algo=gen_algo,                   # h-specific (BC-initialized)
            reward_net=reward_net,               # depth-agnostic!
            n_disc_updates_per_round=4,
            gen_train_timesteps=512,
            allow_variable_horizon=True,         # 4-in-a-row games vary in length
        )
        
        # 3.4. Train
        trainer.train(total_timesteps=50000)
        
        # 3.5. Save results
        results[h] = {
            'reward_net': reward_net,
            'policy': gen_algo,
            'trainer': trainer
        }
    
    # 4. Model selection (best h)
    # Compare discriminator metrics across h values
    for h in [1, 2, 4, 8]:
        print(f"h={h}: disc_acc={results[h]['trainer'].disc_acc:.3f}")
    
    return results
```

**Key Differences from Initial Design**:

| Aspect | Initial Design | Actual Implementation |
|--------|----------------|----------------------|
| **Generator** | BFS Distillation | BC (Behavior Cloning) |
| **Reward Network** | h-conditioned | Completely depth-agnostic |
| **Depths** | [2, 4, 6, 8, 10] | [1, 2, 4, 8] |
| **Approach** | Direct BFS → NN | Policy → Trajectories → BC → PPO |

**Why BC was chosen**:
1. Simpler implementation using imitation library
2. No need for C++ BFS wrapper
3. Same goal achieved: neural policy mimics h-specific behavior
4. Faster to implement and test

### D. Discriminator 활용

**Discriminator의 역할:**
```python
# Discriminator = Reward Network
D(s, a, h) = sigmoid(f(s, a, h))

# f(s, a, h) = r(s, a, h) + γV(s') - V(s)
# where r(s, a, h) = learned reward conditioned on h
```

**Training:**
```python
# 1. Expert data: (s, a, s')_expert
# 2. Generated data: (s, a, s')_generated from policy_h

# Discriminator loss (binary classification)
loss_D = -E_expert[log D(s,a,h)] - E_gen[log(1 - D(s,a,h))]

# Policy loss (fool discriminator)
loss_G = -E_gen[log D(s,a,h)]
```

**Discriminator metrics (중요!):**
```python
# Training callback에서 모니터링
metrics = {
    'disc_acc': overall accuracy,
    'disc_acc_expert': accuracy on expert data,
    'disc_acc_gen': accuracy on generated data
}

# Good training:
# - disc_acc_expert ≈ disc_acc_gen ≈ 0.5 (균형)
# - 너무 높으면 (>0.9): discriminator 승리 → policy 학습 어려움
# - 너무 낮으면 (<0.2): policy 승리 → reward 의미 없음
```

---

## 5. 구현 로드맵

### Phase 1: Environment 구현 ✅
```python
# TODO:
# 1. FourInARowEnv 클래스 작성
# 2. Board state → observation 변환
# 3. Van Opheusden 17 features 구현
# 4. Win/draw 체크
# 5. Legal action masking

# 예상 소요: 1-2주
```

### Phase 2: Expert Data 변환 ✅
```python
# TODO:
# 1. raw_data.csv 파싱 (게임 단위 reconstruction)
# 2. TrajectoryWithRew 형식으로 변환
# 3. Data validation (완전한 게임만)
# 4. Train/test split

# 예상 소요: 1주
```

### Phase 3: BFS Distillation ✅
```python
# TODO:
# 1. C++ BFS wrapper (Python interface)
# 2. BFS rollout 생성 (각 h별)
# 3. Behavior cloning (neural → BFS)
# 4. Validation (neural policy ≈ BFS?)

# 예상 소요: 2-3주
```

### Phase 4: AIRL Training ✅
```python
# TODO:
# 1. Reward network 설계 (with h conditioning)
# 2. AIRL trainer setup
# 3. Training loop (각 h별)
# 4. Discriminator metrics 모니터링
# 5. Model selection (best h)

# 예상 소요: 2-3주
```

### Phase 5: Evaluation ✅
```python
# TODO:
# 1. OOD generalization test
# 2. Reward visualization
# 3. Counterfactual analysis (h 변경)
# 4. Expert vs Novice 비교

# 예상 소요: 2주
```

**총 예상 소요: 2-3개월**

---

## 6. 예상 결과 및 기여

### A. 기대 결과

**1. h별 Reward 분리**
```python
reward_2 = reward_net(state, action, h=2)  # Shallow planning reward
reward_10 = reward_net(state, action, h=10) # Deep planning reward

# Analysis:
# "h=2에서 center control이 더 중요"
# "h=10에서 3-in-a-row가 더 중요"
```

**2. 최적 h 분포**
```python
# Expert vs Novice
expert_best_h = [2, 4, 2, 4, 6, ...]  # 주로 2, 4
novice_best_h = [6, 8, 10, 6, 8, ...]  # 주로 6, 8, 10

# Chi-square test: p < 0.001
# Expert는 shallow planning 선호
```

**3. Discriminator Accuracy**
```python
# Well-trained:
# disc_acc_expert ≈ 0.52
# disc_acc_gen ≈ 0.48
# → Generator가 expert처럼 행동

# h=2: disc_acc = 0.50 (best match for expert)
# h=10: disc_acc = 0.70 (discriminator가 구분 가능)
```

### B. 연구 기여

**1. Reward + Planning 분리**
- Van Opheusden: Heuristic (reward) + BFS (planning) 혼재
- 제안: h-conditioned reward로 분리

**2. Yao (2024) 주장 검증**
- "Planning horizon은 latent confounder"
- h를 명시하면 reward 추론 정확

**3. Counterfactual Analysis**
```python
# "만약 Expert가 h=10으로 plan했다면?"
behavior_counterfactual = policy_h10.generate(expert_board)
# → 성능 저하 예상
```

**4. OOD Generalization**
- 새로운 board 상황에 적용
- Transfer learning (다른 게임?)

---

## 7. 미해결 이슈 및 대안

### Issue 1: 게임 단위 Reconstruction

**문제:**
- raw_data.csv에 게임 ID 없음
- Trial별 board state는 있지만 게임 경계 불명

**해결:**
```python
# Option A: Session 단위로 가정
# - 같은 participant, 같은 session → 하나의 게임?

# Option B: Board state 추적
# - Empty board → 새 게임 시작
# - Win/draw 발생 → 게임 종료

# Option C: data_hvh.txt 활용
# - C++ data_struct.h의 board history
```

### Issue 2: Reward Ground Truth 부재

**문제:**
- AIRL은 reward를 학습하지만 ground truth 없음
- Van Opheusden heuristic도 learned (fitted)

**해결:**
```python
# Validation:
# 1. Win/loss outcome으로 검증
# 2. Human preference로 검증
# 3. Cross-validation (held-out games)
```

### Issue 3: 2-Player Game 처리

**문제:**
- Expert는 Black만 플레이 (White는 AI)
- AIRL은 single-agent 가정

**해결:**
```python
# Option A: Black만 모델링
# - Expert의 Black moves만 학습
# - White는 환경의 일부로

# Option B: Both players
# - Black + White 모두 학습
# - Two-player AIRL (복잡)
```

---

## 8. 최종 판단

### ✅ AIRL 적용 가능!

**조건:**
1. **Environment 구현**: Gymnasium interface (2주)
2. **BFS Distillation**: Neural policy pre-training (3주)
3. **Data Reconstruction**: 게임 단위 trajectory (1주)

### 핵심 설계 결정

**1. State: Feature-based (72 + 17 = 89-dim)**
- Board encoding + Van Opheusden features

**2. Generator: BFS Distillation**
- Pre-train neural policy to mimic BFS
- Fine-tune with PPO in AIRL

**3. Discriminator: h-conditioned Reward Network**
- Input: (state, action, h)
- Output: reward
- Initialize with Van Opheusden heuristic

**4. Training: h별로 별도 학습**
- h ∈ {2, 4, 6, 8, 10}
- Model selection (best h per participant)

### 다음 단계

1. **Prototype**: FourInARowEnv 구현 및 테스트
2. **Data Pipeline**: raw_data.csv → TrajectoryWithRew
3. **BFS Wrapper**: Python interface for C++ BFS
4. **Pilot Study**: 1명 참가자로 전체 파이프라인 검증
5. **Full Training**: 40명 참가자 AIRL

---


## 9. Implementation Technical Notes

**See**: [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) for detailed technical documentation including:

- Environment setup and library versions
- BasicRewardNet usage patterns
- imitation 1.0.1 API details
- Architecture matching (BC → PPO)
- Data formats and dimensions
- AIRL training metrics interpretation
- Common issues and solutions
- Training recommendations

**Implementation Status** (2025-12-25):
- ✅ Steps A-E Complete (generate data → BC → PPO → reward network → AIRL training)
- 🔄 Steps F-G Pending (multi-depth comparison, evaluation)
- ✅ Core principle maintained: h only in POLICY, never in reward network

---

**Last Updated**: 2025-12-25
**Status**: Design document (initial approach in Sections 1-8), actual implementation in Steps A-E
