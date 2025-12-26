# Gymnasium 환경과 AIRL 완전 이해 가이드

## 목표

Phase 2 AIRL 파이프라인 구현을 위한 완전한 이해를 제공합니다.

---

# Part 1: Gymnasium 환경 이해

## 1.1 Gymnasium이란?

**Gymnasium** (구 OpenAI Gym)은 강화학습 환경의 표준 인터페이스입니다.

### 핵심 개념

```
Agent ↔ Environment

Agent가 action을 선택
    ↓
Environment가 (observation, reward, done) 반환
    ↓
Agent가 다음 action 선택
    ↓
반복...
```

### 표준 API

모든 Gymnasium 환경은 다음 메서드를 제공:

```python
class Env:
    def reset(self, seed=None) -> (observation, info):
        """새 에피소드 시작"""
        pass

    def step(self, action) -> (observation, reward, terminated, truncated, info):
        """Action 실행, 결과 반환"""
        pass

    observation_space  # 관찰 공간 정의
    action_space       # 행동 공간 정의
```

---

## 1.2 우리 FourInARowEnv 구조

### 파일 위치
`fourinarow_airl/env.py`

### 전체 구조

```python
class FourInARowEnv(gym.Env):
    """
    4-in-a-row Gymnasium Environment

    State: 89-dim vector
        - 0-35: Black pieces (binary)
        - 36-71: White pieces (binary)
        - 72-88: Van Opheusden 17 features (normalized)

    Action: Discrete(36)
        - 0-35: Board positions

    Reward:
        - +1: Win
        - -1: Loss (실제로는 사용 안 함, 항상 현재 플레이어 기준 +1)
        - 0: Draw or ongoing
    """

    def __init__(self, render_mode=None):
        # [1] Space 정의
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(89,),  # 72 board + 17 features
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(36)

        # [2] State 변수
        self.black_pieces = np.zeros(36)  # 1 = 해당 위치에 Black 돌
        self.white_pieces = np.zeros(36)  # 1 = 해당 위치에 White 돌
        self.current_player = 0  # 0=Black, 1=White
        self.move_count = 0

    def reset(self, seed=None, options=None):
        """
        새 게임 시작

        Returns:
            observation: 89-dim state
            info: 추가 정보 (legal_actions 등)
        """
        super().reset(seed=seed)

        # 빈 보드로 초기화
        self.black_pieces = np.zeros(36)
        self.white_pieces = np.zeros(36)
        self.current_player = 0
        self.move_count = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """
        Action 실행

        Args:
            action: 0-35 (board position)

        Returns:
            observation: 89-dim state (next state)
            reward: +1 (win), 0 (ongoing/draw)
            terminated: True if game ended
            truncated: False (no time limit)
            info: dict with legal_actions
        """
        # [1] Validate action
        if not self._is_valid_action(action):
            # Invalid move → 즉시 종료, 음수 보상
            return self._get_observation(), -1.0, True, False, {}

        # [2] Apply action
        if self.current_player == 0:
            self.black_pieces[action] = 1.0
        else:
            self.white_pieces[action] = 1.0

        self.move_count += 1

        # [3] Check win/draw
        terminated = False
        reward = 0.0

        if self._check_win(action):
            terminated = True
            reward = 1.0  # 현재 플레이어가 승리
        elif self._is_board_full():
            terminated = True
            reward = 0.0  # 무승부

        # [4] Switch player
        self.current_player = 1 - self.current_player

        # [5] Return
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, False, info
```

### 핵심 메서드 상세

**`_get_observation()`**: 89-dim state 생성

```python
def _get_observation(self) -> np.ndarray:
    """
    89-dim observation 생성

    Returns:
        [0-35]: black_pieces
        [36-71]: white_pieces
        [72-88]: Van Opheusden 17 features
    """
    from .features import extract_van_opheusden_features

    features = extract_van_opheusden_features(
        self.black_pieces,
        self.white_pieces,
        self.current_player
    )

    obs = np.concatenate([
        self.black_pieces,   # 36
        self.white_pieces,   # 36
        features             # 17
    ]).astype(np.float32)

    return obs  # shape: (89,)
```

**`_check_win(last_move)`**: 4개 연속 확인

```python
def _check_win(self, last_move: int) -> bool:
    """
    마지막 수가 승리인지 확인

    4가지 방향 체크:
    - Horizontal (0, 1)
    - Vertical (1, 0)
    - Diagonal (1, 1)
    - Anti-diagonal (1, -1)
    """
    row = last_move // 6
    col = last_move % 6

    pieces = (self.black_pieces if self.current_player == 0
              else self.white_pieces)
    board = pieces.reshape(6, 6)

    # 각 방향에 대해 _check_line() 호출
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    for dr, dc in directions:
        if self._check_line(board, row, col, dr, dc):
            return True

    return False
```

---

## 1.3 Gymnasium Environment 사용 예시

### 기본 사용법

```python
from fourinarow_airl import FourInARowEnv

# [1] 환경 생성
env = FourInARowEnv()

# [2] 에피소드 시작
obs, info = env.reset(seed=42)
print(f"Initial observation shape: {obs.shape}")  # (89,)
print(f"Legal actions: {info['legal_actions']}")  # [0,1,2,...,35]

# [3] 게임 진행
done = False
while not done:
    # Action 선택 (여기서는 random)
    legal_actions = env.get_legal_actions()
    action = np.random.choice(legal_actions)

    # Step
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    if done:
        print(f"Game ended! Reward: {reward}")
```

### State Clone (Planning용)

```python
import copy

# [1] 현재 상태 저장
env = FourInARowEnv()
env.reset()
env.step(17)  # 몇 수 진행

# [2] Clone
env_clone = copy.deepcopy(env)

# [3] Clone에서 시뮬레이션
obs_sim, _, _, _, _ = env_clone.step(23)

# [4] 원본은 그대로
print(f"Original move count: {env.move_count}")    # 1
print(f"Clone move count: {env_clone.move_count}") # 2
```

이것이 `DepthLimitedPolicy`에서 h-step lookahead를 가능하게 합니다.

---

# Part 2: AIRL 작동 원리

## 2.1 IRL (Inverse Reinforcement Learning) 개념

### 일반 RL vs IRL

**일반 RL (Forward RL)**:
```
Given: Reward function R(s, a)
Learn: Policy π(a|s) that maximizes E[Σ R(s,a)]
```

**IRL (Inverse RL)**:
```
Given: Expert demonstrations τ_expert = {(s,a,s')...}
Learn: Reward function R(s,a) that explains expert behavior
```

### IRL의 어려움: Ambiguity

같은 behavior를 설명하는 reward는 무한히 많음:
- R(s,a) = 0 for all s,a → 모든 policy가 optimal
- R(s,a) + constant → 똑같은 policy
- Linear scaling: c·R(s,a)

---

## 2.2 AIRL (Adversarial IRL)

**핵심 아이디어**: GAN처럼 discriminator와 generator를 경쟁시켜 reward를 학습

### 구조

```
┌─────────────────────────────────────────┐
│ Expert Demonstrations                   │
│ τ_expert = {(s₀,a₀,s₁), (s₁,a₁,s₂)...} │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         Discriminator D(s,a,s')         │
│  (실제로는 Reward Network r_θ(s,a,s')) │
│                                         │
│  목표: Expert vs Generated 구분         │
│  D(expert) → 1                          │
│  D(generated) → 0                       │
└─────────────────────────────────────────┘
              ↓ reward signal
┌─────────────────────────────────────────┐
│      Generator Policy π_φ(a|s)          │
│   (PPO, SAC 등 RL algorithm 사용)       │
│                                         │
│  목표: Discriminator를 속이기            │
│  즉, Expert처럼 행동하기                 │
└─────────────────────────────────────────┘
              ↓ generates trajectories
┌─────────────────────────────────────────┐
│    Generated Trajectories τ_gen        │
└─────────────────────────────────────────┘
```

### AIRL 학습 루프

```python
# Pseudocode
for iteration in range(num_iterations):
    # [1] Rollout: Generator로 trajectories 생성
    τ_gen = rollout(policy_π, env, num_steps)

    # [2] Discriminator 학습
    # Expert trajectories를 1로, Generated를 0으로 분류
    for (s,a,s') in τ_expert:
        loss_D += -log(D(s,a,s'))  # Expert → 1

    for (s,a,s') in τ_gen:
        loss_D += -log(1 - D(s,a,s'))  # Generated → 0

    update(discriminator_params, loss_D)

    # [3] Generator 학습
    # Discriminator가 주는 reward로 RL 학습
    reward_from_D = log(D(s,a,s')) - log(1-D(s,a,s'))
    update(policy_π, reward=reward_from_D)  # PPO, SAC 등 사용
```

### Discriminator = Reward Network

AIRL에서 discriminator의 출력은 곧 learned reward입니다:

```python
D(s,a,s') = exp(r(s,a,s')) / [exp(r(s,a,s')) + π(a|s)]

# Optimal discriminator일 때:
r(s,a,s') ≈ log(π_expert(a|s)) - log(π_gen(a|s))
```

즉, discriminator를 학습하는 것 = reward function을 학습하는 것

---

## 2.3 Imitation Library의 AIRL 구현

### 파일 참조

우리가 참고한 pedestrian project:
`/Users/jinilkim/.../project_pedestrian/analysis/irl/airl.py`

### 핵심 컴포넌트

**1. RewardNet (Discriminator)**

```python
from imitation.rewards.reward_nets import BasicRewardNet

# Reward network 생성
reward_net = BasicRewardNet(
    observation_space=env.observation_space,  # Box(89,)
    action_space=env.action_space,            # Discrete(36)
    hid_sizes=[64, 64],  # MLP hidden layers
    activation=nn.ReLU
)

# Forward pass
# Input: state (89,), action (36,), next_state (89,)
# Output: reward (scalar)
reward = reward_net(state, action, next_state)
```

**내부 구조**:
```python
class BasicRewardNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hid_sizes):
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + action_dim + obs_dim, hid_sizes[0]),
            nn.ReLU(),
            nn.Linear(hid_sizes[0], hid_sizes[1]),
            nn.ReLU(),
            nn.Linear(hid_sizes[1], 1)  # Output: scalar reward
        )

    def forward(self, state, action, next_state):
        # action을 one-hot으로 변환 (Discrete action space)
        action_onehot = F.one_hot(action, num_classes=36)

        # Concatenate
        x = torch.cat([state, action_onehot, next_state], dim=-1)

        # MLP
        reward = self.mlp(x)

        return reward  # shape: (batch, 1)
```

**2. Generator (RL Algorithm)**

```python
from stable_baselines3 import PPO

# Generator policy 생성
gen_algo = PPO(
    "MlpPolicy",      # Multi-layer perceptron policy
    env,              # Gymnasium environment
    learning_rate=3e-4,
    n_steps=2048,     # Rollout buffer size
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)
```

**3. AIRL Trainer**

```python
from imitation.algorithms.adversarial import airl

trainer = airl.AIRL(
    demonstrations=expert_trajectories,  # List[Trajectory]
    demo_batch_size=256,                 # Expert batch size

    venv=env,                            # Vectorized env (여기서는 단일 env)
    gen_algo=gen_algo,                   # PPO policy

    reward_net=reward_net,               # Discriminator

    # Discriminator training
    n_disc_updates_per_round=4,          # D를 몇 번 업데이트

    # Other hyperparameters
    demo_minibatch_size=64,
)

# Training
trainer.train(total_timesteps=100000)
```

### Training Loop 내부

```python
# AIRL의 실제 training loop (simplified)

for round in range(num_rounds):
    # [1] Generator rollout
    gen_samples = gen_algo.collect_rollouts(n_steps=2048)

    # [2] Discriminator training
    for _ in range(n_disc_updates_per_round):
        # Sample from expert and generated
        expert_batch = sample(expert_trajectories, batch_size)
        gen_batch = sample(gen_samples, batch_size)

        # Compute discriminator loss
        D_expert = discriminator(expert_batch)
        D_gen = discriminator(gen_batch)

        loss_D = -log(D_expert).mean() - log(1 - D_gen).mean()

        # Update discriminator
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    # [3] Generator training (PPO)
    # Use discriminator output as reward
    rewards = log(D_gen) - log(1 - D_gen)
    gen_algo.train(rewards=rewards)

    # [4] Metrics
    disc_acc = (D_expert > 0.5).mean()  # Expert 정확도
    gen_acc = (D_gen < 0.5).mean()      # Generated 정확도
```

---

# Part 3: 우리 프로젝트에서 AIRL 통합

## 3.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│               Expert Data Loading                       │
│  opendata/raw_data.csv → GameTrajectory objects         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                FourInARowEnv                            │
│  - 89-dim observation                                   │
│  - Discrete(36) action                                  │
│  - State clone support                                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│         FOR EACH h ∈ {1, 2, 4, 8}:                     │
│                                                         │
│  ┌──────────────────────────────────────────────┐      │
│  │  Generator: DepthLimitedPolicy(h=h)          │      │
│  │  - h-step lookahead                          │      │
│  │  - Wrapped by PPO for RL training            │      │
│  └──────────────────────────────────────────────┘      │
│                          ↓                              │
│  ┌──────────────────────────────────────────────┐      │
│  │  Discriminator: BasicRewardNet               │      │
│  │  - Input: (s, a, s') each 89-dim             │      │
│  │  - Output: reward (scalar)                   │      │
│  │  - NO h parameter! (same for all h)          │      │
│  └──────────────────────────────────────────────┘      │
│                          ↓                              │
│  ┌──────────────────────────────────────────────┐      │
│  │  AIRL Training                               │      │
│  │  - Discriminator learns to distinguish       │      │
│  │  - Generator learns to match expert          │      │
│  └──────────────────────────────────────────────┘      │
│                          ↓                              │
│  Save: reward_trained_with_h{h}_generator.pt           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Evaluation & Comparison                    │
│  - Which h achieves best discrimination?                │
│  - Which h best matches expert behavior?                │
│  - Can learned h predict expertise?                     │
└─────────────────────────────────────────────────────────┘
```

## 3.2 코드 레벨 통합 계획

### Phase 2a: Baseline AIRL (단일 h)

**Step 1**: Expert data를 imitation library format으로 변환

```python
from fourinarow_airl.data_loader import load_expert_trajectories
from imitation.data.types import Trajectory
import numpy as np

# Load expert trajectories
expert_trajs_raw = load_expert_trajectories(
    'opendata/raw_data.csv',
    player_filter=0,
    max_trajectories=100
)

# Convert to imitation format
expert_trajs_imitation = []
for traj_raw in expert_trajs_raw:
    traj_imitation = Trajectory(
        obs=traj_raw.observations,      # (T+1, 89)
        acts=traj_raw.actions,          # (T,)
        infos=None,
        terminal=True
    )
    expert_trajs_imitation.append(traj_imitation)
```

**Step 2**: Policy wrapper 만들기

```python
from fourinarow_airl.depth_limited_policy import DepthLimitedPolicy
from stable_baselines3.common.policies import BasePolicy

class DepthLimitedPolicyWrapper(BasePolicy):
    """
    DepthLimitedPolicy를 Stable-Baselines3 Policy로 wrapping

    이렇게 하면 PPO와 호환됨
    """
    def __init__(self, observation_space, action_space, h, **kwargs):
        super().__init__(observation_space, action_space)

        self.depth_policy = DepthLimitedPolicy(h=h, **kwargs)
        self.h = h

    def forward(self, obs, deterministic=False):
        """
        Stable-Baselines3가 요구하는 인터페이스

        Returns:
            actions, values, log_probs
        """
        # obs는 batch일 수 있음: (batch_size, 89)
        # 단순화를 위해 batch_size=1로 가정

        # Environment를 복원해야 함 (obs로부터)
        # 이 부분은 복잡함 → 대안: PPO를 직접 사용하지 말고
        # imitation library의 BC (Behavior Cloning) 사용

        pass  # TODO
```

**문제점**: DepthLimitedPolicy는 environment가 필요한데, Stable-Baselines3 policy는 observation만 받음.

**해결책**: 두 가지 접근

**Option A: BC (Behavior Cloning) 사전학습 → AIRL Fine-tuning**

```python
# Step 1: DepthLimitedPolicy로 trajectories 생성
depth_policy = DepthLimitedPolicy(h=4)
env = FourInARowEnv()

depth_trajs = []
for episode in range(100):
    obs, _ = env.reset()
    trajectory_obs = [obs]
    trajectory_acts = []

    done = False
    while not done:
        action, _ = depth_policy.select_action(env)
        obs, reward, terminated, truncated, _ = env.step(action)

        trajectory_obs.append(obs)
        trajectory_acts.append(action)
        done = terminated or truncated

    depth_trajs.append(Trajectory(
        obs=np.array(trajectory_obs),
        acts=np.array(trajectory_acts),
        infos=None,
        terminal=True
    ))

# Step 2: BC로 neural policy 학습 (depth policy 모방)
from imitation.algorithms import bc

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=depth_trajs,
)
bc_trainer.train(n_epochs=100)

# Step 3: 학습된 policy를 PPO로 감싸기
from stable_baselines3 import PPO

gen_policy = PPO(
    policy=bc_trainer.policy,  # BC로 사전학습된 policy
    env=env,
    learning_rate=3e-4,
)

# Step 4: AIRL training
reward_net = BasicRewardNet(...)

trainer = AIRL(
    demonstrations=expert_trajs_imitation,
    gen_algo=gen_policy,
    reward_net=reward_net,
)

trainer.train(total_timesteps=50000)
```

**Option B: 간단한 Random/Heuristic Policy로 시작**

```python
# Baseline AIRL test
# Generator를 단순 random policy로 시작

from stable_baselines3 import PPO

gen_algo = PPO("MlpPolicy", env, ...)

trainer = AIRL(
    demonstrations=expert_trajs,
    gen_algo=gen_algo,
    reward_net=reward_net,
)

trainer.train(...)

# Depth는 나중에 추가
```

---

## 3.3 Depth Integration 전략 (최종)

### 우리가 채택한 방법

```python
# For each depth h:

for h in [1, 2, 4, 8]:
    print(f"Training AIRL with planning depth h={h}")

    # [1] Generate training data using DepthLimitedPolicy
    #     (depth policy의 behavior를 neural net으로 distill)
    depth_policy = DepthLimitedPolicy(h=h, beta=1.0, lapse_rate=0.1)

    depth_trajectories = generate_trajectories(
        policy=depth_policy,
        env=env,
        num_episodes=100
    )

    # [2] BC (Behavior Cloning): Neural policy learns to mimic depth policy
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=depth_trajectories
    )
    bc_trainer.train(n_epochs=50)

    # [3] Wrap BC policy with PPO
    gen_algo = PPO(
        policy=bc_trainer.policy,  # ← 여기가 h-dependent
        env=env,
        learning_rate=3e-4
    )

    # [4] Create depth-AGNOSTIC reward network
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hid_sizes=[64, 64]
    )  # ← NO h parameter! Same for all h

    # [5] AIRL training
    trainer = AIRL(
        demonstrations=expert_trajectories,  # Expert data
        gen_algo=gen_algo,                   # h-dependent generator
        reward_net=reward_net,               # h-agnostic discriminator
        demo_batch_size=256,
        n_disc_updates_per_round=4,
    )

    trainer.train(total_timesteps=100000)

    # [6] Save
    torch.save(reward_net.state_dict(),
               f'models/reward_trained_with_h{h}_generator.pt')
    torch.save(gen_algo.policy.state_dict(),
               f'models/generator_h{h}.pt')

    # [7] Evaluate
    disc_acc = evaluate_discrimination(trainer, expert_trajectories)
    print(f"h={h}: Discrimination accuracy = {disc_acc:.3f}")
```

---

# Part 4: 구현 체크리스트

## Phase 2a: Baseline AIRL (h=4 하나만)

### Step 1: Data Preparation ✅
- [x] Expert trajectories 로드 (완료)
- [ ] Imitation library format 변환

### Step 2: Reward Network 구현
- [ ] `BasicRewardNet` 인스턴스 생성
- [ ] Forward pass 테스트

### Step 3: Generator 준비
- [ ] Option A: BC로 DepthLimitedPolicy(h=4) distill
- [ ] Option B: 단순 MlpPolicy로 시작

### Step 4: AIRL Training
- [ ] AIRL trainer 생성
- [ ] Training loop 실행
- [ ] Metrics 모니터링 (discriminator accuracy)

### Step 5: Evaluation
- [ ] Discrimination accuracy 측정
- [ ] Generated trajectories 시각화
- [ ] Expert vs Generated 비교

## Phase 2b: Multi-Depth Training

- [ ] h ∈ {1, 2, 4, 8} 각각 학습
- [ ] 결과 비교
- [ ] Best h 선택

---

# Part 5: 핵심 개념 요약

## Gymnasium
- **reset()**: 새 에피소드 시작 → (obs, info)
- **step(action)**: Action 실행 → (obs, reward, terminated, truncated, info)
- **observation_space**: State 공간 정의
- **action_space**: Action 공간 정의

## AIRL
- **Discriminator (Reward Net)**: Expert vs Generated 구분
- **Generator (Policy)**: Expert 모방 학습
- **Adversarial Training**: GAN처럼 경쟁
- **Result**: Learned reward function

## 우리 프로젝트
- **Depth in Policy**: h는 generator에만
- **Reward Agnostic**: Discriminator는 h 모름
- **Compare h values**: 어떤 h가 expert를 가장 잘 설명?

---

# Part 6: 다음 단계

## 즉시 구현할 것

1. **Trajectory 변환 함수** (`fourinarow_airl/airl_utils.py`)
   ```python
   def convert_to_imitation_format(game_trajectories):
       """GameTrajectory → imitation.Trajectory"""
       pass
   ```

2. **Reward Network 테스트** (`test_reward_net.py`)
   ```python
   reward_net = BasicRewardNet(...)
   reward = reward_net(state, action, next_state)
   print(f"Reward: {reward}")
   ```

3. **Baseline AIRL Script** (`train_baseline_airl.py`)
   ```python
   # h=4로 단일 실험
   trainer = AIRL(...)
   trainer.train(total_timesteps=10000)  # 짧게 테스트
   ```

준비되셨으면 하나씩 구현해보겠습니다!
