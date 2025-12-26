# Implementation Technical Notes

**Last Updated**: 2025-12-25
**Purpose**: Document actual implementation details, technical discoveries, and solutions

---

## 1. Environment Setup

### Conda Environment

```bash
conda activate pedestrian_analysis
# Python 3.9.7
# imitation==1.0.1
# stable-baselines3
# torch

# OpenMP conflict workaround
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## 2. BasicRewardNet Usage (Critical!)

### Two-Stage Processing Required

```python
from imitation.rewards.reward_nets import BasicRewardNet

# 1. Create reward network
reward_net = BasicRewardNet(
    observation_space=env.observation_space,  # Box(89,)
    action_space=env.action_space,            # Discrete(36)
    hid_sizes=[64, 64]
)

# 2. Forward pass - MUST call preprocess() first!
state_th, action_th, next_state_th, done_th = reward_net.preprocess(
    obs_tensor,      # (batch, 89) FloatTensor
    action_tensor,   # (batch,) LongTensor - indices, NOT one-hot!
    next_obs_tensor, # (batch, 89) FloatTensor
    done_tensor      # (batch,) BoolTensor
)
reward = reward_net(state_th, action_th, next_state_th, done_th)
# Output: (batch,) FloatTensor
```

### Tensor Dimensions

| Input | Shape | Type | Notes |
|-------|-------|------|-------|
| `obs` | `(batch, 89)` | `FloatTensor` | Board (72) + features (17) |
| `action` | `(batch,)` | `LongTensor` | **Action indices**, NOT one-hot |
| `next_obs` | `(batch, 89)` | `FloatTensor` | Next board + features |
| `done` | `(batch,)` | `BoolTensor` | Terminal flags |

**After preprocessing**:
- `action_th`: `(batch, 36)` FloatTensor (one-hot encoded)
- Total MLP input: 125-dim = 89 (obs) + 36 (one-hot action)

---

## 3. imitation 1.0.1 API

### BC (Behavior Cloning)

```python
from imitation.algorithms import bc

# Requires `rng` parameter (no `policy_kwargs`)
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=trajectories,
    rng=np.random.default_rng(42),  # Required!
    batch_size=64
)
# Default architecture: 32x32 MLP
```

### AIRL

```python
from imitation.algorithms.adversarial import airl

trainer = airl.AIRL(
    demonstrations=expert_trajectories,
    demo_batch_size=demo_batch_size,
    venv=venv,
    gen_algo=gen_algo,
    reward_net=reward_net,
    n_disc_updates_per_round=n_disc_updates_per_round,
    gen_train_timesteps=gen_train_timesteps,
    allow_variable_horizon=True,  # Required for 4-in-a-row!
)
```

**Critical**: 4-in-a-row games have variable length → must set `allow_variable_horizon=True`

---

## 4. Architecture Matching (BC → PPO)

### Problem: Size Mismatch

BC (imitation 1.0.1) uses 32x32 MLP by default.
PPO (stable-baselines3) uses 64x64 MLP by default.

### Solution: Auto-detect and Match

```python
# Auto-detect BC policy architecture
first_layer_shape = None
for name, param in bc_policy.named_parameters():
    if 'mlp_extractor.policy_net.0.weight' in name:
        first_layer_shape = param.shape
        break

net_arch_size = first_layer_shape[0]  # e.g., 32

# Configure PPO to match
policy_kwargs = dict(
    net_arch=dict(
        pi=[net_arch_size, net_arch_size],  # [32, 32]
        vf=[net_arch_size, net_arch_size],  # [32, 32]
    )
)

ppo = PPO(
    policy="MlpPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    ...
)
ppo.policy.load_state_dict(bc_policy.state_dict())  # Now works!
```

---

## 5. Data Formats and Dimensions

### Observation Space

```python
observation_space = gym.spaces.Box(
    low=0.0,
    high=1.0,
    shape=(89,),  # 72 (board) + 17 (features)
    dtype=np.float32
)
# [0:36]: Black pieces (6×6 board, flattened)
# [36:72]: White pieces (6×6 board, flattened)
# [72:89]: Van Opheusden features (17-dim)
```

### Action Space

```python
action_space = gym.spaces.Discrete(36)  # Board positions (row-major)
```

### Trajectory Format

```python
from imitation.data.types import Trajectory

# GameTrajectory (custom) → Trajectory (imitation)
imitation_traj = Trajectory(
    obs=observations,    # (T+1, 89) float32
    acts=actions,        # (T,) int64 - action indices
    infos=None,
    terminal=True
)
```

### Network Architectures

| Network | Architecture | Input Dim | Output Dim |
|---------|--------------|-----------|------------|
| BC Policy | [32, 32] MLP | 89 | 36 (logits) |
| PPO Generator | [32, 32] MLP | 89 | 36 (logits) |
| Reward Network | [64, 64] MLP | 125 | 1 (reward) |

---

## 6. AIRL Training Metrics

### Discriminator Metrics

**Well-trained (Generator fools Discriminator)**:
```python
disc_acc ≈ 0.5           # Overall accuracy
disc_acc_expert ≈ 0.5    # Expert classified as expert ~50%
disc_acc_gen ≈ 0.5       # Generated classified as expert ~50%
```

**Undertrained (Discriminator too strong)**:
```python
disc_acc = 0.5           # Overall looks OK, but...
disc_acc_expert = 1.0    # Perfectly identifies expert (too strong!)
disc_acc_gen = 0.0       # Perfectly rejects generated (too strong!)
```

**Interpretation**:
- `disc_acc_expert` = 1.0: Discriminator perfectly distinguishes expert
- `disc_acc_gen` = 0.0: Generator hasn't learned to fool discriminator
- **Solution**: Train longer (increase `total_timesteps`)

### Training Progression

Early training:
```
disc_acc_expert: 1.0 → 0.8 → 0.6 → 0.5
disc_acc_gen: 0.0 → 0.2 → 0.4 → 0.5
```

Well-trained:
```
disc_acc_expert: 0.45-0.55 (balanced)
disc_acc_gen: 0.45-0.55 (balanced)
disc_loss: ~0.69 (log(2), random guessing)
```

---

## 7. Core Principle Verification

### Planning depth h exists ONLY in POLICY

| Component | h present? | Validation |
|-----------|------------|------------|
| DepthLimitedPolicy | ✅ YES | `self.h = h` |
| Observations | ❌ NO | Shape: (batch, 89) |
| BC Training | ❌ NO | Only (obs, acts) used |
| BC Policy | ❌ NO | No depth attributes |
| PPO Generator | ❌ NO | 89-dim observation |
| Reward Network | ❌ NO | Function signature checked |
| Expert Trajectories | ❌ NO | Trajectory(obs, acts) |

**Result**: ✅ Principle fully adhered to

---

## 8. Implementation Status

### Completed (Steps A-E)

- ✅ **Step A**: Generate h-specific training data
  - File: `generate_training_data.py`
  - Creates trajectories using DepthLimitedPolicy(h)

- ✅ **Step B**: Behavior Cloning (BC)
  - File: `train_bc.py`
  - Neural network mimics h-specific behavior

- ✅ **Step C**: Wrap BC with PPO
  - File: `create_ppo_generator.py`
  - BC-initialized PPO generator

- ✅ **Step D**: Depth-agnostic reward network
  - File: `create_reward_net.py`
  - NO h parameter anywhere

- ✅ **Step E**: AIRL Training
  - File: `train_airl.py`
  - h-specific generator + depth-agnostic discriminator

### Next Steps

- 🔄 **Step F**: Multi-depth comparison
- 🔄 **Step G**: Evaluation and analysis

**Validation**: 6/8 checkpoints passed

---

## 9. Common Issues and Solutions

### Issue 1: OpenMP Library Conflict

**Error**:
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

**Solution**:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Issue 2: Variable Horizon Error

**Error**:
```
ValueError: Episodes of different length detected. Variable horizon environments are discouraged.
```

**Solution**:
```python
trainer = airl.AIRL(
    ...,
    allow_variable_horizon=True  # Add this!
)
```

### Issue 3: Action Tensor Shape

**Error**:
```
IndexError: Dimension out of range
```

**Solution**:
```python
# ❌ WRONG
action = torch.LongTensor([[5], [10]])  # (batch, 1)

# ✅ CORRECT
action = torch.LongTensor([5, 10])  # (batch,)
```

### Issue 4: Trajectory Format Mismatch

**Error**:
```
AttributeError: 'dict' object has no attribute 'observations'
```

**Solution**:
```python
# Handle both dict and object formats
if isinstance(traj, dict):
    obs = traj['observations']
    acts = traj['actions']
else:
    obs = traj.observations
    acts = traj.actions
```

---

## 10. Training Recommendations

### Minimal Test (Quick Verification)

```python
total_timesteps = 1024       # Very quick
demo_batch_size = 32
n_disc_updates_per_round = 2
gen_train_timesteps = 256
```

**Result**: Training works, but discriminator too strong

### Short Training (Development)

```python
total_timesteps = 10000      # ~2-3 minutes
demo_batch_size = 64
n_disc_updates_per_round = 4
gen_train_timesteps = 512
```

**Expected**: Better balance, disc_acc metrics approaching 0.5

### Full Training (Production)

```python
total_timesteps = 50000-100000  # ~15-30 minutes
demo_batch_size = 128
n_disc_updates_per_round = 4
gen_train_timesteps = 1024
```

**Expected**: Well-balanced metrics, good imitation

---

## 11. File Locations

```
fourinarow_airl/
├── generate_training_data.py   # Step A
├── train_bc.py                  # Step B
├── create_ppo_generator.py      # Step C
├── create_reward_net.py         # Step D
├── train_airl.py                # Step E
├── airl_utils.py                # Utilities
└── env.py                       # Environment

models/
├── bc_policies/                 # BC outputs
├── ppo_generators/              # PPO outputs
└── airl_results/                # AIRL outputs
```

---

**Last Updated**: 2025-12-25
**Implementation**: Steps A-E complete, ready for multi-depth training
