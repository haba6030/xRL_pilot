# Depth Integration 전략: 상세 구현 가이드

## 목표

**Planning depth h를 POLICY에만 통합하면서, AIRL discriminator(reward network)는 완전히 depth-agnostic하게 유지**

이 문서는 GYMNASIUM_AND_AIRL_GUIDE.md 3.3절의 상세 구현 가이드입니다.

---

# 전체 흐름도

```
For each h ∈ {1, 2, 4, 8}:

[Step A] DepthLimitedPolicy(h) 사용
    ↓ generates trajectories
[Step B] BC (Behavior Cloning): Neural net learns to mimic
    ↓ produces
[Step C] Neural policy that behaves like depth-h
    ↓ wrapped by
[Step D] PPO (for AIRL fine-tuning)
    ↓ combined with
[Step E] Depth-AGNOSTIC reward network
    ↓
[Step F] AIRL training
    ↓
[Step G] Evaluate & Compare h values
```

---

# Step A: Generate h-specific Training Data

## 목적

DepthLimitedPolicy(h)를 사용해 trajectories 생성 → BC 학습용 데이터

## 코드

```python
from fourinarow_airl import FourInARowEnv
from fourinarow_airl.depth_limited_policy import DepthLimitedPolicy
from fourinarow_airl.bfs_wrapper import load_all_participant_parameters
import numpy as np

def generate_depth_limited_trajectories(
    h: int,
    num_episodes: int = 100,
    seed: int = 42
):
    """
    Generate trajectories using DepthLimitedPolicy(h)

    Args:
        h: Planning depth
        num_episodes: Number of episodes to generate
        seed: Random seed

    Returns:
        trajectories: List of (observations, actions) tuples
    """
    # Environment
    env = FourInARowEnv()

    # Load expert parameters for heuristic weights
    params_dict = load_all_participant_parameters(
        'opendata/model_fits_main_model.csv'
    )
    expert_params = params_dict[1]  # Use participant 1

    # Create depth-limited policy
    # ═══════════════════════════════════════════════════════
    # CRITICAL: h는 여기서만 사용됨 (policy internal)
    # ═══════════════════════════════════════════════════════
    policy = DepthLimitedPolicy(
        h=h,                                    # ← h is HERE
        params=expert_params,
        beta=1.0,
        lapse_rate=expert_params.lapse_rate
    )

    trajectories = []
    rng = np.random.default_rng(seed + h)  # h-specific seed

    print(f"Generating {num_episodes} episodes with h={h}...")

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + h + episode * 1000)

        episode_obs = [obs.copy()]  # (T+1,)
        episode_acts = []           # (T,)

        done = False
        step_count = 0
        max_steps = 36  # Board size

        while not done and step_count < max_steps:
            # ═══════════════════════════════════════════════════════
            # VALIDATION CHECKPOINT 1:
            # obs는 89-dim (board + features), NO h information
            # ═══════════════════════════════════════════════════════
            assert obs.shape == (89,), f"Observation should be 89-dim, got {obs.shape}"

            # Select action using h-step planning
            action, planning_result = policy.select_action(env, rng)

            # ═══════════════════════════════════════════════════════
            # VALIDATION CHECKPOINT 2:
            # action은 0-35 범위
            # ═══════════════════════════════════════════════════════
            assert 0 <= action <= 35, f"Action out of range: {action}"

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)

            episode_obs.append(obs.copy())
            episode_acts.append(action)

            done = terminated or truncated
            step_count += 1

        # Store trajectory
        trajectory = {
            'observations': np.array(episode_obs, dtype=np.float32),  # (T+1, 89)
            'actions': np.array(episode_acts, dtype=np.int64),        # (T,)
            'length': len(episode_acts),
            'h': h  # ← Metadata only (NOT used in training!)
        }

        trajectories.append(trajectory)

    avg_length = np.mean([t['length'] for t in trajectories])
    print(f"Generated {len(trajectories)} trajectories")
    print(f"Average length: {avg_length:.1f}")
    print(f"Nodes expanded (approx): {int(avg_length * planning_result.nodes_expanded)}")

    return trajectories
```

## 🚨 위험 요소

### 위험 A1: 'h' metadata가 training에 사용됨
**대응**:
```python
# 'h'는 debugging/logging용으로만 저장
# BC training 시 절대 사용하지 않음

# ❌ WRONG
obs_with_h = np.concatenate([obs, [trajectory['h']]])

# ✅ CORRECT
obs = trajectory['observations']  # h 정보 제외
```

### 위험 A2: 같은 seed로 여러 h 생성
**대응**:
```python
# 각 h마다 다른 seed 사용
rng = np.random.default_rng(seed + h)  # h-dependent seed
```

---

# Step B: Behavior Cloning (BC)

## 목적

Neural network가 DepthLimitedPolicy(h)의 behavior를 모방하도록 학습

## 핵심 원칙

> **BC는 (state → action) mapping만 학습. h는 training에 사용되지 않음.**

## 코드

```python
from imitation.algorithms import bc
from imitation.data import types as il_types
import torch
import torch.nn as nn

def train_bc_policy(
    trajectories: List[dict],
    env,
    h: int,  # For logging only!
    n_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 3e-4
):
    """
    Train BC policy to mimic DepthLimitedPolicy(h)

    Args:
        trajectories: Generated from DepthLimitedPolicy(h)
        env: FourInARowEnv
        h: Planning depth (for logging/saving ONLY, NOT training!)
        n_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        bc_trainer: Trained BC object
    """
    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 3:
    # Convert to imitation format WITHOUT using h
    # ═══════════════════════════════════════════════════════
    imitation_trajectories = []

    for traj in trajectories:
        # ═══════════════════════════════════════════════════════
        # CRITICAL: Only use observations and actions
        # DO NOT use traj['h'] anywhere!
        # ═══════════════════════════════════════════════════════
        obs = traj['observations']   # (T+1, 89)
        acts = traj['actions']       # (T,)

        # Verify dimensions
        assert obs.shape[1] == 89, f"Expected 89-dim obs, got {obs.shape[1]}"
        assert acts.min() >= 0 and acts.max() <= 35, f"Actions out of range"

        imitation_traj = il_types.Trajectory(
            obs=obs,
            acts=acts,
            infos=None,
            terminal=True
        )
        imitation_trajectories.append(imitation_traj)

    print(f"\n[BC Training for h={h}]")
    print(f"Trajectories: {len(imitation_trajectories)}")
    print(f"Total transitions: {sum(len(t.acts) for t in imitation_trajectories)}")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 4:
    # BC policy architecture has NO h parameter
    # ═══════════════════════════════════════════════════════

    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,  # Box(89,)
        action_space=env.action_space,            # Discrete(36)
        demonstrations=imitation_trajectories,
        batch_size=batch_size,
        # Policy network configuration
        policy_kwargs=dict(
            net_arch=[64, 64],  # MLP architecture
            activation_fn=nn.Tanh,
        ),
    )

    # ═══════════════════════════════════════════════════════
    # VALIDATION: Verify policy has no h-related attributes
    # ═══════════════════════════════════════════════════════
    policy = bc_trainer.policy
    suspicious_attrs = [attr for attr in dir(policy) if 'depth' in attr.lower() or attr == 'h']

    if len(suspicious_attrs) > 0:
        print(f"⚠️  WARNING: Found suspicious attributes: {suspicious_attrs}")
    else:
        print(f"✓ Policy has no depth-related attributes")

    # Train
    print(f"Training for {n_epochs} epochs...")
    bc_trainer.train(n_epochs=n_epochs)

    print(f"✓ BC training complete")

    return bc_trainer
```

## 🚨 위험 요소

### 위험 B1: Policy network에 h를 input으로 추가
**잘못된 예**:
```python
# ❌ ABSOLUTELY WRONG
class PolicyWithDepth(nn.Module):
    def forward(self, obs, h):  # ← h in forward!
        x = torch.cat([obs, torch.tensor([h])])
        return self.mlp(x)
```

**올바른 예**:
```python
# ✅ CORRECT
# BC는 기본 policy를 사용 (observation → action)
# h는 전혀 사용되지 않음
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=trajectories
)
# NO h parameter anywhere!
```

### 위험 B2: Demonstrations에 h label 추가
**대응**:
```python
# Trajectory는 (obs, acts)만 포함
# h는 metadata로만 사용 (training에 사용 안 됨)

for traj in trajectories:
    # ✓ Use these
    obs = traj['observations']
    acts = traj['actions']

    # ✗ DO NOT use
    # h = traj['h']  # Ignore this!
```

---

# Step C: Wrap BC Policy with PPO

## 목적

BC로 학습한 policy를 PPO로 감싸서 AIRL에서 fine-tuning 가능하게 만듦

## 코드

```python
from stable_baselines3 import PPO

def create_ppo_from_bc(
    bc_trainer,
    env,
    h: int,  # For logging/saving only!
    learning_rate: float = 3e-4,
):
    """
    Wrap BC policy with PPO for AIRL training

    Args:
        bc_trainer: Trained BC object
        env: FourInARowEnv
        h: Planning depth (metadata only)
        learning_rate: PPO learning rate

    Returns:
        ppo_algo: PPO algorithm with BC-initialized policy
    """
    print(f"\n[Creating PPO from BC policy (h={h})]")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 5:
    # PPO uses BC policy, which is depth-agnostic
    # ═══════════════════════════════════════════════════════

    # Extract BC policy
    bc_policy = bc_trainer.policy

    # Verify policy input dimension
    print(f"BC Policy observation space: {env.observation_space}")
    print(f"BC Policy action space: {env.action_space}")

    # ═══════════════════════════════════════════════════════
    # CRITICAL: PPO receives depth-agnostic policy
    # h는 여기서도 사용되지 않음
    # ═══════════════════════════════════════════════════════

    # Create PPO with BC policy
    ppo_algo = PPO(
        policy=bc_policy,           # ← BC-learned policy (no h!)
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
    )

    print(f"✓ PPO created with BC-initialized policy")
    print(f"✓ Policy is depth-agnostic (only sees 89-dim observations)")

    return ppo_algo
```

## 🚨 위험 요소

### 위험 C1: PPO observation augmentation
**문제**: PPO가 자동으로 observation을 augment할 가능성
**검증**:
```python
# PPO policy의 observation dimension 확인
assert ppo_algo.policy.observation_space.shape == (89,)
```

---

# Step D: Create Depth-AGNOSTIC Reward Network

## 목적

모든 h에 대해 **동일한 architecture**의 reward network 생성

## 코드

```python
from imitation.rewards.reward_nets import BasicRewardNet

def create_reward_network(env):
    """
    Create depth-agnostic reward network

    CRITICAL: This function has NO h parameter!
    Same architecture for ALL h values.

    Args:
        env: FourInARowEnv

    Returns:
        reward_net: BasicRewardNet (depth-agnostic)
    """
    print(f"\n[Creating Depth-AGNOSTIC Reward Network]")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 6:
    # NO h parameter in this function
    # NO h parameter in reward network
    # ═══════════════════════════════════════════════════════

    reward_net = BasicRewardNet(
        observation_space=env.observation_space,  # Box(89,)
        action_space=env.action_space,            # Discrete(36)
        hid_sizes=[64, 64],                       # MLP hidden layers
        activation=nn.Tanh,
    )

    print(f"Reward Network Architecture:")
    print(f"  Input: (state, action, next_state)")
    print(f"  State dim: 89 (board + features, NO depth)")
    print(f"  Action dim: 36 (one-hot encoded)")
    print(f"  Hidden: [64, 64]")
    print(f"  Output: scalar reward")
    print(f"✓ NO h parameter")
    print(f"✓ Same architecture for ALL h values")

    # ═══════════════════════════════════════════════════════
    # VALIDATION: Check for depth-related attributes
    # ═══════════════════════════════════════════════════════
    suspicious_attrs = [
        attr for attr in dir(reward_net)
        if 'depth' in attr.lower() or attr == 'h'
    ]

    if len(suspicious_attrs) > 0:
        raise ValueError(
            f"Reward network has depth-related attributes: {suspicious_attrs}\n"
            f"This violates PLANNING_DEPTH_PRINCIPLES.md!"
        )

    print(f"✓ Validation passed: No depth-related attributes")

    return reward_net
```

## 🚨 위험 요소

### 위험 D1: 실수로 h를 parameter로 전달
**절대 금지**:
```python
# ❌ ABSOLUTELY FORBIDDEN
def create_reward_network(env, h):  # ← NO h parameter!
    reward_net = BasicRewardNet(..., depth=h)
```

**올바름**:
```python
# ✅ CORRECT
def create_reward_network(env):  # No h!
    reward_net = BasicRewardNet(...)  # No h!
```

### 위험 D2: 여러 h에 대해 reward network 재사용
**문제**: 같은 reward_net instance를 여러 h training에 사용
**올바른 방법**:
```python
for h in [1, 2, 4, 8]:
    # Fresh reward network for each h
    reward_net = create_reward_network(env)  # New instance!
    # Train with h-specific generator
```

---

# Step E: AIRL Training

## 목적

h-specific generator + depth-agnostic discriminator로 AIRL 학습

## 코드

```python
from imitation.algorithms.adversarial import airl

def train_airl_for_depth(
    h: int,
    expert_trajectories: List,
    env,
    total_timesteps: int = 100000,
    n_disc_updates_per_round: int = 4,
):
    """
    Train AIRL for specific planning depth h

    Args:
        h: Planning depth (only affects generator!)
        expert_trajectories: Expert demonstrations (depth-agnostic)
        env: Environment
        total_timesteps: Training timesteps
        n_disc_updates_per_round: Discriminator updates

    Returns:
        trainer: Trained AIRL object
        results: Training metrics
    """
    print("=" * 80)
    print(f"AIRL Training for h={h}")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # Step E1: Generate h-specific training data
    # ═══════════════════════════════════════════════════════
    print(f"\n[Step 1] Generate trajectories with h={h}")
    depth_trajectories = generate_depth_limited_trajectories(
        h=h,
        num_episodes=100
    )

    # ═══════════════════════════════════════════════════════
    # Step E2: BC training (h → neural policy)
    # ═══════════════════════════════════════════════════════
    print(f"\n[Step 2] BC training (mimic h={h} policy)")
    bc_trainer = train_bc_policy(
        trajectories=depth_trajectories,
        env=env,
        h=h,  # Metadata only!
        n_epochs=50
    )

    # ═══════════════════════════════════════════════════════
    # Step E3: Wrap with PPO
    # ═══════════════════════════════════════════════════════
    print(f"\n[Step 3] Create PPO generator")
    gen_algo = create_ppo_from_bc(
        bc_trainer=bc_trainer,
        env=env,
        h=h  # Metadata only!
    )

    # ═══════════════════════════════════════════════════════
    # Step E4: Create depth-AGNOSTIC reward network
    # ═══════════════════════════════════════════════════════
    print(f"\n[Step 4] Create depth-AGNOSTIC reward network")
    reward_net = create_reward_network(env)

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 7:
    # Final check before AIRL training
    # ═══════════════════════════════════════════════════════
    print(f"\n[Validation] Pre-training checks:")
    print(f"✓ Generator: Learned from h={h} policy")
    print(f"✓ Discriminator: NO h parameter")
    print(f"✓ Expert data: NO h labels")
    print(f"✓ Observations: 89-dim (board + features, no depth)")

    # Verify expert data
    for i, traj in enumerate(expert_trajectories[:3]):
        assert traj.obs.shape[1] == 89, \
            f"Expert traj {i}: Expected 89-dim, got {traj.obs.shape[1]}"
    print(f"✓ Expert trajectories validated")

    # ═══════════════════════════════════════════════════════
    # Step E5: Create AIRL trainer
    # ═══════════════════════════════════════════════════════
    print(f"\n[Step 5] Create AIRL trainer")

    trainer = airl.AIRL(
        demonstrations=expert_trajectories,      # Expert data
        demo_batch_size=256,

        venv=env,                                # Environment
        gen_algo=gen_algo,                       # h-dependent generator
        reward_net=reward_net,                   # h-AGNOSTIC discriminator!

        n_disc_updates_per_round=n_disc_updates_per_round,
        demo_minibatch_size=64,
    )

    print(f"✓ AIRL trainer created")

    # ═══════════════════════════════════════════════════════
    # Step E6: Train
    # ═══════════════════════════════════════════════════════
    print(f"\n[Step 6] AIRL training ({total_timesteps} timesteps)")

    trainer.train(total_timesteps=total_timesteps)

    print(f"✓ AIRL training complete")

    # ═══════════════════════════════════════════════════════
    # Step E7: Extract results
    # ═══════════════════════════════════════════════════════
    results = {
        'h': h,
        'reward_net': reward_net,
        'generator': gen_algo,
        'trainer': trainer,
    }

    return trainer, results
```

## 🚨 위험 요소

### 위험 E1: Expert trajectories에 h label 추가
**검증**:
```python
# Expert data는 h 정보가 전혀 없어야 함
for traj in expert_trajectories:
    assert not hasattr(traj, 'depth')
    assert not hasattr(traj, 'h')
    assert traj.obs.shape[1] == 89  # No augmented features
```

### 위험 E2: Discriminator가 간접적으로 h 학습
**질문**: "Discriminator가 trajectory pattern으로 h를 추론하면?"

**답변**: **이것은 acceptable함!**

```python
# Pattern recognition is OK:
# - h=1 generator → short trajectories, shallow planning
# - h=8 generator → longer trajectories, deeper planning
# Discriminator가 이 pattern을 학습하는 것은 자연스러움

# What's NOT OK:
# - Discriminator에 명시적 h input
# - Observation에 h 정보 포함
```

**이유**: Discriminator의 목표는 "expert behavior pattern을 학습"하는 것. 만약 expert가 실제로 h=8로 planning한다면, discriminator는 "h=8 pattern"을 학습해야 함.

---

# Step F: Multi-Depth Comparison

## 목적

여러 h 값에 대해 학습 후 비교

## 코드

```python
def train_all_depths(
    depths: List[int],
    expert_trajectories: List,
    env,
    total_timesteps: int = 100000,
):
    """
    Train AIRL for multiple depths and compare

    Args:
        depths: List of planning depths (e.g., [1, 2, 4, 8])
        expert_trajectories: Expert demonstrations
        env: Environment
        total_timesteps: Timesteps per depth

    Returns:
        all_results: Dict mapping h → results
    """
    all_results = {}

    for h in depths:
        print(f"\n{'=' * 80}")
        print(f"Training depth h={h}")
        print(f"{'=' * 80}")

        trainer, results = train_airl_for_depth(
            h=h,
            expert_trajectories=expert_trajectories,
            env=env,
            total_timesteps=total_timesteps
        )

        all_results[h] = results

        # Save models
        import torch
        save_dir = f'models/h{h}'
        os.makedirs(save_dir, exist_ok=True)

        # ═══════════════════════════════════════════════════════
        # NAMING CONVENTION (CRITICAL):
        # NOT "h4_reward.pt" (implies h-specific reward)
        # BUT "reward_trained_with_h4_generator.pt"
        # ═══════════════════════════════════════════════════════

        torch.save(
            results['reward_net'].state_dict(),
            f'{save_dir}/reward_trained_with_h{h}_generator.pt'
        )

        torch.save(
            results['generator'].policy.state_dict(),
            f'{save_dir}/generator_h{h}.pt'
        )

        print(f"✓ Saved models for h={h}")

    return all_results
```

---

# Step G: Evaluation & Comparison

## 평가 지표

### 1. Discrimination Accuracy

```python
def evaluate_discrimination(trainer, expert_trajectories):
    """
    Measure how well discriminator distinguishes expert vs generated

    Target: ~0.5 (means generator matches expert well)
    """
    # Expert accuracy
    expert_logits = []
    for traj in expert_trajectories:
        for t in range(len(traj.acts)):
            logit = trainer.reward_net(
                traj.obs[t],
                traj.acts[t],
                traj.obs[t+1]
            )
            expert_logits.append(logit)

    expert_acc = (torch.sigmoid(expert_logits) > 0.5).float().mean()

    # Generated accuracy
    gen_trajectories = generate_from_policy(trainer.gen_algo, num=100)
    gen_logits = []
    for traj in gen_trajectories:
        for t in range(len(traj.acts)):
            logit = trainer.reward_net(...)
            gen_logits.append(logit)

    gen_acc = (torch.sigmoid(gen_logits) < 0.5).float().mean()

    # Overall accuracy
    disc_acc = (expert_acc + gen_acc) / 2

    return {
        'disc_acc': disc_acc,
        'expert_acc': expert_acc,
        'gen_acc': gen_acc
    }
```

**해석**:
- `disc_acc ~ 0.5`: Generator가 expert를 잘 모방 (good!)
- `disc_acc >> 0.5`: Generator가 expert와 다름 (need more training)

### 2. Imitation Quality

```python
def evaluate_imitation_quality(trainer, expert_trajectories):
    """
    Compare generated vs expert trajectories
    """
    gen_trajs = generate_from_policy(trainer.gen_algo, num=100)

    # Trajectory length
    expert_lengths = [len(t.acts) for t in expert_trajectories]
    gen_lengths = [len(t.acts) for t in gen_trajs]

    # Action distribution
    expert_actions = np.concatenate([t.acts for t in expert_trajectories])
    gen_actions = np.concatenate([t.acts for t in gen_trajs])

    # KL divergence
    expert_dist = np.bincount(expert_actions, minlength=36) / len(expert_actions)
    gen_dist = np.bincount(gen_actions, minlength=36) / len(gen_actions)

    kl_div = np.sum(expert_dist * np.log((expert_dist + 1e-10) / (gen_dist + 1e-10)))

    return {
        'length_diff': abs(np.mean(expert_lengths) - np.mean(gen_lengths)),
        'kl_divergence': kl_div
    }
```

### 3. Best h Selection

```python
def select_best_depth(all_results):
    """
    Select which h best explains expert behavior
    """
    metrics = {}

    for h, results in all_results.items():
        disc_metrics = evaluate_discrimination(results['trainer'], expert_trajs)
        imit_metrics = evaluate_imitation_quality(results['trainer'], expert_trajs)

        metrics[h] = {
            'disc_acc': disc_metrics['disc_acc'],
            'kl_div': imit_metrics['kl_divergence'],
            # Lower is better for both (disc_acc closer to 0.5, kl_div closer to 0)
            'score': abs(disc_metrics['disc_acc'] - 0.5) + imit_metrics['kl_divergence']
        }

    # Best h = lowest score
    best_h = min(metrics.keys(), key=lambda h: metrics[h]['score'])

    print(f"\nBest depth: h={best_h}")
    print(f"Metrics: {metrics[best_h]}")

    return best_h, metrics
```

---

# 최종 Validation 체크리스트

## 전체 Pipeline 검증

- [ ] **Step A**: DepthLimitedPolicy(h) generates 89-dim observations
- [ ] **Step B**: BC trains on (obs, action) only, NO h
- [ ] **Step C**: PPO policy has 89-dim observation space
- [ ] **Step D**: Reward network has NO h parameter
- [ ] **Step E**: AIRL expert data has NO h labels
- [ ] **Step F**: Each h uses fresh reward network instance
- [ ] **Step G**: Terminology: "reward trained with h=X generator"

## 코드 Audit

```python
# Search for forbidden patterns:
grep -r "depth.*reward" *.py  # Should find nothing
grep -r "reward.*depth" *.py  # Should find nothing
grep -r "h.*discriminator" *.py  # Should find nothing

# Allowed patterns:
grep -r "h.*generator" *.py  # OK
grep -r "h.*policy" *.py  # OK
```

---

# 요약: 어디에 h가 있고 없는가

## h가 있는 곳 ✅

1. `DepthLimitedPolicy(h=h)` - Policy definition
2. `generate_depth_limited_trajectories(h=h)` - Data generation
3. File names: `generator_h4.pt`, `reward_trained_with_h4_generator.pt`
4. Metadata/logging: `{'h': h, ...}` (training에 사용 안 됨)

## h가 절대 없어야 하는 곳 ❌

1. `BasicRewardNet(...)` - NO h parameter
2. `reward_net.forward(state, action, next_state)` - NO h input
3. Observations: `(batch, 89)` - NO h in state
4. Expert trajectories: `Trajectory(obs, acts)` - NO h label
5. BC training: `bc.BC(demonstrations=...)` - NO h in data

---

**문서**: DEPTH_INTEGRATION_DETAILED.md
**상태**: 완전 구체화 완료
**다음**: Step-by-step 구현 시작
**환경**: pedestrian_analysis (Python 3.9.7, imitation installed)
