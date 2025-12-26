# Planning-Aware AIRL for 4-in-a-row

Implementation of AIRL (Adversarial Inverse Reinforcement Learning) for the 4-in-a-row game, with **planning depth as an explicit policy-level constraint**.

Based on:
- van Opheusden et al. (2023): Expertise modeling via planning depth
- AIRL framework (Fu et al., 2018): Adversarial reward learning
- Yao et al. (2024): Planning horizon as latent confounder in IRL
- Pedestrian project AIRL implementation

## Overview

This package implements AIRL for 4-in-a-row with a critical design principle:

**Planning depth h is a POLICY constraint, NOT a reward parameter.**

This design enables us to answer: *"Which planning depth h best explains expert behavior?"* by comparing AIRL-trained models across different depth assumptions.

### Key Principle

```python
# ✅ CORRECT: Depth in policy (generator)
generator_h = DepthLimitedPolicy(h=h)  # ← h lives here

# ✅ CORRECT: Depth-agnostic reward network
reward_net = BasicRewardNet(...)  # ← NO h parameter (same for all h)

# ❌ WRONG: h-conditioned reward
reward_net = RewardNet(h=h)  # ← Violates reward-planning disentanglement
```

See `../PLANNING_DEPTH_PRINCIPLES.md` for full theoretical justification.

### Key Features

✅ **Gymnasium Environment** (`env.py`)
- 6×6 board with standard 4-in-a-row rules
- 89-dimensional state space (72 board + 17 Van Opheusden features)
- Discrete(36) action space
- Perfect state cloning support (38μs per clone)
- Win/draw detection and rendering

✅ **Van Opheusden Feature Extraction** (`features.py`)
- 17 heuristic features matching original C++ implementation:
  - Center control (1 feature)
  - Connected 2-in-a-row (4 orientations)
  - Unconnected 2-in-a-row (4 orientations)
  - 3-in-a-row (4 orientations)
  - 4-in-a-row (4 orientations)

✅ **Depth-Limited Planning Policy** (`depth_limited_policy.py`)
- Implements planning depth h as policy constraint
- h-step lookahead using environment cloning
- Heuristic state evaluation (baseline, no C++ BFS needed)
- Verified: Different h → Different behaviors (see verification results)

✅ **Expert Trajectory Loading** (`data_loader.py`)
- Loads from `opendata/raw_data.csv` (67,331 trials → 5,061 games)
- Reconstructs individual games from trial-level data
- Filters by player (Black/White)
- Converts to imitation library format

✅ **BFS Policy Wrapper** (`bfs_wrapper.py`)
- Loads van Opheusden model parameters (40 participants)
- Provides parameter dataclass for easy access
- Ready for C++ BFS integration (optional enhancement)

## Installation

```bash
# Core dependencies (Phase 1 - complete)
pip install gymnasium numpy pandas scipy matplotlib

# AIRL training dependencies (Phase 2 - upcoming)
pip install imitation stable-baselines3 torch
```

## Quick Start

### Test Environment

```python
from fourinarow_airl import FourInARowEnv

env = FourInARowEnv(render_mode='ansi')
obs, info = env.reset()

print(env.render())
print(f"Observation shape: {obs.shape}")  # (89,)
print(f"Legal actions: {info['legal_actions']}")

# Play a move
action = info['legal_actions'][0]
obs, reward, terminated, truncated, info = env.step(action)
```

### Use Depth-Limited Policy

```python
from fourinarow_airl.depth_limited_policy import DepthLimitedPolicy
from fourinarow_airl import FourInARowEnv

env = FourInARowEnv()
obs, info = env.reset()

# Create policy with planning depth h=4
policy = DepthLimitedPolicy(h=4, beta=1.0, lapse_rate=0.1)

# Plan and select action
action, result = policy.select_action(env)
print(f"Best action: {action}")
print(f"Q-value: {result.action_values[action]:.3f}")
print(f"Nodes expanded: {result.nodes_expanded}")

# Execute action
obs, reward, done, _, info = env.step(action)
```

### Load Expert Trajectories

```python
from fourinarow_airl.data_loader import load_expert_trajectories

# Load Black player trajectories
trajectories = load_expert_trajectories(
    csv_path='opendata/raw_data.csv',
    player_filter=0,  # 0=Black, 1=White, None=both
    max_trajectories=100
)

print(f"Loaded {len(trajectories)} trajectories")
print(f"First trajectory: {len(trajectories[0].actions)} moves")
```

### Load BFS Parameters

```python
from fourinarow_airl.bfs_wrapper import load_all_participant_parameters

params_dict = load_all_participant_parameters(
    'opendata/model_fits_main_model.csv'
)

participant_1_params = params_dict[1]
print(f"Pruning threshold: {participant_1_params.pruning_threshold}")
print(f"Lapse rate: {participant_1_params.lapse_rate}")
```

## Package Structure

```
fourinarow_airl/
├── __init__.py              # Package initialization
├── env.py                   # FourInARowEnv Gymnasium environment
├── features.py              # Van Opheusden 17-feature extraction
├── depth_limited_policy.py  # ✨ Depth-limited planning policy
├── data_loader.py           # Expert trajectory loading from CSV
├── bfs_wrapper.py           # BFS policy wrapper and parameter loading
└── README.md                # This file
```

## State Representation

**Observation space**: Box(89,) with values in [0, 1]

- **Positions 0-35**: Black pieces (1 if Black piece at position, 0 otherwise)
- **Positions 36-71**: White pieces (1 if White piece at position, 0 otherwise)
- **Positions 72-88**: Van Opheusden features (17-dim, normalized):
  - Feature 0: Center control (active player)
  - Features 1-4: Connected 2-in-a-row (H, V, D, AD)
  - Features 5-8: Unconnected 2-in-a-row (H, V, D, AD)
  - Features 9-12: 3-in-a-row (H, V, D, AD)
  - Features 13-16: 4-in-a-row (H, V, D, AD)

## Action Representation

**Action space**: Discrete(36)

Actions correspond to board positions in row-major order:
```
Position indices:
  0  1  2  3  4  5
  6  7  8  9 10 11
 12 13 14 15 16 17
 18 19 20 21 22 23
 24 25 26 27 28 29
 30 31 32 33 34 35
```

## Data Format

### Expert Trajectories

`GameTrajectory` dataclass contains:
- `observations`: (T+1, 89) numpy array (states including final)
- `actions`: (T,) numpy array (action indices)
- `rewards`: (T,) numpy array (0 until final +1/-1)
- `player_id`: 0 (Black) or 1 (White)
- `game_id`: Unique game identifier
- `participant_id`: Participant ID from dataset

### BFS Parameters

`BFSParameters` dataclass contains:
- **Search**: `pruning_threshold`, `stopping_probability`
- **Cognitive**: `lapse_rate`, `feature_drop_rate`, `active_scaling_constant`
- **Features**: `center_weight`, `connected_2_weight`, `unconnected_2_weight`,
  `three_in_a_row_weight`, `four_in_a_row_weight`
- **Metadata**: `participant_id`, `log_likelihood`

## Depth Variable Verification ✅

**Question**: Does planning depth h create meaningful behavioral variation?

**Answer**: YES - Empirically verified.

**Evidence** (from `../DEPTH_VARIABLE_VERIFICATION.md`):

| Depth h | Best Action | Best Q-value | Nodes Expanded |
|---------|-------------|--------------|----------------|
| h=1 | 0 | 0.165 | 33 |
| h=2 | 20 | 0.278 | 1,089 |
| h=4 | 15 | 0.272 | 3,102 |
| h=8 | 3 | **1.307** | 6,732 |

**Key findings**:
- Different h values select **completely different actions**
- Deeper planning finds **higher-value strategies** (8× Q-value increase)
- Computational cost scales predictably with depth
- **Conclusion**: h is a useful variable for modeling expertise

## Testing

Each module has a built-in test function:

```bash
# Test environment
cd fourinarow_airl
python3 env.py

# Test features
python3 features.py

# Test depth-limited policy
python3 depth_limited_policy.py

# Test data loading
python3 data_loader.py

# Test BFS wrapper
python3 bfs_wrapper.py
```

### Integration Tests

```bash
# Phase 1 integration test
cd ..
python3 test_phase1_integration.py

# State clone feasibility
python3 test_state_clone.py
```

## Implementation Status

### ✅ Phase 1 Complete (2025-12-23)

**Core Infrastructure**:
- [x] FourInARowEnv Gymnasium environment
- [x] Van Opheusden 17-feature extraction
- [x] Expert trajectory loading (67K trials → 5K games)
- [x] BFS parameter loading (40 participants)

**Planning Depth Components**:
- [x] State clone infrastructure (verified, 38μs per clone)
- [x] Depth-limited planning policy (`DepthLimitedPolicy`)
- [x] Depth utility verification (different h → different behaviors)

**Documentation**:
- [x] Theoretical principles (`../PLANNING_DEPTH_PRINCIPLES.md`)
- [x] Verification results (`../DEPTH_VARIABLE_VERIFICATION.md`)
- [x] Implementation status (`../IMPLEMENTATION_STATUS.md`)

### 🔄 Phase 2: AIRL Training (Next)

**Priority 1: Baseline AIRL**
- [ ] Depth-agnostic reward network (`BasicRewardNet`)
- [ ] AIRL training script for single h
- [ ] Verify training convergence

**Priority 2: Multi-Depth Comparison**
- [ ] Train AIRL for each h ∈ {1, 2, 4, 8}
- [ ] Compare discrimination accuracy across h
- [ ] Identify best-matching h for expert data

**Priority 3: Evaluation**
- [ ] Discrimination accuracy metrics
- [ ] Imitation score calculation
- [ ] Expertise prediction from learned h

### 📋 Phase 3: Advanced Features (Future)

- [ ] C++ BFS integration (optional, for faster/more accurate planning)
- [ ] Extended depth range (h ∈ {2, 4, 6, 8, 10})
- [ ] Expertise prediction validation (AUC metrics)
- [ ] Clinical trait correlation analysis

## Research Questions

With this implementation, we can answer:

1. **Which planning depth best explains expert behavior?**
   - Method: Compare AIRL across h ∈ {1,2,4,8}
   - Expected: Experts → higher h, Novices → lower h

2. **Is variation due to reward or planning?**
   - Method: Same reward architecture, different h
   - Expected: h explains independent variance

3. **Can learned h predict expertise?**
   - Method: Use h as classifier for novice/expert
   - Expected: AUC > 0.7 (similar to van Opheusden PV depth)

## Theoretical Foundation

### Core Principle

**Planning depth h is a policy-level constraint, NOT a reward parameter.**

This separation is critical for:
- **IRL identifiability** (Yao et al., 2024)
- **Interpretability** (reward ≠ planning)
- **Testable hypotheses** (which h best explains data?)

### Key References

- `../PLANNING_DEPTH_PRINCIPLES.md` - Full theoretical justification
- `../DEPTH_VARIABLE_VERIFICATION.md` - Empirical validation
- `../RESPONSE_TO_FEEDBACK.md` - Design decisions and reviewer responses

### Architectural Locks

Three locks ensure reward-planning disentanglement:

1. **Architectural Separation**: h parameter only in policy, never in reward
2. **Training Isolation**: Distillation at policy level only
3. **Interpretation Discipline**: "Reward trained with h=X generator" (not "h-specific reward")

## AIRL Training Procedure (Planned)

```python
# Pseudocode for Phase 2

for h in [1, 2, 4, 8]:
    # 1. Create depth-limited generator
    generator_h = DepthLimitedPolicy(h=h)  # ← h is HERE
    gen_algo = PPO(generator_h, env, ...)

    # 2. Create depth-AGNOSTIC reward network
    reward_net = BasicRewardNet(...)  # ← SAME architecture for all h

    # 3. Train AIRL
    trainer = AIRL(
        demonstrations=expert_trajectories,
        reward_net=reward_net,  # ← Same network
        gen_algo=gen_algo,      # ← Different h
    )
    trainer.train(total_timesteps=100000)

    # 4. Save result (note terminology)
    torch.save(reward_net.state_dict(),
               f'reward_trained_with_h{h}_generator.pt')

# 5. Compare: Which h achieves best expert imitation?
```

## Data Sources

This implementation uses data from the van Opheusden et al. (2023) 4-in-a-row study:

- **Expert trajectories**: `opendata/raw_data.csv` (67,331 trials, ~5,000 games)
- **Model parameters**: `opendata/model_fits_main_model.csv` (40 participants)
- **Original C++ implementation**: `xRL_pilot/Model code/`

## Citation

If you use this code, please cite:

```bibtex
@article{van2023expertise,
  title={Expertise increases planning depth in human gameplay},
  author={van Opheusden, Bas and Kuperwajs, Gianni and Galbiati, Giacomo and
          Bnaya, Zahy and Li, Yunqi and Ma, Wei Ji},
  journal={Nature},
  year={2023}
}

@article{yao2024planning,
  title={Inverse reinforcement learning with the average reward MDP},
  author={Yao, Weichao and others},
  journal={arXiv preprint},
  year={2024},
  note={Planning horizon as latent confounder}
}
```

## Additional Documentation

- **Theoretical principles**: `../PLANNING_DEPTH_PRINCIPLES.md`
- **Complete AIRL design**: `../AIRL_DESIGN.md`
- **Verification results**: `../DEPTH_VARIABLE_VERIFICATION.md`
- **Implementation status**: `../IMPLEMENTATION_STATUS.md`
- **Research discussion**: `../RESEARCH_DISCUSSION.md`

## License

This code is based on the van Opheusden et al. (2023) codebase.
Please refer to the original repository for licensing information.

## Contact

For questions about this implementation:
- Original codebase: [van Opheusden et al. GitHub]
- AIRL framework: [imitation library documentation](https://imitation.readthedocs.io/)
- Planning-IRL theory: Yao et al. (2024) - IRL and Planning

---

**Last Updated**: 2025-12-23
**Status**: Phase 1 Complete, Phase 2 Ready
**Next Step**: Implement `BasicRewardNet` and AIRL training pipeline
