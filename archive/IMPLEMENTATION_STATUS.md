# AIRL Implementation Status

## Summary

Phase 1 implementation is **COMPLETE** (2025-12-23). All foundational components for Planning-Aware AIRL on 4-in-a-row are now implemented and tested.

## Phase 1: Foundational Components ✅ COMPLETE

### 1. FourInARowEnv (Gymnasium Environment) ✅
**File**: `fourinarow_airl/env.py`

**Features**:
- Full Gymnasium API compatibility
- 6×6 board with 4-in-a-row rules
- 89-dimensional observation space (72 board + 17 features)
- Discrete(36) action space
- Win/draw/loss detection
- Legal action masking
- ANSI/human rendering modes

**Test Results**:
```bash
$ cd fourinarow_airl && python3 env.py
✓ Environment initialization
✓ State observation (89-dim)
✓ Legal action detection
✓ Move execution
✓ Win detection (4-in-a-row)
✓ Rendering
```

### 2. Van Opheusden Feature Extraction ✅
**File**: `fourinarow_airl/features.py`

**Features**:
- 17 heuristic features matching C++ implementation:
  1. Center control (1 feature)
  2. Connected 2-in-a-row (4 orientations: H, V, D, AD)
  3. Unconnected 2-in-a-row (4 orientations)
  4. 3-in-a-row (4 orientations)
  5. 4-in-a-row (4 orientations)
- Normalized feature values ([0, 1] range)
- Player-perspective features (active/passive)

**Test Results**:
```bash
$ python3 features.py
✓ Board parsing (36-dim binary → 6×6)
✓ Pattern detection (connected/unconnected)
✓ Orientation handling (H/V/D/AD)
✓ Feature normalization
Sample output:
  Center control: 0.500
  Connected 2 (H): 0.333
  3-in-a-row (H): 0.167
```

### 3. Expert Trajectory Loading ✅
**File**: `fourinarow_airl/data_loader.py`

**Features**:
- Loads from `opendata/raw_data.csv` (67,331 trials)
- Game reconstruction from trial-level data
- Boundary detection (identifies when new games start)
- Player filtering (Black/White/Both)
- Converts to `GameTrajectory` dataclass
- Imitation library format conversion

**Test Results**:
```bash
$ python3 data_loader.py
Loaded 67331 trials
Reconstructed 5061 games
Created 10 trajectories
Average trajectory length: 9.9 moves

✓ Game boundary detection
✓ Board state parsing
✓ Action extraction
✓ Reward assignment (+1/-1/0)
✓ Trajectory format conversion
```

### 4. BFS Policy Wrapper ✅
**File**: `fourinarow_airl/bfs_wrapper.py`

**Features**:
- `BFSParameters` dataclass for model parameters
- Loads from `model_fits_main_model.csv`
- Parameter access for all 40 participants
- Baseline policy implementation (uniform + lapse)
- Ready for C++ BFS integration

**Test Results**:
```bash
$ python3 bfs_wrapper.py
Loaded parameters for 40 participants

Participant 1 parameters:
  Pruning threshold: 2.8756
  Stopping probability: 0.0100
  Lapse rate: 0.0180
  Feature drop rate: 0.2000
  Log-likelihood: 2.0921

✓ Parameter loading (all participants)
✓ Action probability calculation
✓ Action sampling with lapse rate
```

## File Structure

```
xRL_pilot/
├── AIRL_DESIGN.md                    # Complete AIRL architecture (from previous work)
├── RESEARCH_DISCUSSION.md            # Research decisions and open questions
├── IMPLEMENTATION_STATUS.md          # This file
│
├── fourinarow_airl/                  # NEW: AIRL implementation package
│   ├── __init__.py                   # Package initialization
│   ├── README.md                     # Package documentation
│   ├── env.py                        # ✅ Gymnasium environment
│   ├── features.py                   # ✅ Feature extraction
│   ├── data_loader.py                # ✅ Expert trajectory loading
│   └── bfs_wrapper.py                # ✅ BFS policy wrapper
│
├── opendata/                         # Expert data (existing)
│   ├── raw_data.csv                  # 67,331 trials
│   └── model_fits_main_model.csv     # Parameter estimates
│
└── xRL_pilot/Model code/             # Original C++ implementation
    ├── bfs.cpp/h                     # BFS algorithm
    ├── heuristic.cpp/h               # Feature evaluation
    └── board.h                       # Board representation
```

## Next Steps: Phase 2 (AIRL Training)

### Priority 1: Discriminator/Reward Network
**File to create**: `fourinarow_airl/reward_net.py`

```python
class PlanningAwareRewardNet(nn.Module):
    """
    h-conditioned reward network
    Input: (state, action, h_idx) → Output: reward
    """
    def __init__(self, obs_dim=89, action_dim=36, h_values=5):
        # h_embedding: maps h ∈ {2,4,6,8,10} to 8-dim vector
        # MLP: [state + action_onehot + h_emb] → reward
```

**Reference**: `AIRL_DESIGN.md` Section 5.2

### Priority 2: AIRL Training Pipeline
**File to create**: `fourinarow_airl/train_airl.py`

```python
# Load expert trajectories
trajectories = load_expert_trajectories(...)

# Create environment
env = FourInARowEnv()

# Initialize reward network
reward_net = PlanningAwareRewardNet(...)

# Initialize generator (PPO or SAC)
gen_algo = PPO("MlpPolicy", env, ...)

# Create AIRL trainer
trainer = AIRL(
    demonstrations=trajectories,
    reward_net=reward_net,
    gen_algo=gen_algo,
    ...
)

# Train for each h ∈ {2, 4, 6, 8, 10}
for h in [2, 4, 6, 8, 10]:
    trainer.train(total_timesteps=100000)
```

**Reference**: `AIRL_DESIGN.md` Section 6

### Priority 3: BFS Distillation (Optional Enhancement)
**Files to create**:
- `fourinarow_airl/bfs_cpp_interface.py`: ctypes interface to C++ BFS
- `fourinarow_airl/distill_bfs.py`: Neural network behavior cloning

```python
# 1. Generate BFS rollouts with C++ code
# 2. Train neural net to mimic BFS
# 3. Use distilled policy as AIRL generator
```

**Reference**: `AIRL_DESIGN.md` Section 5.1 "BFS Distillation"

## Dependencies

### Currently Required
```bash
pip install gymnasium numpy pandas
```

### For Phase 2
```bash
pip install imitation stable-baselines3 torch tensorboard
```

## Testing Checklist

### Phase 1 ✅ ALL PASSING
- [x] Environment initialization and reset
- [x] Legal action detection
- [x] Move execution (valid/invalid)
- [x] Win detection (horizontal, vertical, diagonal, anti-diagonal)
- [x] Draw detection (full board)
- [x] State observation (89-dim)
- [x] Feature extraction (17 features)
- [x] Game reconstruction from CSV
- [x] Trajectory conversion to imitation format
- [x] Parameter loading (40 participants)
- [x] Action sampling with lapse rate

### Phase 2 (To Be Implemented)
- [ ] Reward network forward pass
- [ ] Discriminator training (expert vs generated)
- [ ] Generator policy training (PPO/SAC)
- [ ] AIRL training convergence
- [ ] Discrimination accuracy metrics
- [ ] Imitation score calculation
- [ ] OOD generalization test

## Key Design Decisions (From AIRL_DESIGN.md)

1. **State Representation**: 89-dim (72 board + 17 Van Opheusden features)
   - Rationale: Incorporates domain knowledge from expert modeling

2. **h-Conditioned Discriminator**: Embedding layer for h ∈ {2,4,6,8,10}
   - Rationale: Planning depth as explicit latent factor (Yao et al., 2024)

3. **BFS Distillation Approach**: Pre-train neural policy to mimic BFS
   - Rationale: Solves non-differentiable BFS problem for gradient-based AIRL

4. **Player Filter**: Train on Black player only (player_filter=0)
   - Rationale: Expert data quality, simpler two-player handling

5. **Fixed Horizon**: Pad trajectories to max length with absorbing states
   - Rationale: Required by AIRL implementation, handles variable game length

## Performance Expectations

Based on pedestrian project AIRL results and van Opheusden analysis:

### Training Metrics
- **Discriminator accuracy**: Should converge to ~0.5 (balanced)
- **Expert accuracy**: ~0.5 (discriminator can't distinguish expert)
- **Generated accuracy**: ~0.5 (discriminator can't distinguish generated)
- **Generator reward**: Should increase over training

### Evaluation Metrics
- **Discrimination test (h → expertise)**: Target AUC > 0.7
  - From van Opheusden: PV depth correlates with expertise
  - Expect learned h to predict novice/expert split

- **Imitation score**: Compare generated trajectories to expert
  - Win rate similarity
  - Move distribution similarity
  - Response time distribution (if available)

## Timeline Estimate

- **Phase 2 (Reward Net + Training)**: 2-3 weeks
  - Reward network implementation: 2-3 days
  - AIRL training pipeline: 3-5 days
  - Training experiments (5 values of h): 1-2 weeks
  - Debugging and tuning: 3-5 days

- **Phase 3 (BFS Distillation)**: 1-2 weeks
  - C++ interface: 3-5 days
  - Distillation training: 5-7 days
  - Integration with AIRL: 2-3 days

**Total estimated**: 4-5 weeks for complete implementation

## Notes

1. **Circular Logic Resolution**: Now using Elo-based classification instead of parameter-based composite scores (see `elo_based_classification.py`)

2. **Depth Correction Mystery**: The "-2" correction in `learning.ipynb` origin is still unclear. Current implementation uses raw PV depth values.

3. **Game Reconstruction**: Successfully detects game boundaries by monitoring piece count. Reconstructed 5,061 games from 67,331 trials.

4. **C++ Integration**: BFS wrapper is currently baseline (uniform + lapse). Full integration requires ctypes bindings or subprocess calls to C++ binary.

5. **Two-Player Handling**: Currently filtering to Black player only. White player modeling requires separate consideration (opponent model).

## References

- **AIRL_DESIGN.md**: Complete architectural design with code templates
- **RESEARCH_DISCUSSION.md**: Research questions and unresolved issues
- **fourinarow_airl/README.md**: Package documentation and usage examples
- van Opheusden et al. (2023): Original C++ implementation and data
- Imitation library docs: https://imitation.readthedocs.io/

## Conclusion

**Phase 1 is COMPLETE and TESTED**. All foundational components are working:
- ✅ Environment (Gymnasium API)
- ✅ Features (17-dim Van Opheusden)
- ✅ Data loading (67K trials → 5K games)
- ✅ Parameter access (40 participants)

**Ready to proceed to Phase 2**: AIRL training pipeline implementation.
