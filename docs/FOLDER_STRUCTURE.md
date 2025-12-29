# Project Folder Structure

**Planning-Aware IRL/AIRL - Detailed Directory Guide**

**Last Updated**: 2025-12-26
**Status**: Phase 0 (Feasibility Study)

---

## Directory Tree

```
xRL_pilot/
├── fourinarow_airl/              # Phase 0 & 2: Planning-Aware AIRL Implementation
├── data/                         # Generated and processed data
├── models/                       # Trained AIRL models
├── figures/                      # Visualizations and plots
├── opendata/                     # Van Opheusden (2023) original dataset
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── papers/                       # Reference PDFs
├── xRL_pilot/                    # Original van Opheusden codebase (C++)
├── archive/                      # Deprecated/old code
├── MASTER_TODO.md                # ⭐ Progress tracking + decision log
├── README.md                     # Main project overview
└── CLAUDE.md                     # Research plan (project instructions)
```

---

## `/fourinarow_airl/` - Main Implementation

**Purpose**: Phase 0 (Feasibility Study) and Phase 2 (AIRL Training) implementation

### Core Files (⭐ = Essential)

#### Data Generation
- **`generate_training_data_fixed_horizon.py`** ⭐
  - Generate fixed-horizon trajectories (h=1, 2, 4)
  - Output: `data/training_trajectories/trajectories_h{1,4}_fixed_horizon.pkl`
  - Usage: `python3 generate_training_data_fixed_horizon.py --h 1 --num_episodes 100`

#### Environment
- **`env.py`** ⭐
  - 4-in-a-row environment (6x6 board)
  - State: 89-dim (72 board + 17 features)
  - Action: 36 positions (0-35)

- **`fixed_horizon_wrapper.py`** ⭐⭐⭐
  - Pedestrian-style fixed horizon wrapper
  - Pads episodes to 36 steps with absorbing states
  - Observation: 89 → 90 dim (+ absorbing flag)
  - **Critical for removing length confounding**

#### Policy
- **`depth_limited_policy.py`** ⭐
  - h-step lookahead planning policy
  - Uses van Opheusden heuristic features
  - Implements DepthLimitedPolicy(h, params, beta, lapse_rate)

- **`bfs_wrapper.py`**
  - BFS parameter loading from van Opheusden model fits
  - `BFSParameters` dataclass
  - `load_all_participant_parameters()` function

- **`features.py`**
  - Van Opheusden 17 feature extraction
  - Board state → feature vector conversion

#### AIRL Training
- **`train_airl_fixed_horizon.py`** ⭐ (To be created)
  - Main AIRL training script
  - Supports h={1,2,4}
  - Configuration: 10K rounds, [8,8] reward net
  - Output: `models/airl_fixed_horizon/h{1,4}/`

- **`airl_utils.py`** ⭐
  - Trajectory format conversion
  - `convert_to_imitation_format_fixed_horizon()` (needs update)
  - AIRL-specific utilities

#### Evaluation & Analysis
- **`analyze_generated_data.py`** ⭐ (To be created)
  - Validate generated data (Step 0.2)
  - Check: episode length, absorbing ratio, KL/JS divergence
  - Output: `figures/data_validation_fixed_horizon.png`

- **`evaluate_discriminators.py`** ⭐ (To be created)
  - Cross-evaluation (Step 0.5)
  - Load D₁, D₄ and test on held-out data
  - Output: Classification accuracy, confusion matrix

- **`extract_human_trajectories.py`** (To be created)
  - Phase 2.5: Extract human data from `opendata/raw_data.csv`
  - Convert to fixed-horizon format (37, 90)
  - Output: `data/human_trajectories_fixed_horizon.pkl`

- **`analyze_human_h.py`** (To be created)
  - Phase 2.5: Infer h from human data
  - Test RQ2, RQ4
  - Correlation: h ~ Elo, Expert vs Novice

#### Documentation
- **`IMPLEMENTATION_GUIDE.md`** ⭐
  - Complete step-by-step execution guide
  - Pedestrian vs current comparison
  - Code examples for all steps

### Legacy Files (from Phase 2 - Option B)

**Status**: Deprecated (replaced by fixed horizon approach)

- `generate_training_data.py` → Use `generate_training_data_fixed_horizon.py`
- `train_bc.py` → Not used (pure AIRL, no BC)
- `create_ppo_generator.py` → Not used
- `create_reward_net.py` → Not used
- `train_airl.py` → Use `train_airl_fixed_horizon.py`

---

## `/data/` - Data Directory

### Structure

```
data/
├── training_trajectories/        # Generated synthetic data (Phase 0)
│   ├── trajectories_h1_fixed_horizon.pkl  # ✅ Generated (100 episodes)
│   ├── trajectories_h4_fixed_horizon.pkl  # ✅ Generated (100 episodes)
│   ├── summary_h1_fixed_horizon.txt
│   └── summary_h4_fixed_horizon.txt
│
├── human_elo_ratings.csv         # ✅ Elo ratings (40 participants)
├── human_game_results.csv        # ✅ Game outcomes (318 games)
├── elo_summary.json              # ✅ Elo statistics
├── README_ELO.md                 # ✅ Elo documentation
│
└── human_trajectories_fixed_horizon.pkl  # 📋 To be created (Phase 2.5)
```

### File Descriptions

#### Training Data
- **`trajectories_h{1,4}_fixed_horizon.pkl`**
  - Format: List[Dict] with keys:
    - `'observations'`: (37, 90) numpy array
    - `'actions'`: (36,) numpy array
    - `'length'`: int (always 36)
    - `'h'`: int (metadata)
    - `'absorbing_steps'`: int
  - Size: ~100 episodes per h

#### Elo Ratings
- **`human_elo_ratings.csv`**
  - Columns: participant, elo, games_played, wins, losses, draws, win_rate, expertise, expert_binary
  - Expertise: 'expert' (n=10), 'intermediate' (n=20), 'novice' (n=10)
  - Used for RQ2, RQ4

---

## `/models/` - Trained Models

### Structure

```
models/
└── airl_fixed_horizon/           # Fixed horizon AIRL models
    ├── h1/                       # h=1 models
    │   ├── reward_net.pt         # Discriminator weights
    │   ├── generator_ppo.zip     # PPO generator
    │   └── training_log.csv      # TensorBoard log
    └── h4/                       # h=4 models
        ├── reward_net.pt
        ├── generator_ppo.zip
        └── training_log.csv
```

### Model Files

- **`reward_net.pt`** ⭐
  - PyTorch state dict
  - Architecture: BasicRewardNet([8, 8], LeakyReLU)
  - Input: (obs, action, next_obs) - obs is (90,)
  - Output: scalar reward

- **`generator_ppo.zip`**
  - Stable-Baselines3 PPO model
  - Used for rollout generation during AIRL training
  - Not used for h inference (only discriminator)

- **`training_log.csv`**
  - TensorBoard log
  - Metrics: disc_acc, disc_acc_expert, disc_acc_gen, mean_reward

---

## `/figures/` - Visualizations

### Current Figures

- `action_distributions.png` - Action distributions (h verification)
- `data_validation_fixed_horizon.png` - Step 0.2 validation (to be created)
- `discriminator_cross_eval.png` - Step 0.5 cross-evaluation (to be created)
- `expertise_analysis.png` - Phase 2.5 results (to be created)

---

## `/opendata/` - Van Opheusden (2023) Dataset

### Files

- **`raw_data.csv`** (67,331 rows)
  - Columns: participant, experiment, color, move, black_pieces, white_pieces, response_time, session number, cross-validation group, time limit
  - Experiments: 'human-vs-human', 'learning', 'time-pressure', etc.

- **`model_fits_main_model.csv`** (40 participants)
  - Van Opheusden cognitive model parameters
  - Columns: pruning threshold, stopping probability, lapse rate, feature drop rate, feature weights (17), log-likelihood

**Usage**:
- Phase 1: Baseline analysis ✅
- Phase 2.5: Extract human trajectories

---

## `/docs/` - Documentation

- **`FOLDER_STRUCTURE.md`** ⭐ (This file)
- **`AIRL_DESIGN.md`** - Original design document (pre-fixed horizon)
- **`IMPLEMENTATION_GUIDE.md`** - Step-by-step execution guide

---

## `/scripts/` - Utility Scripts

- **`compute_elo_ratings.py`** ✅
  - Compute Elo ratings from human-vs-human games
  - Output: `data/human_elo_ratings.csv`, `data/human_game_results.csv`
  - Reusable script (can re-run if data updated)

---

## `MASTER_TODO.md` ⭐⭐⭐

**Purpose**: Central progress tracking with decision log

**Structure**:
- Phase 0 checklist with success criteria
- Results log (actual values)
- Decision log (go/no-go decisions at each step)
- Expectations vs Reality comparison

**See**: `MASTER_TODO.md` for detailed tracking

---

## File Naming Conventions

### Python Scripts
- `{action}_{object}.py` - e.g., `generate_training_data.py`
- `{action}_{object}_fixed_horizon.py` - Fixed horizon version

### Data Files
- `{type}_h{value}_fixed_horizon.pkl` - e.g., `trajectories_h1_fixed_horizon.pkl`
- `{type}.csv` - e.g., `human_elo_ratings.csv`

### Model Files
- `{model_type}.pt` - PyTorch models
- `{model_type}.zip` - Stable-Baselines3 models

### Documentation
- `{TOPIC}_{TYPE}.md` - All caps for main docs

---

## Deprecated/Archive

### `/archive/` (if created)
- Old Phase 2 Option B files
- Variable-length trajectory code
- Pre-fixed horizon experiments

**Note**: Not deleted for reference, but not actively maintained

---

## Quick Navigation

**Starting Phase 0?**
1. Read: `README.md`
2. Follow: `fourinarow_airl/IMPLEMENTATION_GUIDE.md`
3. Track: `MASTER_TODO.md`

**Looking for data?**
- Generated: `data/training_trajectories/`
- Original: `opendata/`
- Elo ratings: `data/human_elo_ratings.csv`

**Looking for code?**
- Data generation: `fourinarow_airl/generate_training_data_fixed_horizon.py`
- Fixed horizon: `fourinarow_airl/fixed_horizon_wrapper.py`
- AIRL training: `fourinarow_airl/train_airl_fixed_horizon.py` (to be created)

**Need help?**
- Implementation guide: `fourinarow_airl/IMPLEMENTATION_GUIDE.md`
- This structure: `docs/FOLDER_STRUCTURE.md`
- Research plan: `CLAUDE.md`

---

**Last Updated**: 2025-12-26
**Maintained By**: [Your Name]
