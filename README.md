# Planning-Aware IRL/AIRL

**Modeling planning mechanisms as explicit factors in Inverse Reinforcement Learning for expertise and clinical trait prediction**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Status**: Phase 0 (Feasibility Study) - In Progress
**Last Updated**: 2025-12-26

---

## 🎯 Research Questions

This project is a **feasibility study** for applying Planning-Aware AIRL to behavioral data. We test the approach on 4-in-a-row (van Opheusden et al., 2023) before applying to pedestrian crossing task.

### Primary Research Questions

**RQ1: Model Feasibility**
> Can fixed planning depth *h* be explicitly modeled and identified from behavior alone?

- **H1a**: Different *h* values produce measurably different behavioral patterns
- **H1b**: AIRL discriminator can learn to distinguish trajectories by *h*
- **Success Criteria**:
  - KL divergence > 0.1 between h=1 vs h=4
  - Discriminator accuracy > 70% on held-out test set

**RQ2: Replication & Extension**
> Can we replicate van Opheusden et al. (2023) findings using Planning-Aware AIRL?

- **Original Finding**: Experts use *shallower* planning (PV depth: Expert 6.2 vs Novice 7.3 steps, p<0.01)
- **Our Test**: Does inferred *h* from AIRL correlate with expertise?
- **Success Criteria**:
  - If negative correlation → replicates original (efficient planning)
  - If positive correlation → needs theoretical explanation
  - If no correlation → *h* may not capture expertise

**RQ3: Parameter Identifiability**
> Is *h* an identifiable structural parameter from behavior?

- **Test**: Ground truth recovery from synthetic data
- **Success Criteria**:
  - Train D₁, D₄ on h=1, h=4 data
  - Cross-validation accuracy > 70%
  - Within-participant consistency (same h across sessions)

**RQ4: Expertise Prediction**
> Does inferred *h* predict expertise (Elo rating)?

- **Analysis**: Correlation and group comparison (n=40 participants)
- **Success Criteria**:
  - |Spearman r| > 0.4, p < 0.05 (adequate power with n=40)
  - Or meaningful group difference (Cohen's d > 0.9)

---

## 📊 Dataset

**Source**: van Opheusden et al. (2023) 4-in-a-row behavioral dataset

- **Participants**: 40 humans
- **Trials**: 67,331 game moves
- **Elo Ratings**: 1464-1535 (computed from 318 human-vs-human games)
- **Expertise Groups**: 10 experts, 20 intermediate, 10 novices (tertile-based)

**Data files**:
- `opendata/raw_data.csv`: Trial-level behavioral data
- `opendata/model_fits_main_model.csv`: Van Opheusden model parameters
- `data/human_elo_ratings.csv`: Elo ratings with expertise labels
- `data/human_game_results.csv`: Game outcomes (318 games)

**Reference**: https://www.nature.com/articles/s41586-023-06124-2

---

## 🔬 Analysis Pipeline

### Phase 0: Feasibility Study ⭐ **Current Phase**

**Goal**: Test whether Planning-Aware AIRL is viable before applying to pedestrian task

**Status**: Data generation complete, validation in progress

#### Methodological Innovation: Fixed Horizon Wrapper

**Problem**: Variable episode lengths create confounding
```
h=1 episodes: avg 17 steps
h=4 episodes: avg 26 steps
→ Discriminator can cheat by learning episode length!
```

**Solution**: Pedestrian-style fixed horizon wrapper
```python
# All episodes padded to 36 steps
# Observations: 89-dim → 90-dim (+ absorbing flag)
# Absorbing states: explicit termination signal
```

**Impact**:
- ✅ Removes length-based confounding
- ✅ Forces discriminator to learn behavioral patterns
- ✅ Enables valid statistical comparison

**Reference**: Pedestrian project (`project_pedestrian/analysis/irl/util.py`)

#### Phase 0 Steps

| Step | Task | Status | Output |
|------|------|--------|--------|
| **0.1** | Generate fixed-horizon data (h=1,4) | ✅ Complete | 100 eps × 2h = 200 episodes |
| **0.2** | Validate data (KL/JS divergence) | 🔄 In Progress | RQ1a answer |
| **0.3** | Pilot AIRL training (h=1) | 📋 Planned | Feasibility check |
| **0.4** | Full AIRL training (h=1,4) | 📋 Planned | RQ1b answer |
| **0.5** | Cross-evaluation | 📋 Planned | RQ3 answer |
| **0.6** | Human data extraction | 📋 Planned | Phase 2.5 prep |
| **0.7** | Expertise analysis | 📋 Planned | RQ2, RQ4 answer |

**Key Files**:
- `fourinarow_airl/fixed_horizon_wrapper.py`: Fixed horizon implementation
- `fourinarow_airl/generate_training_data_fixed_horizon.py`: Data generation
- `fourinarow_airl/IMPLEMENTATION_GUIDE.md`: Complete execution guide
- `MASTER_TODO.md`: Detailed progress tracking with decision log

**Decision Points**:
```
Step 0.2: KL < 0.1? → STOP (h doesn't create difference)
Step 0.3: Disc acc < 0.7? → Debug hyperparameters
Step 0.5: Accuracy < 70%? → h not identifiable
Step 0.7: No correlation? → Negative result (still valuable!)
```

---

### Phase 1: Behavioral Modeling ✅ **Complete**

**Goal**: Understand van Opheusden (2023) dataset and establish baselines

#### Key Findings

**Planning Depth Pattern**:
```
Expert:  6.23 ± 1.30 steps (PV depth)
Novice:  7.29 ± 0.55 steps
Correlation with Elo: r = -0.50, p < 0.01

→ Experts use SHALLOWER planning (efficient, not brute-force)
```

**Expertise Classification**:
- AUC: 0.982 (parameters only)
- Top features: log-likelihood, pruning threshold
- Planning depth coefficient: -0.59 (deeper → novice direction)

**Implications for Phase 0**:
- Hypothesis: Inferred h should be LOWER for experts
- If our result differs → needs theoretical explanation

---

### Phase 2: Planning-Aware AIRL 🚧 **Redesigned**

**Previous Status**: 71% complete (Option B baseline)

**Current Status**: Restarted with Pedestrian methodology

**Major Changes**:
1. ❌ **Removed**: Variable-length trajectories (89-dim obs)
2. ✅ **Added**: Fixed horizon wrapper (90-dim obs with absorbing flag)
3. ✅ **Added**: Large-scale data generation (100+ episodes per h)
4. ✅ **Added**: Proper statistical framework (participant-level analysis)

**Rationale**:
- Variable horizon creates confounding bias
- Small sample size (5 episodes) lacks statistical power
- Need rigorous validation before pedestrian application

**New Approach**:
```python
# Data: 100 episodes × 2 h values
# All episodes: exactly 36 steps (fixed horizon)
# Observation: (37, 90) - includes absorbing flag
# AIRL training: 10K rounds per h
# Evaluation: Cross-validation, participant-level analysis
```

---

### Phase 2.5: Human Data Analysis 📋 **Planned**

**Goal**: Test RQ2 and RQ4 on real human data

**Data Source**:
- 40 participants from `opendata/raw_data.csv`
- Extract human-vs-human game trajectories
- Convert to fixed-horizon format (37, 90)

**Analysis**:
```python
# Per-participant (n=40, independent samples)
for participant in 1..40:
    scores_h1 = mean([D₁(traj) for traj in participant_trajectories])
    scores_h4 = mean([D₄(traj) for traj in participant_trajectories])
    inferred_h = argmax(scores_h1, scores_h4)

# Statistical tests
- Correlation: inferred_h ~ Elo (Spearman)
- Group test: Expert vs Novice (Mann-Whitney U)
- Effect size: Cohen's d, AUC
```

**Expected Timeline**: After Phase 0 completion

---

### Phase 3: Pedestrian Application 🔮 **Future**

**Condition**: Proceed only if Phase 0 shows positive results

**Requirements**:
- RQ1 ✅: h is identifiable (accuracy > 70%)
- RQ3 ✅: AIRL framework works
- RQ4 ✅ (preferred): h predicts expertise

**If Phase 0 Fails**:
- Negative results are publishable (method paper)
- Reconsider approach or task choice
- Document lessons learned

---

## 📁 Repository Structure

```
xRL_pilot/
├── fourinarow_airl/              # Phase 0 & 2 implementation
│   ├── fixed_horizon_wrapper.py       # ⭐ Pedestrian-style wrapper
│   ├── generate_training_data_fixed_horizon.py  # Large-scale data gen
│   ├── train_airl_fixed_horizon.py    # AIRL training script
│   ├── evaluate_discriminators.py     # Cross-evaluation (RQ3)
│   ├── analyze_generated_data.py      # Data validation (RQ1a)
│   ├── extract_human_trajectories.py  # Phase 2.5 data prep
│   ├── analyze_human_h.py             # Phase 2.5 analysis (RQ2, RQ4)
│   ├── env.py                         # 4-in-a-row environment
│   ├── depth_limited_policy.py        # h-step lookahead policy
│   ├── airl_utils.py                  # Trajectory conversion
│   └── IMPLEMENTATION_GUIDE.md        # Complete execution guide
│
├── data/
│   ├── training_trajectories/         # Generated data (Phase 0)
│   │   ├── trajectories_h1_fixed_horizon.pkl  # 100 episodes, h=1
│   │   ├── trajectories_h4_fixed_horizon.pkl  # 100 episodes, h=4
│   │   └── summary_h*.txt                     # Statistics
│   ├── human_elo_ratings.csv          # Elo ratings (40 participants)
│   ├── human_game_results.csv         # Game outcomes (318 games)
│   └── human_trajectories_fixed_horizon.pkl  # Phase 2.5 (future)
│
├── models/
│   └── airl_fixed_horizon/            # Trained models
│       ├── h1/                        # h=1 discriminator & generator
│       └── h4/                        # h=4 discriminator & generator
│
├── figures/                           # Visualizations
│   ├── data_validation_fixed_horizon.png
│   ├── discriminator_cross_eval.png
│   └── expertise_analysis.png
│
├── opendata/                          # Van Opheusden (2023) data
│   ├── raw_data.csv                   # 67K behavioral trials
│   └── model_fits_main_model.csv      # Original model parameters
│
├── docs/
│   ├── AIRL_DESIGN.md                 # Original design doc
│   ├── FOLDER_STRUCTURE.md            # ⭐ Detailed structure guide
│   └── IMPLEMENTATION_GUIDE.md        # Step-by-step guide
│
├── MASTER_TODO.md                     # ⭐ Progress tracking + decisions
├── README.md                          # This file
└── CLAUDE.md                          # Research plan (project instructions)
```

**Key**: ⭐ = Essential for Phase 0

See `docs/FOLDER_STRUCTURE.md` for complete file descriptions.

---

## 🚀 Quick Start

### Prerequisites

```bash
conda activate pedestrian_analysis  # Reuse pedestrian environment
cd fourinarow_airl
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Phase 0 Execution (Current)

**Step 1: Validate Generated Data** (30 minutes)
```bash
python3 analyze_generated_data.py

# Expected output:
# - All episodes exactly 36 steps ✓
# - KL divergence > 0.1 ✓
# - JS divergence > 0.1 ✓
```

**Step 2: Pilot AIRL Training** (1-2 hours)
```bash
python3 train_airl_fixed_horizon.py --h 1 --airl_train_n_rounds 1000

# Monitor: discriminator accuracy > 0.7
```

**Step 3: Full Training** (overnight)
```bash
# h=1 (4-6 hours)
python3 train_airl_fixed_horizon.py --h 1 --airl_train_n_rounds 10000

# h=4 (4-6 hours)
python3 train_airl_fixed_horizon.py --h 4 --airl_train_n_rounds 10000
```

**Step 4: Evaluation**
```bash
python3 evaluate_discriminators.py

# Expected: Classification accuracy > 70%
```

**See `MASTER_TODO.md` for detailed checklist and decision log.**

---

## 📊 Current Results

### Phase 0: Data Generation ✅

```
Generated Data (2025-12-26):
  h=1: 100 episodes
    - Episode length: 36 steps (fixed)
    - Avg real gameplay: ~17 steps
    - Avg absorbing: ~19 steps

  h=4: 100 episodes
    - Episode length: 36 steps (fixed)
    - Avg real gameplay: ~26 steps
    - Avg absorbing: ~10 steps

Behavioral Difference (preliminary, 5 episodes):
  KL divergence: 0.1642
  JS divergence: 0.2178

  → Promising signal for RQ1a ✓
  → Need validation with full 100 episodes
```

### Phase 1: Baseline Findings (from van Opheusden)

```
Expertise Pattern:
  Expert PV depth: 6.23 ± 1.30
  Novice PV depth: 7.29 ± 0.55
  Correlation: r = -0.50, p < 0.01

  → Experts use shallower planning
  → Hypothesis for RQ2/RQ4
```

---

## 📚 Key References

### Theoretical Foundation

1. **van Opheusden, B., et al. (2023)**. Expertise increases planning depth in human gameplay. *Nature*, 618, 1000-1005.
   - https://www.nature.com/articles/s41586-023-06124-2
   - **Relevance**: RQ2 replication target, PV depth methodology

2. **Yao, W., et al. (2024)**. Planning horizon as a latent confounder in inverse reinforcement learning. *arXiv:2409.18051*.
   - **Relevance**: Theoretical motivation for explicit h modeling

3. **Mhammedi, Z., et al. (2023)**. Reinforcement learning for multi-step inverse kinematics. *arXiv:2304.05889*.
   - **Relevance**: Multi-step perspective on IRL

### Methodological Reference

4. **Kakade, S., & Langford, J. (2002)**. Approximately optimal approximate reinforcement learning. *ICML*.
   - **Relevance**: Variable horizon bias in policy gradient

5. **Imitation Learning Documentation**. Variable horizon environments considered harmful.
   - https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html
   - **Relevance**: Fixed horizon wrapper design

---

## 📝 Citation

If you use this code or methodology, please cite:

```bibtex
@article{vanopheusden2023expertise,
  title={Expertise increases planning depth in human gameplay},
  author={van Opheusden, Bas and others},
  journal={Nature},
  volume={618},
  pages={1000--1005},
  year={2023}
}
```

---
