# Planning-Aware IRL/AIRL

**Modeling planning mechanisms as explicit factors in Inverse Reinforcement Learning for expertise and clinical trait prediction**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Research Goals

This project investigates whether **planning depth** can be explicitly modeled as an inferable parameter in IRL/AIRL, improving:

1. **Expertise discrimination**: Can planning depth distinguish experts from novices?
2. **Reward identifiability**: Does modeling planning improve IRL interpretability?
3. **Clinical prediction**: Can planning mechanisms explain clinical traits (e.g., anxiety)?
4. **Neural correlates**: Do planning parameters map to fMRI activity patterns?

Building on:
- **van Opheusden et al. (2023)**: Expertise increases planning depth in 4-in-a-row
https://www.nature.com/articles/s41586-023-06124-2 
- **Yao et al. (2024)**: Planning horizon as latent confounder in IRL
https://arxiv.org/abs/2409.18051 
- **Mhammedi et al. (2023)**: Multi-step inverse RL perspective
https://arxiv.org/abs/2304.05889 

---

## 📊 Dataset

**Source**: van Opheusden et al. (2023) 4-in-a-row behavioral dataset
- **Participants**: 40 humans
- **Trials**: 67,331 game moves
- **Conditions**: learning, time pressure, eye tracking, fMRI, generalization
- **Models**: 22 cognitive model variants (ablations, alternatives)

**Data files**: `opendata/` (CSV format)

---

## 🔬 Analysis Pipeline

### Phase 1: Behavioral Modeling ✅ (In Progress)

**Status**: Data exploration and baseline analysis complete

1. **Data reanalysis** (`data_reanalysis.py`)
   - Parameter distributions
   - Expertise classification (composite score)
   - Visualization suite

2. **Model comparison** (`model_comparison_analysis.py`)
   - Compare 22 model variants
   - Log-likelihood ranking
   - Participant-level preferences

3. **Immediate analysis** (`immediate_analysis.py`)
   - Planning depth vs expertise
   - Discrimination tests (AUC)
   - Response time correlations

**Key Finding**: Expert players show *shallower* planning depth than novices (p=0.01), suggesting efficient rather than deep planning.

### Phase 2: Planning-Aware AIRL 🚧 (71% Complete)

**Goal**: Implement Planning-Aware AIRL with discrete planning depth h ∈ {1,2,4,8}

**Main Approach**: **Option A (Pure NN)** ⭐ - Random 초기화, 순수 AIRL 학습
**Baseline**: Option B (BC) - BFS → BC → AIRL (Steps A-E 완료)

**Status**: Baseline 완료 (71%), Main experiments 진행 예정

#### ✅ Completed Steps (Option B - Baseline)

| Step | Description | File | Status |
|------|-------------|------|--------|
| A | h-specific 학습 데이터 생성 | `generate_training_data.py` | ✅ |
| B | Behavior Cloning (BC) | `train_bc.py` | ✅ |
| C | BC를 PPO로 래핑 | `create_ppo_generator.py` | ✅ |
| D | Depth-AGNOSTIC 보상 네트워크 | `create_reward_net.py` | ✅ |
| E | AIRL 학습 | `train_airl.py` | ✅ |
| F | Multi-Depth 비교 | (next) | 🔄 |
| G | 평가 및 분석 | (planned) | 📋 |

#### 🔄 Next: Option A Main Experiments

- [ ] Option A 학습 (h=1,2,4,8) - 50K-100K steps each
- [ ] Performance evaluation
- [ ] Option A vs B comparison

**핵심 원칙**: Planning depth h는 **Policy에만** 존재, **Reward Network**에는 없음

```python
# ✅ CORRECT
policy = DepthLimitedPolicy(h=h)              # h HERE
reward_net = create_reward_network(env)       # NO h!
observations.shape == (T+1, 89)               # NO h!
```

**Quick Start**:
```bash
cd fourinarow_airl
conda activate pedestrian_analysis
export KMP_DUPLICATE_LIB_OK=TRUE

# Run full pipeline
python3 generate_training_data.py --num_episodes 100
python3 train_bc.py --n_epochs 50
python3 create_ppo_generator.py
python3 train_airl.py --total_timesteps 50000
```

**문서**: [PHASE2_PROGRESS.md](progress/PHASE2_PROGRESS.md), [AIRL_DESIGN.md](docs/AIRL_DESIGN.md)

### Phase 3: Clinical & Neural 🔮 (Planned)

**Goal**: Apply Planning-Aware AIRL to clinical traits and neural correlates

1. **Clinical modeling**
   - Clinical traits → planning parameters
   - Explainable individual differences

2. **Neural correlates**
   - fMRI trial-wise regressors
   - Planning parameter mapping

---

## 🚀 Quick Start

### Installation after setting conda env

```bash
# Clone repository
git clone https://github.com/haba6030/xRL_pilot.git
cd xRL_pilot

# Install Python dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# Optional: Jupyter for notebooks
pip install jupyter
```

### Run Analyses

```bash
# Data reanalysis
python data_reanalysis.py

# Model comparison
python model_comparison_analysis.py

# Immediate analysis (requires depth_by_session.txt)
python immediate_analysis.py

# View results
open analysis_*.png
```

### Explore Data

```python
import pandas as pd

# Load raw behavioral data
raw = pd.read_csv('opendata/raw_data.csv')
print(f"Trials: {len(raw)}, Participants: {raw['participant'].nunique()}")

# Load model fits
main_model = pd.read_csv('opendata/model_fits_main_model.csv')
print(main_model[['pruning threshold', 'lapse rate', 'log-likelihood']].describe())
```

---

## 📁 Repository Structure

```
xRL_pilot/
├── fourinarow_airl/          # Phase 2 implementation (Planning-Aware AIRL)
│   ├── generate_training_data.py  # Step A
│   ├── train_bc.py                # Step B
│   ├── create_ppo_generator.py    # Step C
│   ├── create_reward_net.py       # Step D
│   ├── train_airl.py              # Step E
│   ├── airl_utils.py              # Utilities
│   └── fourinarow_env.py          # Environment
├── data/
│   ├── training_trajectories/     # Step A outputs
│   └── expert_trajectories/       # Expert data
├── models/
│   ├── bc_policies/               # Step B outputs
│   ├── ppo_generators/            # Step C outputs
│   └── airl_results/              # Step E outputs
├── opendata/                  # Phase 1 experimental data (CSV)
│   ├── raw_data.csv          # 67K trials
│   └── model_fits_*.csv      # 22 model variants
├── papers/                    # Reference papers (PDF)
├── xRL_pilot/                # van Opheusden (2023) codebase
│   ├── Model code/           # C++ implementation
│   │   ├── bfs.cpp           # Best-first search + PV depth
│   │   ├── heuristic.cpp     # 17 feature weights
│   │   └── matlab wrapper/   # Parameter fitting (BADS)
│   └── Analysis notebooks/   # Jupyter notebooks
├── *.py                      # Phase 1 analysis scripts
├── AIRL_DESIGN.md            # Phase 2 design document
├── PHASE2_PROGRESS.md        # Phase 2 progress tracking
├── IMPLEMENTATION_NOTES.md   # Technical implementation details
├── CLAUDE.md                 # Full research plan
└── README.md                 # This file
```

**Phase 2 문서**:
- [AIRL_DESIGN.md](docs/AIRL_DESIGN.md) - Planning-Aware AIRL 설계
- [AIRL_COMPLETE_GUIDE.md](docs/AIRL_COMPLETE_GUIDE.md) - 전체 실행 가이드 ⭐
- [PHASE2_PROGRESS.md](progress/PHASE2_PROGRESS.md) - 현재 진행 상황 (71% complete)
- [IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) - 구현 기술 참고사항

**Phase 1 문서** (archived):
- [PROJECT_SUMMARY.md](archive/PROJECT_SUMMARY.md) - Phase 1 detailed documentation
- [FOLDER_STRUCTURE.md](archive/FOLDER_STRUCTURE.md) - Complete directory guide

---

## 📈 Current Results

### Expertise Classification

**Baseline (parameters only)**:
- AUC: **0.982**
- Accuracy: 96.7%
- Top features: log-likelihood (+1.76), pruning threshold (+1.46)

**With planning depth**:
- AUC: 0.987 (marginal improvement)
- **Finding**: Depth coefficient is *negative* (-0.59)
  - Deeper planning → Novice direction
  - Supports "efficient planning" hypothesis

### Planning Depth Pattern

```
Expert:  6.23 ± 1.30 steps
Novice:  7.29 ± 0.55 steps
p = 0.011

Correlation with performance: r = -0.50 (p < 0.01)
→ Deeper planning associated with *worse* performance
```

**Interpretation**: Expertise reflects efficient pruning, not brute-force depth.

### Model Comparison

**Log-likelihood ranking** (higher = better):
1. MCTS: 2.00
2. No pruning: 2.00
3. Main model: 1.95
4. Fixed depth: 1.94

---

## 🤝 Contributing

This is a research project. For collaboration inquiries:
- See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for research overview
- New contributors: Read documentation in order (README → PROJECT_OVERVIEW → AIRL_DESIGN)
- Questions: Open an issue on GitHub

---

## 📚 References

1. **van Opheusden, B., et al. (2023)**. Expertise increases planning depth in human gameplay. *Nature*.
2. **Yao, W., et al. (2024)**. Planning horizon as a latent confounder in inverse reinforcement learning.
3. **Mhammedi, Z., et al. (2023)**. Reinforcement learning for multi-step inverse kinematics.

---

## 📄 License

MIT License (see LICENSE file)

---

## 🔗 Links

- **Original codebase**: [van Opheusden et al. (2023)](https://github.com/original-repo)
- **Project overview**: See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- **Design document**: See [docs/AIRL_DESIGN.md](docs/AIRL_DESIGN.md)

---

**Last Updated**: 2025-12-26
**Current Phase**: Phase 2 - Planning-Aware AIRL (71% complete)
