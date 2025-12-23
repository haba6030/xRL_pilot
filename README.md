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

### Phase 2: Fixed-h Modeling 🚧 (Next)

**Goal**: Implement discrete planning depth h ∈ {1,2,3,4,5} as explicit parameter

1. **C++ implementation**
   - Modify `heuristic` class for fixed depth
   - Recompile with depth constraints

2. **Parameter fitting**
   - Use MATLAB BADS optimizer
   - Fit (β, lapse) per participant per h
   - Select optimal h via AIC/BIC

3. **Model selection**
   - Optimal h distribution across participants
   - Test h ~ expertise relationship

### Phase 3: Planning-Aware AIRL 📋 (Planned)

**Goal**: Compare standard AIRL vs planning-aware AIRL

1. **AIRL baseline**
   - Infer reward assuming implicit planning
   - Standard discriminator + generator

2. **Planning-constrained AIRL**
   - Treat h as explicit factor
   - Compare h ∈ {1,2,3,4,5} variants
   - Evaluate reward identifiability

3. **Evaluation**
   - Likelihood, OOD generalization
   - Turing test realism

### Phase 4: Clinical & Neural 🔮 (Exploratory)

- Clinical traits → planning parameters
- fMRI trial-wise regressors
- Individual differences mapping

---

## 🚀 Quick Start

### Installation

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
├── opendata/                  # Experimental data (CSV)
│   ├── raw_data.csv          # 67K trials
│   └── model_fits_*.csv      # 22 model variants
├── papers/                    # Reference papers (PDF)
├── xRL_pilot/                # van Opheusden (2023) codebase
│   ├── Model code/           # C++ implementation
│   │   ├── bfs.cpp           # Best-first search + PV depth
│   │   ├── heuristic.cpp     # 17 feature weights
│   │   └── matlab wrapper/   # Parameter fitting (BADS)
│   └── Analysis notebooks/   # Jupyter notebooks
├── *.py                      # Analysis scripts
├── PROJECT_SUMMARY.md        # Detailed project documentation
├── FOLDER_STRUCTURE.md       # Complete directory guide
└── README.md                 # This file
```

See `FOLDER_STRUCTURE.md` for detailed file descriptions.

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
- See `PROJECT_SUMMARY.md` for detailed methodology
- New contributors: Follow "Team Onboarding Guide" in project summary
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
- **Project documentation**: See `PROJECT_SUMMARY.md`
- **Detailed structure**: See `FOLDER_STRUCTURE.md`

---

**Last Updated**: 2024-12-17
