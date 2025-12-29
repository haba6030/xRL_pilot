# Documentation Index

**Last Updated**: 2025-12-29

This directory contains all research documentation for the Planning-Aware AIRL project.

---

## 📋 Current Results (RQ1-2 Answered)

### [MULTICLASS_RESULTS.md](MULTICLASS_RESULTS.md)
**Multi-Class Discriminator: Binary vs 4-Class Comparison**

**Key Findings**:
- Multi-class discriminator: **93.8% accuracy** (h=1,2,3,4)
- Human planning depth: **E[h] = 2.87 ± 0.08** (NOT h=4!)
- Binary discriminator overclaimed h=4 due to lack of intermediate classes
- Probability distribution: P(h=1)=12.8%, P(h=2)=22.6%, P(h=3)=29.7%, P(h=4)=34.9%

**Contents**:
1. Executive Summary
2. Discriminator Performance Comparison
3. Human Player Results (Binary vs Multi-Class)
4. Why Binary Was Miscalibrated
5. Interpretation: What is E[h]=2.9?
6. Implications for Research Goals
7. Key Takeaways

**Read this for**: Understanding why multi-class is better than binary

---

### [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md)
**Discriminator Validation: Detecting Bias**

**Key Findings**:
- Random policy → h_score = 0.68 (expected 0.5) **❌ Bias detected**
- Greedy 1-step → h_score = 0.42 (expected 0.1-0.3) **⚠️ Weaker signal**
- Binary discriminator has **+0.18 bias** toward h=4
- This explained why all humans were classified as h≈4

**Contents**:
1. Executive Summary
2. Test Results (Random, Greedy, Entropy)
3. Integrated Analysis
4. Implications for Human Results
5. Why Synthetic h=1 vs h=4 Still Works
6. Conclusions & Next Steps

**Read this for**: Understanding discriminator calibration issues

---

### [HUMAN_H_ANALYSIS.md](HUMAN_H_ANALYSIS.md)
**Human Planning Depth Estimation**

**Key Findings**:
- All 40 players: h_score > 0.78 (binary discriminator)
- Mean h_score: 0.936 ± 0.044 (very low variance)
- 5 possible interpretations explored
- Conclusion: Need multi-class for accurate estimation

**Contents**:
1. Per-Player Results (Binary Discriminator)
2. Distribution Analysis
3. Interpretations (5 hypotheses)
4. Comparison with Random/Greedy
5. Key Questions
6. Next Steps (Multi-Class Discriminator)

**Read this for**: Understanding binary discriminator results on humans

---

### [RQ_PROGRESS.md](RQ_PROGRESS.md)
**Research Question Progress Tracker**

**Status Summary**:
- **RQ1**: ✅ VALIDATED - Planning depth is identifiable (93.8% acc)
- **RQ2**: ✅ ANSWERED - Humans plan E[h]=2.87 (not h=4)
- **RQ3**: 🔄 IN PROGRESS - Expertise discrimination
- **RQ4**: ⏳ FUTURE - Clinical applications

**Contents**:
1. Objective Definitions (5 objectives)
2. Progress Status Per Objective
3. Key Results & Findings
4. Data Inventory & Splits
5. Next Steps

**Read this for**: Quick overview of project status

---

## 🗂️ Historical Documentation (Moved to backup/)

### Methodology & Implementation
- **BREAKTHROUGH_SUMMARY.md**: Journey to KL=0.1049 (separate encoders discovery)
- **CODE_WALKTHROUGH.md**: Detailed code flow with h tracking
- **IMPLEMENTATION_GUIDE.md**: Step-by-step implementation guide
- **MHAMMEDI_COMPARISON.md**: Theory comparison with Mhammedi(2023)

### Phase Results
- **STEP03_AIRL_DISCRIMINATOR.md**: Binary discriminator results (98.3% accuracy)
- **CONTINUOUS_H_ROADMAP.md**: Future directions (continuous h)

**Location**: `backup/outdated_docs/`
**Note**: These are still valuable references but represent earlier stages of the project

---

## 📊 Quick Reference

### For Understanding Results
1. Start with **README.md** (project overview)
2. Read **MULTICLASS_RESULTS.md** (main findings)
3. Check **VALIDATION_RESULTS.md** (why it matters)

### For Implementing Methods
1. Read **README.md** (pipeline overview)
2. Check code comments in:
   - `preprocess_multistep_ik_data.py`
   - `train_separate_h_models.py`
   - `generate_trajectories_separate_h.py`
   - `train_multiclass_discriminator.py`
3. Reference **backup/outdated_docs/IMPLEMENTATION_GUIDE.md** for details

### For Understanding Theory
1. **README.md** → Methodology section
2. **backup/outdated_docs/MHAMMEDI_COMPARISON.md** → Theory
3. **backup/outdated_docs/CODE_WALKTHROUGH.md** → h's four roles

---

## 📈 Key Metrics Summary

| Metric | Value | Source |
|--------|-------|--------|
| Multi-class accuracy | 93.8% | [MULTICLASS_RESULTS.md](MULTICLASS_RESULTS.md) |
| Binary accuracy | 98.3% | [HUMAN_H_ANALYSIS.md](HUMAN_H_ANALYSIS.md) |
| Human E[h] | 2.87 ± 0.08 | [MULTICLASS_RESULTS.md](MULTICLASS_RESULTS.md) |
| Binary discriminator bias | +0.18 | [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md) |
| KL divergence (h=1 vs h=4) | 0.1049 | backup/BREAKTHROUGH_SUMMARY.md |
| Sample size | 40 players | [RQ_PROGRESS.md](RQ_PROGRESS.md) |

---

## 🔄 Document Status

### Active (docs/)
- ✅ MULTICLASS_RESULTS.md - Main findings
- ✅ VALIDATION_RESULTS.md - Calibration analysis
- ✅ HUMAN_H_ANALYSIS.md - Binary discriminator results
- ✅ RQ_PROGRESS.md - Progress tracker

### Archived (backup/outdated_docs/)
- 📦 BREAKTHROUGH_SUMMARY.md - Historical (KL=0.1049)
- 📦 STEP03_AIRL_DISCRIMINATOR.md - Historical (binary)
- 📦 CODE_WALKTHROUGH.md - Reference
- 📦 IMPLEMENTATION_GUIDE.md - Reference
- 📦 MHAMMEDI_COMPARISON.md - Reference
- 📦 CONTINUOUS_H_ROADMAP.md - Future (incomplete)

---

## 📝 Contributing

When adding new documentation:
1. Create file in `docs/` directory
2. Add entry to this index
3. Update README.md if major finding
4. Keep filename descriptive and uppercase
5. Include date and key findings at top

---

## 🔍 Search Guide

**Looking for...**
- Main results → **MULTICLASS_RESULTS.md**
- Discriminator bias → **VALIDATION_RESULTS.md**
- Binary vs multi-class → **MULTICLASS_RESULTS.md** §3
- Human h distribution → **MULTICLASS_RESULTS.md** §3.2
- Why h=2.87 not h=4 → **MULTICLASS_RESULTS.md** §4
- Validation tests → **VALIDATION_RESULTS.md** §2
- Random policy test → **VALIDATION_RESULTS.md** §2.1
- Greedy policy test → **VALIDATION_RESULTS.md** §2.2
- Project status → **RQ_PROGRESS.md**
- Research questions → **RQ_PROGRESS.md** §1
- Next steps → **README.md** §Next Steps

---

**Maintainer**: Jinil Kim
**Project**: Planning-Aware AIRL
**GitHub**: (to be added)
