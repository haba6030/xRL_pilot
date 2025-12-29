# MASTER TODO - Planning-Aware AIRL Feasibility Study

**Project**: Planning-Aware IRL/AIRL for Expertise Prediction
**Phase**: Phase 0 (Feasibility Study)
**Start Date**: 2025-12-26
**PI**: [Your Name]

---

## 📋 Overview

This document tracks the entire Phase 0 feasibility study, recording:
1. **Progress**: What's completed, in progress, planned
2. **Results**: Actual experimental results with timestamps
3. **Decisions**: Go/No-Go decisions at critical checkpoints
4. **Expectations**: Success criteria and rationale

**Purpose**: Determine if Planning-Aware AIRL is viable before applying to pedestrian task

---

## 🎯 Research Questions (RQ)

| RQ | Question | Answers in | Success Criteria |
|----|----------|------------|------------------|
| **RQ1** | Can *h* be modeled and identified? | Step 0.2, 0.3, 0.4 | KL>0.1, disc_acc>0.7 |
| **RQ2** | Can we replicate van Opheusden (2023)? | Step 0.7 | Correlation direction matches |
| **RQ3** | Is *h* identifiable from behavior? | Step 0.5 | Classification acc>70% |
| **RQ4** | Does *h* predict expertise? | Step 0.7 | &#124;r&#124;>0.4, p<0.05 |

---

## 📊 Phase 0: Feasibility Study

### Step 0.1: Generate Fixed-Horizon Data ✅

**Status**: **COMPLETE**
**Date Completed**: 2025-12-26
**Time Spent**: ~3 hours (h=1: 30min, h=4: 2.5hr)

#### Task
- [x] Generate h=1 data (100 episodes)
- [x] Generate h=4 data (100 episodes)
- [x] Save trajectories to `data/training_trajectories/`

#### Commands Run
```bash
python3 generate_training_data_fixed_horizon.py --h 1 --num_episodes 100 --seed 42
python3 generate_training_data_fixed_horizon.py --h 4 --num_episodes 100 --seed 43
```

#### Results
```
h=1 Data:
  Total episodes: 100
  Episode length: 36 steps (all fixed)
  Avg absorbing steps: 19.2 ± X  # Fill in after validation
  Avg real gameplay: 16.8 ± X
  File size: ~XXX MB
  Location: data/training_trajectories/trajectories_h1_fixed_horizon.pkl

h=4 Data:
  Total episodes: 100
  Episode length: 36 steps (all fixed)
  Avg absorbing steps: 10.3 ± X  # Fill in after validation
  Avg real gameplay: 25.7 ± X
  File size: ~XXX MB
  Location: data/training_trajectories/trajectories_h4_fixed_horizon.pkl
```

#### Validation Checks
- [x] All episodes exactly 36 steps
- [x] Observation shape: (37, 90) for all
- [x] Action shape: (36,) for all
- [x] Absorbing flag correct (0 for ongoing, 1 for absorbing)

#### Decision
**✅ PROCEED** - Data generation successful, fixed horizon working as expected

---

### Step 0.2: Validate Data (KL/JS Divergence) 🔄

**Status**: **IN PROGRESS**
**Started**: 2025-12-26
**Expected Duration**: 30 minutes

#### Task
- [ ] Run `analyze_generated_data.py`
- [ ] Check episode length distribution (should be constant at 36)
- [ ] Check absorbing state ratio (h=1 vs h=4)
- [ ] Compute KL/JS divergence on real actions (absorbing excluded)
- [ ] Generate validation plots

#### Success Criteria (RQ1a)
```
PASS if ALL of:
  1. All episodes == 36 steps (no variation)
  2. KL divergence > 0.1
  3. JS divergence > 0.1

FAIL if ANY of:
  1. Episode length varies (fixed horizon broken)
  2. KL < 0.05 (too similar)
  3. JS < 0.05 (too similar)
```

#### Expected Results (from 5-episode pilot)
```
KL divergence: 0.15-0.20 (pilot: 0.1642)
JS divergence: 0.20-0.25 (pilot: 0.2178)

If higher: ✅ Even better!
If lower but >0.1: ✅ Still acceptable
If <0.1: ❌ STOP and reconsider h values
```

#### Results (Fill after execution)
```
Date run: ___________
KL divergence: _______
JS divergence: _______
Chi-square: χ²=______, p=______

Plot saved: figures/data_validation_fixed_horizon.png
```

#### Decision Gate
```
IF KL > 0.1 AND JS > 0.1:
  ✅ PROCEED to Step 0.3 (Pilot AIRL)
  Rationale: h creates measurable behavioral difference

ELIF KL > 0.05 AND JS > 0.05:
  ⚠️  DISCUSS: Weak signal, may need:
    - Try h=1 vs h=8 (larger gap)
    - Increase beta (sharper policy)
    - Check if lapse rate too high

ELSE:
  ❌ STOP Phase 0
  Rationale: h doesn't create sufficient behavioral difference
  Alternative: Planning depth may not be identifiable in 4-in-a-row
  Publication: Negative result paper
```

**Decision**: ___ (Fill after analysis)

**Rationale**: _________________________________________

---

### Step 0.3: Pilot AIRL Training (h=1 only) 📋

**Status**: **PLANNED**
**Expected Start**: After Step 0.2 passes
**Expected Duration**: 1-2 hours

#### Task
- [ ] Update `airl_utils.py` for 90-dim observations
- [ ] Create `train_airl_fixed_horizon.py` script
- [ ] Run pilot training (h=1, 1K rounds)
- [ ] Monitor discriminator accuracy
- [ ] Check for training stability (no divergence)

#### Configuration
```python
h = 1
num_expert_episodes = 100
airl_train_n_rounds = 1000  # Pilot (10% of full)
gen_train_timesteps = 512
ppo_n_steps = 512
demo_batch_size = 256
reward_net_hid_sizes = [8, 8]
reward_net_activation = LeakyReLU
disc_optimizer_lr = 1e-3
```

#### Success Criteria (RQ1b - Pilot)
```
PASS if ALL of:
  1. Training completes without errors
  2. Discriminator accuracy converges to >0.6
  3. No gradient explosion/vanishing
  4. Generator learns non-random policy

FAIL if ANY of:
  1. Training diverges (NaN losses)
  2. Disc acc stuck at ~0.5 (random)
  3. Disc acc >0.95 (overfitting to episode IDs)
```

#### Expected Results
```
Final discriminator accuracy: 0.65-0.75
  - disc_acc_expert: 0.70-0.80 (D correctly identifies expert as expert)
  - disc_acc_gen: 0.60-0.70 (D correctly rejects generator)

Training stability: Losses should decrease smoothly
Generator performance: Should learn to play reasonable games
```

#### Results (Fill after execution)
```
Date run: ___________
Training time: _______ minutes
Final disc_acc: _______
Final disc_acc_expert: _______
Final disc_acc_gen: _______

Convergence: [Smooth / Unstable / Failed]
Model saved: models/airl_fixed_horizon/h1_pilot/
```

#### Decision Gate
```
IF disc_acc > 0.6 AND training stable:
  ✅ PROCEED to Step 0.4 (Full AIRL)
  Rationale: AIRL framework works, ready for full training

ELIF disc_acc = 0.5-0.6:
  ⚠️  TUNE HYPERPARAMETERS:
    - Increase reward net size: [8,8] → [32,32]
    - Adjust learning rate: 1e-3 → 5e-4
    - Increase demo_batch_size: 256 → 512
  Then re-run Step 0.3

ELSE:
  ❌ DEBUG REQUIRED:
    - Check data loading (absorbing flag handling)
    - Check reward net architecture
    - Check if discriminator learning from length (should be impossible with fixed horizon)
```

**Decision**: ___ (Fill after pilot)

**Rationale**: _________________________________________

---

### Step 0.4: Full AIRL Training (h=1, h=4) 📋

**Status**: **PLANNED**
**Expected Start**: After Step 0.3 passes
**Expected Duration**: 8-12 hours (overnight run)

#### Task
- [ ] Full training h=1 (10K rounds)
- [ ] Full training h=4 (10K rounds)
- [ ] Save models to `models/airl_fixed_horizon/h{1,4}/`
- [ ] Extract training logs

#### Configuration
```python
airl_train_n_rounds = 10000  # Full training
# Other params same as pilot
```

#### Success Criteria
```
PASS if BOTH models achieve:
  1. Final disc_acc > 0.7
  2. Convergence (no increasing trend in last 1K rounds)
  3. Reasonable generator behavior (not random)

Target: disc_acc ≈ 0.75-0.85
  Too low (<0.7): Weak discriminator
  Too high (>0.95): Overfitting
```

#### Expected Results
```
h=1 Model:
  Final disc_acc: 0.75-0.85
  Convergence round: ~5000-7000

h=4 Model:
  Final disc_acc: 0.75-0.85
  Convergence round: ~5000-7000
```

#### Results (Fill after execution)
```
h=1:
  Date: ___________
  Training time: _______ hours
  Final disc_acc: _______
  Converged at round: _______
  Model: models/airl_fixed_horizon/h1/reward_net.pt

h=4:
  Date: ___________
  Training time: _______ hours
  Final disc_acc: _______
  Converged at round: _______
  Model: models/airl_fixed_horizon/h4/reward_net.pt
```

#### Decision Gate
```
IF BOTH models converged with acc > 0.7:
  ✅ PROCEED to Step 0.5 (Cross-Evaluation)
  Rationale: Models trained successfully

ELIF ONE model failed:
  ⚠️  INVESTIGATE:
    - Compare hyperparameters
    - Check data quality for failed h
    - Re-run failed h with adjusted params

ELSE (both failed):
  ❌ RECONSIDER APPROACH:
    - May need different h values (1 vs 8?)
    - May need larger reward net
    - May need more expert data (>100 episodes)
```

**Decision**: ___ (Fill after training)

**Rationale**: _________________________________________

---

### Step 0.5: Cross-Evaluation (RQ3 Test) 📋

**Status**: **PLANNED**
**Expected Start**: After Step 0.4 complete
**Expected Duration**: 1 hour

#### Task
- [ ] Load trained D₁ and D₄
- [ ] Load test data (held-out 20 episodes each)
- [ ] Compute cross-scoring matrix
- [ ] Compute classification accuracy
- [ ] Generate confusion matrix

#### Analysis Plan
```python
# Cross-scoring matrix
D₁_on_h1_test = mean([D₁(traj) for traj in test_h1])  # Should be HIGH
D₁_on_h4_test = mean([D₁(traj) for traj in test_h4])  # Should be LOW
D₄_on_h1_test = mean([D₄(traj) for traj in test_h1])  # Should be LOW
D₄_on_h4_test = mean([D₄(traj) for traj in test_h4])  # Should be HIGH

# Classification
for traj in test_h1 + test_h4:
    predicted_h = 1 if D₁(traj) > D₄(traj) else 4

accuracy = correct_predictions / total_predictions
```

#### Success Criteria (RQ3)
```
PASS if:
  Classification accuracy > 70%

EXCELLENT if:
  Classification accuracy > 85%

FAIL if:
  Classification accuracy < 60%
  (barely better than chance=50%)
```

#### Expected Results
```
Cross-Scoring Matrix:
           D₁ score  D₄ score
h=1 test   0.8       0.3      ← D₁ prefers h=1
h=4 test   0.3       0.8      ← D₄ prefers h=4

Classification Accuracy: 75-85%
Confusion Matrix:
           Pred h=1  Pred h=4
True h=1      18         2
True h=4       4        16
```

#### Results (Fill after execution)
```
Date: ___________

Cross-Scoring Matrix:
           D₁ score  D₄ score
h=1 test   _______   _______
h=4 test   _______   _______

Classification Accuracy: _______%

Confusion Matrix:
           Pred h=1  Pred h=4
True h=1   _______   _______
True h=4   _______   _______

Plot saved: figures/discriminator_cross_eval.png
```

#### Decision Gate - RQ3 Answer
```
IF accuracy > 70%:
  ✅ RQ3 ANSWER: YES
  Conclusion: h is identifiable from behavior
  Implication: Can proceed to human data analysis
  Publication: Strong positive result
  PROCEED to Step 0.6

ELIF accuracy = 60-70%:
  ⚠️  RQ3 ANSWER: PARTIAL
  Conclusion: h is weakly identifiable
  Options:
    a) Proceed to Step 0.6 with caveat
    b) Try h=1 vs h=8 (larger gap)
  Publication: Method paper with limitations

ELSE (accuracy < 60%):
  ❌ RQ3 ANSWER: NO
  Conclusion: h is NOT identifiable
  Implication: Planning-Aware AIRL doesn't work for 4-in-a-row
  DO NOT proceed to Phase 2.5
  Publication: Negative result (still valuable!)
```

**Decision**: ___ (Fill after evaluation)

**RQ3 Answer**: ___ (YES / PARTIAL / NO)

**Rationale**: _________________________________________

---

### Step 0.6: Extract Human Data 📋

**Status**: **PLANNED**
**Depends On**: Step 0.5 (RQ3 = YES or PARTIAL)
**Expected Duration**: 3-5 days (complex)

#### Task
- [ ] Parse `opendata/raw_data.csv`
- [ ] Identify human-vs-human games
- [ ] Segment into individual games
- [ ] Convert to fixed-horizon format (37, 90)
- [ ] Save per-participant data

#### Challenges
```
1. Game segmentation:
   - No explicit game ID in raw_data.csv
   - Need to detect new games (empty board)

2. Board state parsing:
   - 'black_pieces', 'white_pieces' are 36-char strings
   - Need to convert to 6x6 board → 72-dim state

3. Feature extraction:
   - Need to compute van Opheusden 17 features
   - Use features.py

4. Fixed horizon conversion:
   - Variable-length games → 36 steps
   - Apply FixedHorizonWrapper logic retroactively
```

#### Expected Output
```python
human_data = {
    'participant_1': {
        'elo': 1520,
        'expertise': 'expert',
        'trajectories': [
            {'observations': (37, 90), 'actions': (36,)},
            ...  # 50-200 games
        ],
        'n_games': 127
    },
    ...  # 40 participants
}

# Saved to: data/human_trajectories_fixed_horizon.pkl
```

#### Validation
```
- [ ] All participants have data (n=40)
- [ ] Trajectory shapes correct: (37, 90) observations
- [ ] Absorbing flags set correctly
- [ ] Total games ≈ original dataset
```

#### Results (Fill after completion)
```
Date completed: ___________
Participants processed: _______ / 40
Total games extracted: _______
Avg games per participant: _______
File saved: data/human_trajectories_fixed_horizon.pkl
File size: _______ MB
```

#### Decision
```
IF data extraction successful:
  ✅ PROCEED to Step 0.7

ELSE:
  ⚠️  DEBUG data parsing
  May need to simplify or use subset of participants
```

**Decision**: ___ (Fill after extraction)

---

### Step 0.7: Human Data Analysis (RQ2, RQ4) 📋

**Status**: **PLANNED**
**Depends On**: Step 0.6 complete
**Expected Duration**: 2-3 days

#### Task
- [ ] Load human trajectories
- [ ] Infer h for each participant (D₁ vs D₄ scoring)
- [ ] Test RQ2: h ~ Elo correlation
- [ ] Test RQ4: Expert vs Novice comparison
- [ ] Generate publication-quality plots

#### Analysis Plan
```python
# 1. Per-participant h inference
for pid in 1..40:
    trajs = human_data[pid]['trajectories']
    scores_h1 = [D₁(traj) for traj in trajs]
    scores_h4 = [D₄(traj) for traj in trajs]

    # Participant-level summary
    inferred_h[pid] = 1 if mean(scores_h1) > mean(scores_h4) else 4
    confidence[pid] = abs(mean(scores_h1) - mean(scores_h4))

# 2. RQ2: Correlation test
from scipy.stats import spearmanr
corr, p = spearmanr(inferred_h, Elo)

# 3. RQ4: Group comparison
from scipy.stats import mannwhitneyu
experts = inferred_h[Elo >= Q75]  # n=10
novices = inferred_h[Elo < Q25]   # n=10
U, p = mannwhitneyu(experts, novices)
```

#### Success Criteria

**RQ2 (Replication)**:
```
REPLICATES van Opheusden if:
  Negative correlation (r < 0, p < 0.05)
  Experts → lower h (efficient planning)

CONTRADICTS if:
  Positive correlation (r > 0, p < 0.05)
  Experts → higher h (needs explanation)

NO RESULT if:
  No correlation (p >= 0.05)
```

**RQ4 (Expertise Prediction)**:
```
PASS if:
  |r| > 0.4, p < 0.05  (correlation)
  OR Cohen's d > 0.9 (group difference)

PARTIAL if:
  |r| > 0.3, p < 0.05  (weak but significant)

FAIL if:
  p >= 0.05 (no relationship)
```

#### Expected Results (Hypotheses)
```
H1 (van Opheusden replication):
  Correlation: r = -0.3 to -0.5, p < 0.05
  Experts: mean h ≈ 1.2 (more h=1)
  Novices: mean h ≈ 3.8 (more h=4)

Rationale: Experts use efficient (shallow) planning
```

#### Results (Fill after analysis)
```
Date: ___________

RQ2 Results:
  Spearman correlation: r = _______, p = _______
  Direction: [Negative / Positive / None]
  Replicates van Opheusden? [YES / NO / UNCLEAR]

RQ4 Results:
  Expert h (mean±std): _______ ± _______
  Novice h (mean±std): _______ ± _______
  Mann-Whitney U: U = _______, p = _______
  Cohen's d: _______

Interpretation:
  _________________________________________
  _________________________________________

Plot saved: figures/expertise_analysis.png
```

#### Decision Gates

**RQ2 Decision**:
```
IF correlation matches van Opheusden direction:
  ✅ RQ2 ANSWER: YES (Replication successful)
  Publication: Strong validation of method

ELIF opposite direction BUT explainable:
  ⚠️  RQ2 ANSWER: PARTIAL
  Need theoretical explanation:
    - PV depth ≠ lookahead depth?
    - Task differences?
  Publication: Interesting divergence

ELSE:
  ❌ RQ2 ANSWER: NO
  Our method doesn't capture same construct
  Publication: Negative result
```

**RQ4 Decision**:
```
IF significant correlation (|r|>0.4, p<0.05):
  ✅ RQ4 ANSWER: YES
  h predicts expertise
  Apply to Pedestrian task

ELIF weak correlation (|r|>0.3, p<0.05):
  ⚠️  RQ4 ANSWER: PARTIAL
  h has some predictive power
  Consider other factors

ELSE:
  ❌ RQ4 ANSWER: NO
  h doesn't predict expertise
  Reconsider individual difference modeling
```

**RQ2 Answer**: ___ (Fill after analysis)

**RQ4 Answer**: ___ (Fill after analysis)

**Overall Rationale**: _________________________________________

---

## 🔬 Final Phase 0 Decision

### Go/No-Go for Pedestrian Application

**Criteria for PROCEED to Pedestrian**:
```
REQUIRED (ALL must be YES):
  ✅ RQ1: h is identifiable (Step 0.2, 0.5)
  ✅ RQ3: Classification acc > 70%

PREFERRED (at least 1 should be YES/PARTIAL):
  ⚠️  RQ2: Replication (YES or PARTIAL acceptable)
  ⚠️  RQ4: Expertise prediction (YES or PARTIAL)
```

**Decision Matrix**:
```
RQ1 + RQ3 + (RQ2 or RQ4) = YES:
  ✅ STRONG GO
  Confidence: High
  Next: Apply full pipeline to Pedestrian
  Publication: Full paper (method + application)

RQ1 + RQ3 = YES, RQ2/RQ4 = NO:
  ⚠️  CONDITIONAL GO
  Confidence: Medium
  Next: Apply to Pedestrian with caution
  Publication: Method paper

RQ1 or RQ3 = NO:
  ❌ NO GO
  Confidence: N/A
  Next: Reconsider approach or task
  Publication: Negative result (valuable!)
```

### Final Decision (Fill after Step 0.7)

**Date**: ___________

**Summary of Results**:
```
RQ1: ___ (YES / PARTIAL / NO)
RQ2: ___ (YES / PARTIAL / NO)
RQ3: ___ (YES / PARTIAL / NO)
RQ4: ___ (YES / PARTIAL / NO)
```

**Decision**: ✅ PROCEED / ⚠️  CONDITIONAL / ❌ STOP

**Rationale**:
```
_________________________________________
_________________________________________
_________________________________________
```

**Next Steps**:
```
IF PROCEED:
  - [ ] Apply to Pedestrian dataset
  - [ ] Write manuscript
  - [ ] Target venue: _______

IF CONDITIONAL:
  - [ ] Limited Pedestrian pilot
  - [ ] Method paper first
  - [ ] Full application later

IF STOP:
  - [ ] Document lessons learned
  - [ ] Negative result paper
  - [ ] Consider alternative approaches
```

---

## 📈 Progress Summary

### Completed Steps
- [x] Step 0.1: Data generation (2025-12-26)
- [ ] Step 0.2: Data validation
- [ ] Step 0.3: Pilot AIRL
- [ ] Step 0.4: Full AIRL
- [ ] Step 0.5: Cross-evaluation
- [ ] Step 0.6: Human data extraction
- [ ] Step 0.7: Human analysis

### Timeline
```
Week 1 (2025-12-23 to 2025-12-29):
  ✅ Step 0.1 (Complete)
  🔄 Step 0.2 (In Progress)

Week 2 (2025-12-30 to 2026-01-05):
  Target: Steps 0.3, 0.4

Week 3-4 (2026-01-06 to 2026-01-19):
  Target: Steps 0.5, 0.6, 0.7

Target Completion: 2026-01-31
```

---

## 📊 Results Dashboard

### Key Metrics at a Glance

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Data Generation** |
| Episodes per h | 100 | 100 | ✅ |
| Episode length | 36 (all) | 36 | ✅ |
| **Behavioral Difference (RQ1a)** |
| KL divergence | >0.1 | ___ | ⏳ |
| JS divergence | >0.1 | ___ | ⏳ |
| **AIRL Training (RQ1b)** |
| h=1 disc acc | >0.7 | ___ | ⏳ |
| h=4 disc acc | >0.7 | ___ | ⏳ |
| **Identifiability (RQ3)** |
| Classification acc | >70% | ___ | ⏳ |
| **Correlation (RQ2/RQ4)** |
| h-Elo correlation | &#124;r&#124;>0.4 | ___ | ⏳ |
| Expert vs Novice | p<0.05 | ___ | ⏳ |

### Traffic Light Status
```
🟢 GREEN: On track, proceeding as expected
🟡 YELLOW: Caution, monitoring closely
🔴 RED: Blocked, intervention needed

Current Status: 🟢 GREEN (Step 0.1 complete, proceeding to 0.2)
```

---

## 📝 Decision Log

### Major Decisions

**Decision 1: Adopt Fixed Horizon Wrapper** (2025-12-26)
```
Context: Variable episode lengths (h=1:17, h=4:26) create confounding
Decision: Implement Pedestrian-style fixed horizon wrapper
Rationale: Remove length-based discrimination, force behavioral learning
Impact: Complete code redesign, but scientifically rigorous
Outcome: ✅ Successfully implemented and tested
```

**Decision 2: Use h={1, 4} instead of {1, 2, 4, 8}** (2025-12-26)
```
Context: Verification showed h=1 vs h=4 has clearest difference (KL=0.16)
         h=8 training unstable, h=2 vs h=4 weak difference
Decision: Focus on h={1, 4} for Phase 0
Rationale: Maximize signal, ensure training stability
Impact: Reduces training time 50%, simplifies analysis
Outcome: ⏳ To be validated in Step 0.2
```

**Decision 3: Large Sample Size (100 episodes per h)** (2025-12-26)
```
Context: Pilot with 5 episodes lacked statistical power
Decision: Generate 100 episodes per h value
Rationale: Achieve power=0.8 for medium effects, enable robust AIRL training
Impact: 3 hours data generation time
Outcome: ✅ Complete
```

*Add more decisions as Phase 0 progresses*

---

## 💭 Lessons Learned

### Technical
- Fixed horizon wrapper is CRITICAL for fair comparison
- Absorbing states must be explicitly flagged in observations
- Small sample sizes (n=5) are insufficient for reliable conclusions
- *Add more as you discover them*

### Methodological
- Pedestrian project provides excellent reference implementation
- P.I. perspective requires explicit decision criteria upfront
- Negative results are publishable if well-documented
- *Add more as you learn*

### Practical
- Data generation for h=4 takes ~3x longer than h=1
- AIRL training may take 4-6 hours per h value
- Human data extraction is complex, allocate sufficient time
- *Add more as you progress*

---

## 📚 References for Decision-Making

1. **Statistical Power**: n=100 gives power=0.85 for d=0.5 (medium effect)
2. **KL Divergence Threshold**: 0.1 is "meaningful difference" (rule of thumb)
3. **Discriminator Accuracy**: >0.7 indicates learning (>0.5=chance, <0.95=not overfitting)
4. **Correlation Thresholds**: Small r=0.3, Medium r=0.5, Large r=0.7 (Cohen, 1988)

---

**Last Updated**: 2025-12-26
**Next Update**: After Step 0.2 completion
**Owner**: [Your Name]

**How to Use This Document**:
1. Update after each step completion
2. Fill in "Results" sections with actual values
3. Make Go/No-Go decisions at each gate
4. Document rationale for all major decisions
5. Use as basis for manuscript methods section
