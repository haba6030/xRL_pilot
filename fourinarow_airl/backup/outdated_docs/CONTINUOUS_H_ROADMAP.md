# Continuous Learnable h Parameter: Roadmap

**Date**: 2025-12-29
**Context**: Post-BREAKTHROUGH (KL=0.1049 achieved with discrete h=1,4)
**Goal**: Extend to continuous h for psychological computational modeling

---

## Background

### Current Implementation (Discrete h)
```
h ∈ {1, 2, 3, 4, 5}  # Discrete set
Separate models: model_h1, model_h2, ..., model_h5
Individual fitting: h_hat = argmax_{h} LL(data | h, β, lapse)
Result: h_hat ∈ {1, 2, 3, 4, 5}
```

**Success**: KL(h=1 || h=4) = 0.1049 ✅

### Proposed Extension (Continuous h)
```
h ∈ [1.0, 5.0]  # Continuous range
Individual fitting: (h_hat, β_hat, lapse_hat) = argmax LL(data | h, β, lapse)
Result: h_hat ∈ ℝ (e.g., 2.73, 3.91)
```

**Motivation**:
- Standard in cognitive modeling (DDM, POMDP)
- Fine-grained individual differences
- Hierarchical modeling: h ~ Normal(μ_group, σ)
- Clinical correlations: continuous h vs anxiety score

---

## Core Ideas

### Idea 1: Continuous h as Cognitive Parameter

**Psychological interpretation**:
- h represents **effective planning depth** (not discrete steps)
- Novices: h ≈ 1.2-2.5 (myopic planning)
- Experts: h ≈ 3.7-4.8 (far-sighted planning)
- Clinical: h as marker of planning capacity

**Advantages**:
- Captures graded individual differences
- Enables correlation analyses (h vs Elo, h vs anxiety)
- Standard for parameter recovery studies

### Idea 2: Interpolation Between Discrete Models

**Key insight**: Don't retrain - reuse existing models!
```python
# h=2.3 means "30% toward h=3 from h=2"
prob_h2.3 = 0.7 * prob_h2 + 0.3 * prob_h3
```

**Assumptions**:
- Behavior changes smoothly between discrete h
- Linear interpolation approximates intermediate planning depths
- Requires validation: does interpolated h=2.5 match true h=2.5 agent?

### Idea 3: Joint Optimization with β, lapse

**Individual-level fitting**:
```python
(h*, β*, lapse*) = argmax_{h,β,lapse} Σ_t log π(a_t | s_t; h, β, lapse)
```

**Advantages**:
- h is on equal footing with other cognitive parameters
- Gradient-based optimization (L-BFGS-B)
- Enables parameter recovery: generate with h=2.7 → recover h≈2.7

---

## Implementation Options

### Option A: Interpolation-Based (Simple)

**Architecture**:
```python
class InterpolatedDepthPolicy:
    """
    Uses pre-trained discrete h models (h=1,2,3,4,5)
    Interpolates for continuous h values
    """
    def __init__(self, model_dict):
        self.models = model_dict  # {1: model_h1, 2: model_h2, ...}

    def predict_proba(self, state, h_continuous):
        """
        h=2.3 → 0.7*model_h2 + 0.3*model_h3
        """
        h_low = floor(h_continuous)
        h_high = ceil(h_continuous)
        weight = h_continuous - h_low

        if h_low == h_high:
            return self.models[h_low].predict_proba(state)

        prob_low = self.models[h_low].predict_proba(state)
        prob_high = self.models[h_high].predict_proba(state)

        # Linear interpolation
        prob = (1 - weight) * prob_low + weight * prob_high
        return prob / prob.sum()  # Renormalize
```

**Pros**:
- ✅ Reuses existing separate models
- ✅ Maintains KL=0.1049 at h=1,4
- ✅ Simple implementation (1 day)
- ✅ No retraining needed

**Cons**:
- ⚠️ Linear interpolation assumption (not validated)
- ⚠️ Still need 5 discrete models
- ⚠️ Meaning of h=2.73 unclear

**Use case**: Quick pilot for individual fitting

---

### Option B: h-Conditional Neural Network (Joint Model)

**Architecture**:
```python
class HConditionalPolicy(nn.Module):
    """
    Single model that takes h as input
    Like Mhammedi(2023) joint approach
    """
    def __init__(self):
        self.h_embed = nn.Embedding(100, 16)  # h discretized to 100 bins
        self.state_encoder = nn.Linear(89, 128)
        self.joint = nn.Sequential(
            nn.Linear(128 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36)  # action logits
        )

    def forward(self, state, h_value):
        h_idx = int((h_value - 1.0) / 4.0 * 99)  # [1,5] → [0,99]
        h_emb = self.h_embed(h_idx)
        state_feat = self.state_encoder(state)
        combined = torch.cat([state_feat, h_emb], dim=-1)
        return self.joint(combined)
```

**Training**:
```python
# Joint training on all h data
for batch in dataloader:
    states, h_values, actions = batch
    logits = model(states, h_values)
    loss = nn.CrossEntropyLoss()(logits, actions)
    loss.backward()
```

**Pros**:
- ✅ Single model (efficient)
- ✅ True continuous h (no interpolation)
- ✅ Gradient-based optimization
- ✅ Scalable to many h values

**Cons**:
- ❌ **H-INTERFERENCE PROBLEM** (same as train_multistep_ik_sklearn.py failure)
- ❌ KL divergence likely to decrease (0.1049 → 0.04?)
- ❌ Requires retraining from scratch
- ❌ Already failed in previous experiments

**Use case**: If h-interference can be solved (e.g., larger model, better h encoding)

---

### Option C: Hybrid (Discrete + Continuous Interpolation)

**Strategy**:
1. Keep separate discrete models (h=1,2,3,4,5)
2. Use continuous h only at individual fitting stage
3. Interpolate between models for likelihood computation

**Architecture**:
```python
# Training: Same as current (separate models)
model_h1.fit(data_h1)  # h=1 data only
model_h2.fit(data_h2)  # h=2 data only
...

# Individual fitting: Continuous h
def fit_participant(participant_data):
    """
    Optimize continuous h per participant
    """
    def neg_loglik(params):
        h, beta, lapse = params

        ll = 0
        for traj in participant_data:
            for t in range(len(traj)):
                # Interpolate models at continuous h
                probs = interpolate_models(traj[t]['state'], h)
                probs = apply_softmax_lapse(probs, beta, lapse)
                ll += np.log(probs[traj[t]['action']])

        return -ll

    result = minimize(
        neg_loglik,
        x0=[2.5, 1.0, 0.1],  # h, beta, lapse
        bounds=[(1.0, 5.0), (0.1, 10.0), (0.0, 0.5)],
        method='L-BFGS-B'
    )

    return {
        'h': result.x[0],      # Continuous h_hat
        'beta': result.x[1],
        'lapse': result.x[2]
    }
```

**Pros**:
- ✅ Keeps successful separate models
- ✅ Maintains KL=0.1049 for h=1,4
- ✅ Continuous h for individual differences
- ✅ Best of both worlds

**Cons**:
- ⚠️ Interpolation validity still unclear
- ⚠️ 5 models needed

**Use case**: Recommended for near-term implementation

---

## Validation Strategy

### Test 1: Parameter Recovery

**Goal**: Check if continuous h is identifiable

```python
# Generate synthetic data with known h
true_h = 2.7
true_beta = 1.5
true_lapse = 0.1

synthetic_data = generate_trajectories(
    policy=InterpolatedDepthPolicy(models),
    h=true_h,
    beta=true_beta,
    lapse=true_lapse,
    n_episodes=100
)

# Fit and recover
recovered = fit_participant(synthetic_data)

# Check recovery
print(f"True h: {true_h:.2f}, Recovered: {recovered['h']:.2f}")
print(f"True β: {true_beta:.2f}, Recovered: {recovered['beta']:.2f}")

# Success criterion: |recovered_h - true_h| < 0.5
```

**Expected outcome**:
- If recovery works → continuous h is identifiable
- If not → may need discrete h only

---

### Test 2: Interpolation Validity

**Goal**: Check if interpolated h=2.5 matches true h=2.5

```python
# Option 1: If we train h=2.5 model separately
model_h2p5_true = train_on_h2p5_data()
prob_true = model_h2p5_true.predict_proba(test_states)

# Option 2: Interpolate between h=2 and h=3
prob_interp = 0.5 * model_h2.predict_proba(test_states) + \
              0.5 * model_h3.predict_proba(test_states)

# Compare
kl = KL_divergence(prob_true, prob_interp)
print(f"KL(true h=2.5 || interpolated h=2.5) = {kl:.4f}")

# Success criterion: KL < 0.05 (close match)
```

**If fails**: Linear interpolation invalid → need nonlinear or abandon interpolation

---

### Test 3: Individual Differences

**Goal**: Check if continuous h captures meaningful variation

```python
# Fit multiple participants
results = []
for participant_id in participant_ids:
    data = load_participant_data(participant_id)
    params = fit_participant(data)
    results.append({
        'id': participant_id,
        'h': params['h'],
        'beta': params['beta'],
        'Elo': get_elo_rating(participant_id)
    })

# Analyze
df = pd.DataFrame(results)
correlation = df['h'].corr(df['Elo'])

print(f"Correlation(h, Elo) = {correlation:.3f}")

# Hypothesis: Experts have higher h
# Success criterion: correlation > 0.3
```

---

## Phased Roadmap

### Phase 0: Current Status (DONE ✅)
**Status**: Discrete h=1,4 with separate models
**Achievement**: KL=0.1049
**Files**:
- `train_separate_h_models.py`
- `generate_trajectories_separate_h.py`
- `compare_separate_h_distributions.py`

---

### Phase 1: AIRL Discriminator (PRIORITY - Next 1-2 weeks)
**Goal**: Validate planning-aware AIRL framework
**Tasks**:
1. ✅ Train h=1 and h=4 policies (DONE)
2. ⏳ Implement AIRL discriminator with depth-specific rewards
3. ⏳ Test reward identifiability
4. ⏳ Check: Can discriminator distinguish h=1 vs h=4?

**Files to create**:
- `train_airl_depth_aware.py`
- `evaluate_reward_identifiability.py`

**Success criterion**: Discriminator accuracy > 80% on h=1 vs h=4

**Why first**: Validate core framework before adding complexity

---

### Phase 2: Expand Discrete h (2-3 weeks)
**Goal**: Train models for h=2,3,5
**Tasks**:
1. Preprocess h=2,3,5 data (extend `preprocess_multistep_ik_data.py`)
2. Train separate models for each h
3. Generate trajectories for all h
4. Measure KL divergence matrix (all pairs)

**Expected output**:
```
KL Divergence Matrix:
     h=1   h=2   h=3   h=4   h=5
h=1  0.00  0.05  0.07  0.10  0.12
h=2  0.05  0.00  0.03  0.06  0.08
h=3  0.07  0.03  0.00  0.04  0.06
h=4  0.10  0.06  0.04  0.00  0.03
h=5  0.12  0.08  0.06  0.03  0.00
```

**Success criterion**: KL increases with |h_i - h_j|

---

### Phase 3: Continuous h Implementation (1 week)
**Goal**: Implement Option C (Hybrid approach)
**Tasks**:

#### Task 3.1: Interpolation Policy Class
```python
# File: interpolated_depth_policy.py

class InterpolatedDepthPolicy:
    """
    Policy that interpolates between discrete h models
    Supports continuous h ∈ [1.0, 5.0]
    """
    def __init__(self, model_paths):
        self.models = {}
        for h, path in model_paths.items():
            self.models[h] = joblib.load(path)

    def predict_proba(self, state, h_continuous, rollout_env=None):
        """
        Get action probabilities for continuous h
        """
        # Implementation from Option A/C
        pass
```

**Files**: `interpolated_depth_policy.py`

#### Task 3.2: Individual Fitting Pipeline
```python
# File: fit_continuous_h.py

def fit_participant_continuous(participant_data, model_dict):
    """
    Fit continuous h, beta, lapse to participant data
    """
    # Implementation from Option C
    pass

def fit_all_participants(dataset):
    """
    Fit all participants and return DataFrame
    """
    results = []
    for pid in dataset.participant_ids:
        params = fit_participant_continuous(
            dataset.get_participant(pid),
            model_dict
        )
        results.append({'id': pid, **params})
    return pd.DataFrame(results)
```

**Files**: `fit_continuous_h.py`

#### Task 3.3: Validation Scripts
```python
# File: validate_continuous_h.py

def test_parameter_recovery():
    """Test 1: Parameter recovery"""
    pass

def test_interpolation_validity():
    """Test 2: Interpolation validity"""
    pass

def test_individual_differences():
    """Test 3: Individual differences"""
    pass

if __name__ == '__main__':
    print("Running continuous h validation...")
    test_parameter_recovery()
    test_interpolation_validity()
    test_individual_differences()
```

**Files**: `validate_continuous_h.py`

**Deliverables**:
- Interpolation policy class
- Individual fitting pipeline
- Validation report

---

### Phase 4: Expertise Discrimination (2-3 weeks)
**Goal**: Test if h discriminates novice vs expert
**Tasks**:
1. Label participants as novice/expert (Elo threshold or percentile)
2. Fit continuous h for all participants
3. Train classifier: h → novice/expert
4. Compute ROC-AUC, classification accuracy

**Analysis**:
```python
# File: expertise_discrimination.py

# Fit all participants
participant_params = fit_all_participants(dataset)

# Label novice/expert (e.g., Elo < median = novice)
median_elo = participant_params['Elo'].median()
participant_params['expertise'] = (
    participant_params['Elo'] > median_elo
).astype(int)

# Logistic regression: h → expertise
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = participant_params[['h', 'beta', 'lapse']]
y = participant_params['expertise']

clf = LogisticRegression()
clf.fit(X, y)
y_pred = clf.predict_proba(X)[:, 1]

auc = roc_auc_score(y, y_pred)
print(f"ROC-AUC (h+β+lapse → expertise): {auc:.3f}")

# Test h alone
X_h = participant_params[['h']]
clf_h = LogisticRegression()
clf_h.fit(X_h, y)
y_pred_h = clf_h.predict_proba(X_h)[:, 1]
auc_h = roc_auc_score(y, y_pred_h)
print(f"ROC-AUC (h only → expertise): {auc_h:.3f}")
```

**Success criterion**: AUC(h → expertise) > 0.7

---

### Phase 5: Hierarchical Bayesian Model (Optional, 3-4 weeks)
**Goal**: Full Bayesian treatment of continuous h
**Model**:
```
Group level:
  μ_h ~ Normal(3, 1)
  σ_h ~ HalfNormal(1)

Individual level:
  h_i ~ Normal(μ_h, σ_h)
  β_i ~ LogNormal(0, 1)
  lapse_i ~ Beta(2, 18)  # Prior favoring low lapse

Likelihood:
  a_t ~ Categorical(π(·|s_t; h_i, β_i, lapse_i))
```

**Implementation**: PyMC3 or Stan

**Files**: `hierarchical_h_model.py`

**Deliverables**:
- Posterior distributions for h_i
- Group-level μ_h, σ_h estimates
- Expertise comparison: μ_h(expert) vs μ_h(novice)

---

### Phase 6: Clinical Extension (Future)
**Goal**: Link h to clinical traits
**Requirements**:
- Collect anxiety/impulsivity scores
- Fit continuous h to clinical population
- Test correlations: h vs anxiety, h vs impulsivity

**Hypotheses**:
- H1: Anxiety → lower h (reduced planning)
- H2: Impulsivity → lower h (myopic choices)
- H3: h mediates anxiety → behavior link

**Analysis**: Structural equation modeling (SEM)

---

## Implementation Priority

### Immediate (This Week)
1. ⏳ **AIRL discriminator** (Step 0.3 - STEP03_AIRL_DISCRIMINATOR.md)
   - Validate planning-aware AIRL works
   - Required before continuous h

### Short-term (Next 2-4 Weeks)
2. ⏳ **Expand to h=2,3,5** (Phase 2)
   - Get complete discrete h set
   - Measure KL divergence matrix

3. ⏳ **Implement continuous h** (Phase 3)
   - Interpolation policy
   - Individual fitting pipeline
   - Validation tests

### Mid-term (1-2 Months)
4. ⏳ **Expertise discrimination** (Phase 4)
   - Primary research question
   - Paper-worthy result

5. ⏳ **Parameter recovery study**
   - Validate identifiability
   - Standard for cognitive modeling papers

### Long-term (2-3 Months)
6. ⏳ **Hierarchical Bayesian model** (Phase 5)
   - Full Bayesian treatment
   - Group-level inference

7. ⏳ **Clinical extension** (Phase 6)
   - Requires new data collection
   - Separate study

---

## Key Design Decisions

### Decision 1: Interpolation vs Joint Model

**Chosen**: Interpolation (Option C)

**Rationale**:
- Preserves successful separate models (KL=0.1049)
- Avoids h-interference problem
- Quick implementation
- Can switch to joint model later if needed

**Risk**: Interpolation validity not guaranteed
**Mitigation**: Validation Test 2 (interpolation validity check)

---

### Decision 2: Linear vs Nonlinear Interpolation

**Chosen**: Linear (for now)

**Rationale**:
- Simple, interpretable
- Common in cognitive modeling
- Easy to implement

**Alternative**: If linear fails validation
```python
# Softmax interpolation (nonlinear)
def softmax_interpolate(prob_low, prob_high, h_continuous):
    h_frac = h_continuous - floor(h_continuous)
    logits_low = np.log(prob_low + 1e-10)
    logits_high = np.log(prob_high + 1e-10)
    logits = (1 - h_frac) * logits_low + h_frac * logits_high
    return softmax(logits)
```

---

### Decision 3: Number of Discrete h Values

**Chosen**: 5 values (h=1,2,3,4,5)

**Rationale**:
- Sufficient coverage [1, 5]
- Manageable training cost (5 models)
- Interpolation resolution: 0.25 effective granularity

**Alternative**: If finer granularity needed
- Add h=1.5, 2.5, 3.5, 4.5 → 9 models
- But increases training cost 1.8x

---

## Expected Outcomes

### Quantitative Metrics

| Metric | Current (Discrete) | Target (Continuous) |
|--------|-------------------|-------------------|
| **h granularity** | 5 levels | Infinite |
| **h_hat precision** | ±0.5 (quantized) | ±0.1 (continuous) |
| **Parameter recovery** | N/A | |h_recovered - h_true| < 0.3 |
| **Expertise AUC** | TBD | > 0.7 |
| **Model count** | 5 separate | 5 separate + interpolation |
| **Fit time per participant** | ~5 min | ~10 min (optimization) |

### Qualitative Benefits

1. **Standard cognitive modeling**
   - Continuous parameters (like DDM drift rate, POMDP discount)
   - Enables hierarchical Bayesian models
   - Parameter recovery studies

2. **Fine-grained individual differences**
   - Novice A: h=1.2 (very myopic)
   - Novice B: h=1.8 (slightly better)
   - Expert A: h=3.9 (good planning)
   - Expert B: h=4.7 (exceptional planning)

3. **Clinical applications**
   - Continuous correlation: h vs anxiety score
   - Mediation analysis: anxiety → h → behavior
   - Treatment effects: pre-therapy h vs post-therapy h

4. **Theory development**
   - Test smooth vs discrete planning theory
   - Expertise as continuous dimension
   - Planning capacity as individual difference

---

## Risks and Mitigations

### Risk 1: Interpolation Invalid
**Symptom**: Interpolated h=2.5 doesn't match true h=2.5
**Impact**: Continuous h meaningless
**Mitigation**:
- Run validation Test 2 early
- If fails, use discrete h only
- Or train more discrete models (h=1.5, 2.5, ...)

### Risk 2: h Not Identifiable
**Symptom**: Parameter recovery fails (|recovered - true| > 1.0)
**Impact**: Can't trust individual h estimates
**Mitigation**:
- Collect more data per participant
- Simplify model (fix β, lapse to constants)
- Use hierarchical priors to stabilize estimates

### Risk 3: No Expertise Correlation
**Symptom**: Corr(h, Elo) ≈ 0
**Impact**: h doesn't predict expertise
**Mitigation**:
- May still predict other traits (anxiety, impulsivity)
- Interaction effects: h × β predicts expertise
- Nonlinear relationships (U-shaped?)

### Risk 4: Computational Cost
**Symptom**: Individual fitting takes too long (>30 min/participant)
**Impact**: Can't scale to large datasets
**Mitigation**:
- Parallelize across participants
- Use better optimization (quasi-Newton methods)
- Pre-compute rollouts and cache

---

## Success Criteria

### Phase 1 (AIRL): ✅ if
- Discriminator accuracy > 80% (h=1 vs h=4)
- Learned reward generalizes to OOD states
- Reward identifiability demonstrated

### Phase 3 (Continuous h): ✅ if
- Parameter recovery: |recovered - true| < 0.5 for h
- Interpolation validity: KL(true h=2.5 || interp h=2.5) < 0.1
- Individual h estimates vary meaningfully (std > 0.5)

### Phase 4 (Expertise): ✅ if
- ROC-AUC(h → expertise) > 0.7
- Correlation(h, Elo) > 0.3, p < 0.05
- Experts have significantly higher h (t-test p < 0.01)

### Overall Project: ✅ if
- Planning-aware AIRL paper accepted
- h discriminates expertise
- Method applied to ≥2 tasks (4-in-a-row + another)

---

## File Structure (After Full Implementation)

```
fourinarow_airl/
├── data/
│   ├── multistep_ik/
│   │   ├── ik_pairs_h1.pkl
│   │   ├── ik_pairs_h2.pkl
│   │   ├── ik_pairs_h3.pkl
│   │   ├── ik_pairs_h4.pkl
│   │   └── ik_pairs_h5.pkl
│   ├── separate_h_trajectories/
│   │   ├── h1_trajectories.pkl
│   │   ├── h2_trajectories.pkl
│   │   ├── h3_trajectories.pkl
│   │   ├── h4_trajectories.pkl
│   │   └── h5_trajectories.pkl
│   └── participant_fits/
│       └── continuous_h_fits.csv
│
├── models/
│   ├── separate_h/
│   │   ├── model_h1.pkl
│   │   ├── model_h2.pkl
│   │   ├── model_h3.pkl
│   │   ├── model_h4.pkl
│   │   └── model_h5.pkl
│   └── airl/
│       ├── discriminator_h1_vs_h4.pth
│       └── reward_network.pth
│
├── Core implementation (DONE ✅)
├── preprocess_multistep_ik_data.py
├── train_separate_h_models.py
├── generate_trajectories_separate_h.py
├── compare_separate_h_distributions.py
│
├── Phase 1: AIRL (IN PROGRESS ⏳)
├── pilot_airl_discriminator.py
├── train_airl_depth_aware.py
├── evaluate_reward_identifiability.py
│
├── Phase 3: Continuous h (TODO 📝)
├── interpolated_depth_policy.py          # NEW
├── fit_continuous_h.py                    # NEW
├── validate_continuous_h.py               # NEW
│
├── Phase 4: Expertise (TODO 📝)
├── expertise_discrimination.py            # NEW
├── parameter_recovery_study.py            # NEW
│
├── Phase 5: Hierarchical (FUTURE 🔮)
├── hierarchical_h_model.py                # NEW
│
├── Documentation
├── README.md
├── BREAKTHROUGH_SUMMARY.md
├── MHAMMEDI_COMPARISON.md
├── CONTINUOUS_H_ROADMAP.md               # THIS FILE
└── STEP03_AIRL_DISCRIMINATOR.md
```

---

## Timeline Estimate

```
Week 1-2:   AIRL discriminator (Phase 1) - PRIORITY
Week 3-4:   Expand discrete h (Phase 2)
Week 5:     Implement continuous h (Phase 3)
Week 6-7:   Expertise discrimination (Phase 4)
Week 8:     Parameter recovery study
Week 9-10:  Hierarchical Bayesian model (Phase 5) - OPTIONAL
Week 11-12: Paper writing
```

**Total**: ~3 months to continuous h expertise paper

---

## Key References

### Continuous h in Cognitive Modeling
- **Ratcliff & McKoon (2008)**: Diffusion Decision Model with continuous drift rate
- **Daw et al. (2011)**: Model-based/model-free trade-off (continuous weight)
- **Collins & Frank (2012)**: Working memory capacity as continuous parameter

### Multi-Step Inverse Kinematics
- **Mhammedi et al. (2023)**: RL from Passive Data via Latent Intentions
- **Yao et al. (2024)**: IRL and Planning (horizon as confounder)

### Planning Depth
- **Van Opheusden et al. (2023)**: Expertise via planning depth (4-in-a-row)
- **Huys et al. (2012)**: Planning depth in depression (sequential decision task)

---

## Summary

### Current Status
- ✅ Discrete h=1,4 working (KL=0.1049)
- ✅ Separate models successful
- ⏳ AIRL in progress

### Next Steps (Recommended Order)
1. **Week 1-2**: AIRL discriminator (validate framework)
2. **Week 3-4**: Expand to h=2,3,5 (complete discrete set)
3. **Week 5**: Continuous h implementation (interpolation)
4. **Week 6-7**: Expertise discrimination (main result)

### Implementation Strategy
- **Chosen approach**: Option C (Hybrid - discrete models + continuous fitting)
- **Rationale**: Preserves KL=0.1049, avoids h-interference, quick implementation
- **Validation**: 3 tests (recovery, interpolation, individual differences)

### Expected Impact
- **Methodological**: Planning-aware IRL with continuous h
- **Empirical**: h discriminates expertise (ROC-AUC > 0.7 expected)
- **Theoretical**: Planning depth as identifiable individual difference
- **Clinical**: Potential link to anxiety, impulsivity (future work)

### Risk Management
- Main risk: Interpolation validity
- Mitigation: Early validation, fallback to discrete h
- Contingency: Train more discrete models if needed

---

**Status**: Ready for Phase 1 (AIRL) ✅
**Timeline**: ~3 months to paper-ready results
**Confidence**: High (building on successful foundation)

---

End of Roadmap
