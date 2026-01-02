# Methodological Comparison: Our Approach vs. Mhammedi et al. (2023) MusIK

Direct comparison of our future-state-conditioned inverse modeling approach with the multi-step inverse kinematics (MusIK) framework.

---

## MusIK Algorithm (Mhammedi et al. 2023)

### Problem setting

Block MDP with high-dimensional observations x (no direct access to latent state z). Goal: Learn representation ϕ: x → z that enables efficient RL.

### Algorithm structure

```
Initialization: Ψ(1) = ∅

For h = 2 to H:
    # IKDP(h) - Inverse Kinematics Dynamic Programming
    For t = h-1 down to 1:
        D_t = ∅

        Repeat n times:
            # 1. Roll-in: Sample from Ψ(t) to reach timestep t
            π_rollin ∈ Ψ(t)
            Execute π_rollin → reach time t, observe x_t

            # 2. Random action at t
            a_t ~ Uniform(A)

            # 3. Roll-out: Use previously learned policies for t+1 to h
            if t < h-1:
                i ~ S_h  # Sample suffix policy index
                Execute π̂(i, t+1) → reach time h, observe x_h

            # 4. Collect transition
            Add (x_t, x_h, a_t, i) to D_t

        # 5. Learn inverse model: (x_t, x_h) → (a_t, i)
        Train f_t to predict both action AND next policy index

        # 6. Construct non-Markovian policy
        For each i ∈ S_h:
            Define π̂(i, t):
                At time t: (a, j) = f_t(x_t, x_h)
                At time t+1: Execute π̂(j, t+1)

    # Output: Policy cover at depth h
    Ψ(h) = { π̂(i, 1) | i ∈ S_h }
```

### Key properties

**Layer-by-layer construction**:
- Builds Ψ(2), then Ψ(3), ..., up to Ψ(H)
- Each layer uses previous layers as building blocks

**Backward in time**:
- For each h, constructs policies from t=h-1 down to t=1
- Uses roll-out from already-learned future policies

**Non-Markovian stitching**:
- Inverse model predicts (action, suffix_index)
- Policies route to sub-policies, not just actions
- Handles permutation/aliasing in Block MDPs

**Output**:
- Policy cover Ψ(H): set of policies that collectively reach all states achievable in H steps
- Not a single policy, but a library of exploration policies

---

## Our Approach

### Problem setting

Fully observable board game with complete state information. Goal: Estimate planning depth h from human behavioral data.

### Algorithm structure

```
Training phase:

For h in [1, 2, 3, 4]:
    # Extract h-specific training data
    For each game:
        For each timestep t where t+h < game_length:
            Extract (s_t, s_{t+h}, a_t)
            # s_{t+h} is ACTUAL game continuation (not simulated)

    # Train independent model for this h
    model_h = MLPClassifier()
    X_h = concat(s_t, s_{t+h})  # 178-dim
    y_h = a_t  # 36-dim (action space)

    model_h.fit(X_h, y_h)

Inference phase:

For each player:
    For each move (s_t, a_t):
        # Get actual game continuations
        Observe s_{t+1}, s_{t+2}, s_{t+3}, s_{t+4} from game record

        # Compute likelihoods under each h-model
        For h in [1, 2, 3, 4]:
            ℓ_h = model_h.predict_proba([s_t, s_{t+h}])[a_t]

        # Bayesian posterior
        P(h | move) = ℓ_h / Σ_h' ℓ_h'

    # Aggregate over player's moves
    E[h]_player = Σ_moves P(h | move) × h
```

### Key properties

**Independent training**:
- Each h has separate model (no shared encoder)
- No layer-by-layer construction

**Forward in time**:
- Use actual game continuations (no roll-in/roll-out)
- Retrospective analysis (requires complete game records)

**Markovian models**:
- Predict action only (no policy routing)
- Simple concatenation of states

**Output**:
- Per-player planning depth estimate E[h]
- Not policy cover, just depth estimate

---

## Direct Comparison

| Aspect | MusIK | Our Method |
|--------|-------|------------|
| **Goal** | Representation learning | Planning depth estimation |
| **Environment** | Block MDP (latent state) | Fully observable game |
| **Output** | Policy cover Ψ(H) | Planning depth estimates |
| **Encoder** | Shared across h (learns ϕ) | Separate models per h |
| **Construction** | Layer-by-layer (h=2→H) | Independent per h |
| **Time direction** | Backward (t=h-1→1) | Forward (use actual data) |
| **Data collection** | Roll-in + random + roll-out | Actual game records |
| **Inverse model** | (x_t, x_h) → (a, suffix_index) | (s_t, s_{t+h}) → a |
| **Policy type** | Non-Markovian (routed) | Markovian (direct action) |
| **Complexity** | High (exploration algorithm) | Low (supervised learning) |

---

## What We Borrowed from MusIK

**Conceptual level**:
1. Multi-step inverse modeling idea: Using (state_t, state_{t+h}) as input
2. Planning depth as explicit factor: Different h captures different temporal structure
3. Inverse model perspective: Backward inference from transitions

**NOT borrowed**:
1. IKDP algorithm (layer-by-layer construction)
2. Roll-in/roll-out mechanism
3. Non-Markovian policy stitching
4. Policy cover objective
5. Representation learning
6. Exploration focus

**Honest characterization**: We use "multi-step inverse modeling" inspired by MusIK's conceptual framework, but our implementation is much simpler (supervised learning on human data) rather than MusIK's complex exploration algorithm.

---

## Core Methodological Assumptions

### Assumption 1: Planning = Forward simulation with discrete horizon

If person plans with depth h, they mentally simulate:
```
s_t → s_{t+1} → s_{t+2} → ... → s_{t+h}
```
and choose a_t based on this h-step simulation.

**Problems**:
- Assumes discrete h ∈ {1,2,3,4} (reality may be continuous or mixed)
- Assumes explicit forward simulation (may use heuristics without simulation)
- Assumes fixed h per decision (may adapt within single choice)

**Evidence from data**: Posterior P(h | move) often distributes across multiple h values (not winner-take-all), suggesting mixed or continuous depths.

### Assumption 2: Concatenation captures planning relationship

Model architecture:
```python
X = concat(s_t, s_{t+h})  # Naïve concatenation
hidden = MLP(X)
logits = output_layer(hidden)
```

**Problems**:
- No explicit temporal structure
- No attention mechanism to weight features by horizon
- Linear combinations may not capture complex planning interactions
- Network must discover temporal relationships implicitly

**Alternative architectures** (not implemented):
- Separate encoders: z_t = enc(s_t), z_h = enc(s_{t+h}), then combine
- Trajectory encoding: model full sequence s_t → s_{t+1} → ... → s_{t+h}
- Attention over timesteps: weight intermediate states

**Evidence**: Despite simple architecture, discriminator achieves 93.8% accuracy (h=1 vs h=4), suggesting some signal is captured.

### Assumption 3: Statistical association reflects planning

We measure: P(a_t | s_t, s_{t+h}) highest for some h

We infer: Person planned with depth h

**Problems**:
- Correlation ≠ causation
- Cannot distinguish:
  - Forward planning: "I simulate h steps to choose action"
  - Pattern recognition: "I recognize h-step patterns heuristically"
  - Task constraint: "h-step information is decision-relevant regardless of simulation"

**Example ambiguity**:
```
Situation: Opponent has 3-in-a-row, one empty space
Action: Block the threat (h=1 has highest likelihood)

Interpretation A (planning):
  Player simulates 1 step: "If I don't block, opponent wins"
  → Explicit forward simulation with h=1

Interpretation B (heuristic):
  Player recognizes threat pattern: "3-in-a-row = must block"
  → No simulation, immediate response

Our method: Cannot distinguish these cases
```

---

## Two-Player Confound (Critical Issue)

In two-player games, future state s_{t+h} is not determined by player's action a_t alone:

```
s_{t+h} = f(s_t, a_t, a_t^opp, a_{t+1}, a_{t+1}^opp, ..., a_{t+h-1}^opp)
                   ↑      ↑       ↑         ↑              ↑
                 player opponent player  opponent    opponent
```

**Consequence**: "Action to reach s_{t+h}" is ill-defined.

The same player action a_t can lead to completely different states s_{t+h} depending on opponent responses:

```
Example:
s_t = "Player places piece at center"
a_t = position 18 (center square)

Possible s_{t+4} outcomes:
- Opponent plays randomly → s_{t+4} favors player
- Opponent blocks strategically → s_{t+4} neutral
- Opponent counter-attacks → s_{t+4} threatens player

Same a_t, different s_{t+4}!
```

**What we actually measure**: Statistical dependency between (a_t, s_{t+h}) conditional on s_t, where s_{t+h} reflects both players' actions.

**Assumption**: This dependency structure varies systematically with planning depth h despite opponent confounding.

**Partial justification**: Human opponents behave systematically (not random), so s_{t+h} distributions do carry information about player's a_t. But this is weaker than single-agent setting.

---

## Conservative Interpretation: Decision-Relevant Temporal Horizon

More defensible claim: We measure which temporal horizon's state information best predicts observed actions.

**What this means**:
- h=1 wins → s_{t+1} is most informative for predicting a_t
- h=2 wins → s_{t+2} is most informative for predicting a_t

**This could reflect**:
1. Actual planning: Person simulated h steps ahead
2. Task constraint: Game dynamics make h-step information decision-relevant
3. State correlation: s_{t+h} correlates with features that determine a_t
4. Heuristic pattern: h-step temporal patterns trigger specific responses

**We cannot distinguish these from behavior alone**.

**Advantage**: This interpretation makes fewer assumptions about cognitive mechanisms while preserving the empirical findings.

---

## State-Dependence Hypothesis

If planning depth varies by game context rather than individual differences, our method may be detecting task constraints rather than cognitive processes:

**Threatening position**:
```
Board state: Opponent has 3-in-a-row with one empty square
h=1 model: highest likelihood for blocking action

Possible explanations:
A. Player simulates 1 step: "If I don't block, I lose"
B. Immediate future is highly constrained (forced move)
   → Only s_{t+1} is decision-relevant
C. Threat-recognition heuristic: "Block 3-in-a-row" (no simulation)

All produce same statistical signature: P(a | s_t, s_{t+1}) > P(a | s_t, s_{t+4})
```

**Calm position**:
```
Board state: No immediate threats, multiple strategic options
h=2 or h=3 model: highest likelihood

Possible explanations:
A. Player simulates 2-3 steps to evaluate options
B. Medium-term information discriminates between strategic choices
   → s_{t+2} or s_{t+3} is decision-relevant
C. Pattern recognition over 2-3 step temporal contexts

Again, same statistical signature
```

**Reframing**: h varies by game context because different situations make different time scales decision-relevant, not necessarily because people adaptively change simulation depth.

---

## What Remains Robust

Despite methodological limitations, empirical findings hold:

### 1. Decision-relevant horizon is identifiable

Discriminator accuracy: 93.8% (h=1 vs h=4)

Models trained on h=1 data behave systematically differently from h=4 models. Bayesian posterior P(h | move) varies across moves. This is a real signal, not noise.

### 2. No correlation with expertise

```
                    Correlation with Elo
Rollout-free:       r = -0.01, p = 0.94
Random rollout:     r = +0.03, p = 0.85
Opponent rollout:   r = -0.02, p = 0.88

Group comparison (tertile split):
Experts:  E[h] = 1.77
Novices:  E[h] = 1.77
F(2,37) = 0.02, p = 0.98
```

Null result is consistent across methods. Not an artifact of one particular estimation approach.

### 3. Rollout simulation creates measurable bias

```
Method              E[h]    Bias
Rollout-free:       1.78    —
Opponent rollout:   2.62    +0.84 (+47%)
Random rollout:     2.87    +1.09 (+61%)
```

Systematic bias is explainable: Random futures are more diverse than human futures, benefiting longer-horizon models disproportionately. Method sensitivity to future state distribution is confirmed.

### 4. Alignment with van Opheusden et al.

```
van Opheusden PV depth (tree exploration): 6-7 steps
Our decision horizon:                      1.78 steps

Interpretation: Wide search (exploration), narrow decision scope (relevance)
```

Players explore 6-7 steps to verify choices, but only ~2 steps are decision-relevant. Consistent with distinction between search breadth and decision horizon.

---

## Implications for Different Audiences

### For IRL applications

Conservative interpretation is actually more useful.

Whether we measure "planning depth" or "decision-relevant temporal horizon," the key points remain:
- h is identifiable from behavior
- h confounds reward inference if not modeled
- h does NOT predict expertise

Practical guidance:
```
Recommended: Model r(s, a | h) - condition rewards on estimated h
Do NOT:      Interpret h as skill marker - use heuristic features instead
```

### For cognitive science

Acknowledge fundamental ambiguity: Cannot distinguish forward simulation from heuristic pattern recognition.

Frame state-dependence as central finding: h varies by game context (threatening vs. calm), not by individual differences.

Test hypothesis directly: Extract board features (threat level, complexity, phase) and analyze P(h | board_features).

### For methods development

Improvements to address limitations:
1. Trajectory modeling: Encode full sequence s_t → ... → s_{t+h}
2. Attention mechanisms: Learn to weight features by temporal relevance
3. Single-agent tasks: Eliminate opponent confound
4. Process tracing: Collect eye-tracking or verbal protocols to validate interpretation

---

## Recommended Framing

**What to say**:

"We estimate decision-relevant temporal horizons from behavioral data using future-state-conditioned inverse models. Under the assumption of forward mental simulation, this corresponds to planning depth. However, it may also reflect which temporal scale is most informative for decisions, regardless of explicit simulation mechanism.

The method measures statistical association between actions and state pairs (s_t, s_{t+h}). In two-player games, s_{t+h} depends on both players' actions, so we capture dependency structure rather than causal 'actions to reach states.'"

**Explicit limitations**:
- Simple concatenation architecture may not capture complex planning processes
- Assumes discrete horizons h ∈ {1,2,3,4}
- Cannot distinguish mental simulation from pattern recognition
- Two-player confound: s_{t+h} not determined by a_t alone
- Measures decision-relevant horizon (statistical construct), not necessarily planning depth (cognitive mechanism)

**Robust findings**:
- Decision-relevant horizon is identifiable (93.8% accuracy)
- No correlation with expertise (r = -0.01, consistent across methods)
- Rollout method creates systematic bias (+1.09 steps)
- Findings align with van Opheusden's tree exploration vs. decision horizon distinction

**Contribution despite limitations**:

The null result on expertise is the key empirical finding. It holds across three independent estimation methods, making it unlikely to be artifact. This challenges assumptions in both IRL (h predicts skill) and cognitive science (expertise = deeper planning).

The methodological contribution is demonstrating that decision-relevant temporal horizons can be estimated from behavior, and that estimation method (rollout vs. rollout-free) substantially affects results.

---

**Last updated**: 2026-01-02
