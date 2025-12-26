# Planning Depth Integration Principles

## Critical Design Principle (Non-Negotiable)

**Distillation occurs ONLY at the policy (generator) level.**
**The reward network MUST remain depth-agnostic.**

This document clarifies the exact role of planning depth h in our AIRL framework and prevents the catastrophic failure mode where planning constraints leak into the reward representation.

---

## 1. The Failure Mode We Must Avoid (Yao et al., 2024 Warning)

### What Goes Wrong

If planning depth is encoded into the **reward network**:

```python
# ❌ WRONG: Depth-conditioned reward
class DepthConditionedRewardNet(nn.Module):
    def __init__(self):
        self.h_embedding = nn.Embedding(5, 8)  # h → embedding
        # ...

    def forward(self, state, action, h):
        h_emb = self.h_embedding(h)
        # Reward becomes a function of h
        reward = self.mlp(state, action, h_emb)
        return reward
```

**Consequence**:
1. The reward itself encodes "this action is good IF you plan h steps ahead"
2. Planning depth is no longer a **testable hypothesis** but a **baked-in assumption**
3. Comparing different h values becomes meaningless (each h has its own reward)
4. **Reward-planning disentanglement collapses** (Yao et al., 2024)

### The Research Question Dies

Original question:
> "Which planning depth h best explains expert behavior?"

After reward contamination:
> "We trained 5 different reward functions, one for each h. They all work." ← Meaningless

---

## 2. Correct Architecture: Depth Only in Generator

### Structure

```
Expert Data (human trajectories)
    ↓
AIRL Training Loop:
    ↓
┌──────────────────────────────────────────────┐
│ Generator (Policy) ← h is HERE               │
│   - BFS with depth limit h                   │
│   - OR: Neural policy distilled from BFS_h   │
│   - Generates trajectories τ_gen             │
└──────────────────────────────────────────────┘
    ↓ τ_gen
┌──────────────────────────────────────────────┐
│ Discriminator (Reward Network)               │
│   - NO h parameter                           │
│   - Input: (s, a, s') only                   │
│   - Output: r(s, a, s')                      │
│   - SAME architecture for all h              │
└──────────────────────────────────────────────┘
    ↓ reward signal
Back to Generator (RL update)
```

### Code Template (Correct)

```python
# ✅ CORRECT: Depth in generator, not discriminator

for h in [2, 4, 6, 8, 10]:
    print(f"Training with planning depth h={h}")

    # 1. Create depth-limited generator
    generator_h = DepthLimitedBFSPolicy(h=h)  # ← h is here
    gen_algo = PPO(generator_h, env, ...)

    # 2. Create depth-AGNOSTIC reward network
    reward_net = BasicRewardNet(          # ← NO h parameter
        observation_space=env.observation_space,
        action_space=env.action_space,
        hid_sizes=[64, 64]
    )
    # Same architecture for all h!

    # 3. Train AIRL
    trainer = AIRL(
        demonstrations=expert_trajectories,
        reward_net=reward_net,
        gen_algo=gen_algo,
        ...
    )

    trainer.train(total_timesteps=100000)

    # 4. Save result
    # Note the naming: "reward learned WITH h={h} generator"
    # NOT "h-specific reward"
    torch.save(reward_net.state_dict(), f'reward_trained_with_h{h}.pt')
```

---

## 3. What Does "Depth-Specific Reward" Actually Mean?

### Wrong Interpretation ❌

> "We learn 5 different reward functions, one optimized for each planning depth."

This implies r_2(s,a), r_4(s,a), ..., r_10(s,a) are fundamentally different objectives.

### Correct Interpretation ✅

> "We use the SAME reward network architecture, but train it with 5 different generators (each with different h). The learned reward differs because the generator's behavior differs."

The reward difference is a **consequence** of generator planning depth, not a **design choice**.

### Analogy

Think of AIRL as "reverse-engineering the reward from observed behavior":

- Expert data: Human gameplay (fixed)
- Generator with h=2: Generates shallow-planning trajectories
  - AIRL learns: "What reward makes shallow planning match expert?"
  - Result: reward_h2(s,a)

- Generator with h=8: Generates deep-planning trajectories
  - AIRL learns: "What reward makes deep planning match expert?"
  - Result: reward_h8(s,a)

**Key insight**: If expert data IS shallow planning, then:
- reward_h2 will be **simple** (matches easily)
- reward_h8 will be **complex** (needs to compensate for over-planning)

This asymmetry tells us about the expert's true planning depth.

---

## 4. BFS Distillation: Policy-Level Only

### What Is Being Distilled?

```
C++ BFS with depth h
    ↓ (behavior cloning)
Neural Policy_h  ← Distillation happens HERE
    ↓ (AIRL fine-tuning)
Final Generator_h
    ↓ (generates trajectories)
Discriminator learns reward ← NO distillation here
```

**Distillation = Teaching a neural network to mimic BFS behavior**

NOT: "Extracting BFS heuristic into reward network"

### Two-Stage Process (Recommended)

**Stage 1: Pre-training (Behavior Cloning)**

```python
# Generate BFS rollouts with depth h
bfs_policy = BFSPolicy(h=h, params=expert_params)
trajectories_h = []
for episode in range(1000):
    obs = env.reset()
    traj = []
    while not done:
        action = bfs_policy.select_action(obs, h=h)  # Uses h-limited search
        traj.append((obs, action))
        obs, reward, done, _, _ = env.step(action)
    trajectories_h.append(traj)

# Train neural network to mimic BFS
neural_policy_h = NeuralPolicy()
neural_policy_h.train_behavior_cloning(trajectories_h)
```

**Stage 2: AIRL Fine-tuning**

```python
# Use pre-trained neural policy as AIRL generator
gen_algo = PPO(neural_policy_h, env, ...)  # ← Initialized from BFS distillation

# Reward network starts from scratch (or random init)
reward_net = BasicRewardNet(...)  # ← NO h, NO BFS knowledge

trainer = AIRL(
    demonstrations=expert_trajectories,
    reward_net=reward_net,
    gen_algo=gen_algo,
)

trainer.train(...)
```

**What Gets Distilled Where**:
- BFS planning strategy → Neural policy (Stage 1)
- Expert behavior → Reward network (Stage 2, via AIRL)

**What Does NOT Get Distilled**:
- BFS heuristic → Reward network ❌
- Planning depth → Reward network ❌

---

## 5. Experimental Comparison Across h

### What We Compare

| h | Generator | Reward Net | Training Result |
|---|-----------|------------|-----------------|
| 2 | BFS depth=2 | **Same arch** | reward_h2.pt |
| 4 | BFS depth=4 | **Same arch** | reward_h4.pt |
| 6 | BFS depth=6 | **Same arch** | reward_h6.pt |
| 8 | BFS depth=8 | **Same arch** | reward_h8.pt |
| 10 | BFS depth=10 | **Same arch** | reward_h10.pt |

### Metrics

1. **Discriminator accuracy** (expert vs generated)
   - Lower is better (means generator matches expert)
   - Hypothesis: Best match at h = expert's true depth

2. **Reward complexity** (number of active features)
   - If h is too shallow: reward must compensate → complex
   - If h matches expert: reward is simpler
   - If h is too deep: reward must suppress over-planning → complex

3. **Expertise prediction**
   - Does learned h correlate with expert/novice labels?
   - Van Opheusden (2023): Experts plan deeper

### Research Question (Precise Formulation)

> "Given expert trajectories, which planning depth assumption h (in the generator) requires the simplest reward function to match expert behavior?"

**NOT**:
> "Which h-specific reward best fits expert data?" ← This is circular

---

## 6. Extension to Human Data: Framing Principles

### Wrong Framing ❌

> "Humans use BFS algorithm with depth h."

This is indefensible. Humans don't run BFS.

### Correct Framing ✅

> "BFS with depth h serves as a **computational probe**: a structured hypothesis about planning horizon that we can test empirically."

### What We Actually Claim

1. **Computational level** (Marr):
   - Humans solve 4-in-a-row by evaluating future positions
   - Depth h = how far ahead they simulate

2. **Algorithmic level**:
   - BFS is a **normative model** (optimal given constraints)
   - Not a **process model** (not claiming humans run BFS)

3. **Experimental test**:
   - IF human behavior matches BFS_h output
   - THEN human planning horizon ≈ h (computationally)
   - Regardless of actual neural implementation

### Analogous Example

Physics: "Projectile motion follows parabolic trajectory"
- NOT claiming: "The ball computes y = ax² + bx + c"
- Claiming: "Ball's behavior is consistent with this equation"

Our claim: "Expert behavior is consistent with h=8 planning horizon"
- NOT claiming: "Expert runs BFS with h=8"
- Claiming: "Expert's decisions match what BFS(h=8) would produce"

---

## 7. Summary: The Three Locks

To preserve reward-planning disentanglement, we enforce:

### Lock 1: Architectural Separation
```python
# Generator
class DepthLimitedPolicy:
    def __init__(self, h):  # ← h lives here
        self.h = h

# Discriminator
class RewardNet:
    def __init__(self):  # ← NO h parameter
        # Architecture is h-agnostic
```

### Lock 2: Training Isolation
```
Distillation phase:
    BFS → Neural Policy  ✅ (policy-level)
    BFS → Reward Net     ❌ (forbidden)

AIRL phase:
    Expert → Reward Net  ✅ (via discriminator)
    Depth h → Reward Net ❌ (only via generator behavior)
```

### Lock 3: Interpretation Discipline

When writing paper/talks:
- ✅ "Reward learned using h=6 generator"
- ❌ "h=6 reward"
- ✅ "Planning depth as generator constraint"
- ❌ "Planning depth as reward parameter"
- ✅ "BFS as computational probe"
- ❌ "BFS as human algorithm"

---

## 8. Checklist: Before Any Implementation/Writing

- [ ] Does the reward network have ANY h-related input? → Must be NO
- [ ] Does the discriminator see h? → Must be NO
- [ ] Is h mentioned in reward network architecture? → Must be NO
- [ ] Are we comparing "generators with different h" or "rewards with different h"? → Must be former
- [ ] Does the paper say "humans use BFS"? → Must say "BFS as probe"
- [ ] Can we answer "Why not just use depth-conditioned reward?"? → Yes (violates disentanglement)

---

## 9. Responses to Potential Reviewer Attacks

### Attack 1: "This is just MPC-IRL"

**Response**:
> "Unlike MPC-IRL which treats planning as a fixed algorithmic choice, we treat planning depth as a **latent variable to be inferred**. By comparing AIRL performance across h ∈ {2,4,6,8,10}, we identify which depth best explains expert behavior, enabling expertise prediction."

### Attack 2: "Why not condition reward on h directly?"

**Response**:
> "Conditioning reward on h conflates two distinct sources of behavioral variation: reward differences and planning capacity differences. By keeping the reward network depth-agnostic, we can isolate planning depth as an independent factor, which is critical for our research question about expertise and cognitive constraints."

Reference: Yao et al. (2024) - planning horizon as latent confounder in IRL.

### Attack 3: "BFS is not a realistic human model"

**Response**:
> "BFS serves as a **computational probe**, not a process model. We make no claims about neural implementation. The test is behavioral: IF expert decisions match BFS(h=8) output, THEN the expert's planning horizon is computationally equivalent to h=8, regardless of algorithm. This follows the Marr computational level of analysis."

Reference: van Opheusden et al. (2023) used same approach.

---

## References

- **Yao et al. (2024)**: "IRL and Planning" - Planning horizon as latent confounder
- **van Opheusden et al. (2023)**: "Expertise increases planning depth" - BFS as normative model
- **Fu et al. (2018)**: AIRL - Adversarial IRL framework
- **Marr (1982)**: Vision - Computational, algorithmic, implementational levels

---

## Document Status

- **Version**: 1.0
- **Date**: 2025-12-23
- **Purpose**: Prevent reward-planning conflation in AIRL implementation
- **Audience**: Internal research team, future code reviewers

**This document overrides any conflicting statements in**:
- AIRL_DESIGN.md (Section 5-6)
- IMPLEMENTATION_STATUS.md (Phase 2 descriptions)
- Any code comments that say "depth-specific reward"

**All future implementations MUST comply with these principles.**
