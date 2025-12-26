# Phase 2 Implementation: Validation Checklist

## 목표

AIRL 파이프라인 구현 시 **원칙 위반** 및 **의도하지 않은 학습**을 방지합니다.

---

## Critical Principle (절대 위반 금지)

> **Planning depth h는 POLICY에만 존재. Reward network는 depth-agnostic.**

이 원칙이 지켜지지 않으면:
- ❌ Reward-planning disentanglement 붕괴
- ❌ 연구 질문 무의미화
- ❌ 이론적 타당성 상실

---

# Step 1: Trajectory Conversion

## 📋 파일
`fourinarow_airl/airl_utils.py` - `convert_to_imitation_format()`

## ⚠️ 잠재적 위험

### 위험 1: 정보 손실
**문제**: Trajectory 변환 과정에서 중요 정보 손실
**검증**:
```python
# 변환 전후 비교
assert np.array_equal(original.observations, converted.obs)
assert np.array_equal(original.actions, converted.acts)
```
**구현**: ✅ `airl_utils.py:357-367`에서 검증

### 위험 2: Depth 정보 유출
**문제**: Observation에 depth 정보가 포함될 가능성
**검증**:
```python
# Observation은 89-dim만 포함
# 0-35: Black pieces (board state)
# 36-71: White pieces (board state)
# 72-88: Van Opheusden features
# NO depth information!
```
**구현**: ✅ `airl_utils.py:111-120`에서 명시적 확인

### 위험 3: Data type 불일치
**문제**: imitation library가 요구하는 dtype과 불일치
**검증**:
```python
observations = observations.astype(np.float32)  # imitation requirement
actions = actions.astype(np.int64)
```
**구현**: ✅ `airl_utils.py:63-70`

## ✅ Validation Checkpoints

- [x] **Checkpoint 1**: Observation shape (T+1, 89) 보존
- [x] **Checkpoint 2**: Action shape (T,) 보존
- [x] **Checkpoint 3**: 정보 손실 없음 (array_equal 검증)
- [x] **Checkpoint 4**: Terminal flag 올바름
- [x] **Checkpoint 5**: Action range [0, 35] 검증
- [x] **Checkpoint 6**: NO depth information in observations

## 🧪 테스트 실행

```bash
cd fourinarow_airl
python3 airl_utils.py
```

**예상 출력**:
```
Trajectory Conversion Validation Report
✓ Converted N trajectories
✓ Observation shape: (T+1, 89)
✓ NO planning depth h in observations
```

---

# Step 2: Reward Network Setup

## 📋 파일
`test_reward_net.py` (작성 예정)

## ⚠️ 잠재적 위험

### 위험 1: Depth parameter가 reward network에 추가
**문제**: 실수로 h를 reward network input에 포함
**금지 패턴**:
```python
# ❌ WRONG
class RewardNet(nn.Module):
    def __init__(self, obs_dim, action_dim, h):  # ← h parameter!
        self.h_embedding = nn.Embedding(5, 8)

    def forward(self, state, action, next_state, h):  # ← h in forward!
        h_emb = self.h_embedding(h)
        x = torch.cat([state, action, next_state, h_emb])
        return self.mlp(x)
```

**올바른 패턴**:
```python
# ✅ CORRECT
from imitation.rewards.reward_nets import BasicRewardNet

reward_net = BasicRewardNet(
    observation_space=env.observation_space,  # Box(89,)
    action_space=env.action_space,            # Discrete(36)
    hid_sizes=[64, 64]
)
# NO h parameter anywhere!
```

**구현 검증**: ✅ `airl_utils.py:validate_airl_setup()`에서 자동 검사

### 위험 2: Observation dimension 불일치
**문제**: Reward network가 89-dim을 기대하지 않음
**검증**:
```python
# Test forward pass
obs = env.reset()  # (89,)
action = env.action_space.sample()
next_obs, _, _, _, _ = env.step(action)

reward = reward_net(obs, action, next_obs)
print(f"Reward: {reward}")  # Should work without error
```
**구현**: ✅ `airl_utils.py:validate_reward_network_forward_pass()`

### 위험 3: Imitation library 버전 차이
**문제**: Imitation library API가 버전마다 다를 수 있음
**검증**:
```python
import imitation
print(f"Imitation version: {imitation.__version__}")
# Expected: >= 0.4.0
```

## ✅ Validation Checkpoints

- [ ] **Checkpoint 1**: Reward network에 depth 관련 attribute 없음
- [ ] **Checkpoint 2**: Forward pass에 h parameter 없음
- [ ] **Checkpoint 3**: 89-dim observation 처리 가능
- [ ] **Checkpoint 4**: Discrete(36) action 처리 가능
- [ ] **Checkpoint 5**: Output scalar reward

## 🧪 테스트 실행 (예정)

```bash
python3 test_reward_net.py
```

---

# Step 3: Generator Policy Setup

## 📋 파일
`fourinarow_airl/policy_wrapper.py` (작성 예정)

## ⚠️ 잠재적 위험

### 위험 1: Depth가 observation에 추가됨
**문제**: PPO policy가 observation으로 depth를 받을 가능성
**금지 패턴**:
```python
# ❌ WRONG: Augmented observation
obs_with_h = np.concatenate([obs, [h]])  # (90,) with depth!
action = policy(obs_with_h)
```

**올바른 패턴**:
```python
# ✅ CORRECT: Depth는 policy 내부에서만 사용
class DepthLimitedPolicyWrapper:
    def __init__(self, h):
        self.h = h  # ← Depth stored internally
        self.depth_policy = DepthLimitedPolicy(h=h)

    def __call__(self, obs):
        # obs는 89-dim (depth 정보 없음)
        # self.h를 내부적으로만 사용
        action = self.depth_policy.select_action(obs, h=self.h)
        return action
```

### 위험 2: BC (Behavior Cloning) 시 depth 유출
**문제**: BC training 시 depth가 feature로 추가될 가능성
**검증**:
```python
# BC demonstrations는 89-dim observation만 포함
for traj in bc_trajectories:
    assert traj.obs.shape[1] == 89
    # NO depth column!
```

### 위험 3: 여러 h를 동시에 학습
**문제**: 실수로 여러 h의 데이터를 섞어서 학습
**원칙**:
```python
# ✅ CORRECT: 각 h마다 별도 학습
for h in [1, 2, 4, 8]:
    # Create h-specific generator
    generator_h = create_generator(h=h)

    # Train AIRL with THIS h only
    trainer = AIRL(gen_algo=generator_h, ...)
    trainer.train()

    # Save separately
    save(f'generator_h{h}.pt')
```

## ✅ Validation Checkpoints

- [ ] **Checkpoint 1**: Policy observation은 89-dim (depth 없음)
- [ ] **Checkpoint 2**: Depth는 policy 내부 변수로만 존재
- [ ] **Checkpoint 3**: BC trajectories에 depth 정보 없음
- [ ] **Checkpoint 4**: 각 h마다 독립적으로 학습
- [ ] **Checkpoint 5**: Generator outputs만 h-dependent

---

# Step 4: AIRL Training

## 📋 파일
`train_baseline_airl.py` (작성 예정)

## ⚠️ 잠재적 위험

### 위험 1: Discriminator가 depth를 간접적으로 학습
**문제**: Different h generators가 identifiable한 pattern을 만들면, discriminator가 이를 학습할 가능성
**예시**:
- h=1: 짧은 trajectory
- h=8: 긴 trajectory
- Discriminator가 "trajectory length"로 depth를 추론

**대응**:
```python
# Trajectory length normalization 또는
# Fixed-length padding 사용
# 하지만 이는 information loss를 유발할 수 있음

# Better: Accept this as valid signal
# Discriminator는 "behavior pattern"을 학습하는 것이 목표
# Pattern이 depth와 correlate되는 것은 자연스러움
# 단, discriminator architecture 자체에 h가 없어야 함
```

**원칙**: Discriminator가 trajectory pattern으로부터 depth를 **추론**하는 것은 OK. 하지만 **명시적 depth input**은 절대 안 됨.

### 위험 2: Reward network 가중치 공유
**문제**: 여러 h 실험 시 실수로 같은 reward network를 재사용
**검증**:
```python
# ❌ WRONG: Reusing same reward_net
reward_net = BasicRewardNet()
for h in [1, 2, 4, 8]:
    trainer = AIRL(reward_net=reward_net, ...)  # Same object!
    trainer.train()

# ✅ CORRECT: Fresh reward_net for each h
for h in [1, 2, 4, 8]:
    reward_net = BasicRewardNet()  # New instance
    trainer = AIRL(reward_net=reward_net, ...)
    trainer.train()
```

### 위험 3: Expert data contamination
**문제**: Expert data가 h label을 포함
**검증**:
```python
# Expert trajectories should NOT have depth labels
for traj in expert_trajectories:
    # Only (s, a, s') tuples
    # NO h information
    assert not hasattr(traj, 'depth')
    assert not hasattr(traj, 'h')
```

**구현**: ✅ Expert data는 GameTrajectory에서 변환되며, h 정보 없음

### 위험 4: Evaluation metric 오해
**문제**: Discriminator accuracy를 잘못 해석
**올바른 해석**:
```python
# Discriminator accuracy ~ 0.5 = GOOD
# (Expert와 Generated를 구분 못함 = Generator가 잘 학습됨)

# Discriminator accuracy >> 0.5 = BAD
# (Generator가 Expert와 다름)

# 우리의 목표:
# h별로 학습 후, 어떤 h가 가장 빨리 acc ~ 0.5에 도달하는가?
```

## ✅ Validation Checkpoints

- [ ] **Checkpoint 1**: Discriminator에 depth input 없음 (재확인)
- [ ] **Checkpoint 2**: 각 h마다 fresh reward network
- [ ] **Checkpoint 3**: Expert data에 depth label 없음
- [ ] **Checkpoint 4**: Metrics 올바르게 해석 (disc_acc ~ 0.5 목표)
- [ ] **Checkpoint 5**: Training 안정성 (loss divergence 없음)

---

# Step 5: Results Analysis

## 📋 파일
`analyze_airl_results.py` (작성 예정)

## ⚠️ 잠재적 위험

### 위험 1: "h-specific reward" 용어 사용
**문제**: 결과 분석 시 "h=4 reward"처럼 표현
**올바른 표현**:
```python
# ❌ WRONG terminology
"h=4 reward network"
"Reward for depth 4"

# ✅ CORRECT terminology
"Reward learned with h=4 generator"
"Reward trained using depth-4 policy"
```

### 위험 2: Reward network 비교 시 h 혼동
**문제**: 여러 h의 reward를 직접 비교
**주의**:
```python
# Reward networks are NOT directly comparable
# Each was learned with different generator

# What we compare:
# - Discrimination accuracy (which h → best acc?)
# - Imitation quality (which h → trajectories most similar to expert?)
# - Expertise prediction (which h → best expert/novice classifier?)
```

### 위험 3: Causal inference 오류
**문제**: "h=8이 best이므로 expert는 h=8로 planning한다"
**올바른 해석**:
```python
# ✅ CORRECT interpretation:
# "Expert behavior is MOST CONSISTENT with h=8 planning assumption"
# NOT: "Expert uses h=8 algorithm"
```

## ✅ Validation Checkpoints

- [ ] **Checkpoint 1**: 용어 사용 정확 ("trained with h=X")
- [ ] **Checkpoint 2**: Reward network 비교 방법 타당
- [ ] **Checkpoint 3**: Causal claim 적절
- [ ] **Checkpoint 4**: 결과 해석 PLANNING_DEPTH_PRINCIPLES.md 준수

---

# Overall Validation Protocol

## Pre-Implementation Checklist

매 단계 구현 **전**:

- [ ] PLANNING_DEPTH_PRINCIPLES.md 재확인
- [ ] 해당 단계 위험 요소 검토
- [ ] Validation checkpoints 준비

## Implementation Checklist

구현 **중**:

- [ ] Depth가 reward network에 들어가지 않는지 확인
- [ ] Observation이 89-dim만 유지하는지 확인
- [ ] 각 h가 독립적으로 학습되는지 확인

## Post-Implementation Checklist

구현 **후**:

- [ ] 모든 validation checkpoints 통과
- [ ] Test 코드 실행 및 검증
- [ ] 결과를 문서화할 때 용어 정확히 사용

---

# Emergency: If Principle Violated

만약 구현 과정에서 원칙 위반이 발견되면:

## 1. STOP immediately
구현을 멈추고 코드 리뷰

## 2. Identify the violation
어디서 depth가 reward로 유출되었는가?

## 3. Fix at the source
해당 코드를 완전히 제거 또는 수정

## 4. Re-validate
모든 checkpoints 재확인

## 5. Document
RESPONSE_TO_FEEDBACK.md에 기록

---

# Quick Reference: What Goes Where

```
┌─────────────────────────────────────────┐
│ Planning Depth h                        │
│ Location: POLICY (Generator) ONLY      │
│                                         │
│ ✓ DepthLimitedPolicy(h=h)              │
│ ✓ BC training with h-specific data     │
│ ✓ File names: generator_h4.pt          │
│                                         │
│ ✗ NOT in reward network                │
│ ✗ NOT in discriminator                 │
│ ✗ NOT in observations                  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Reward Network                          │
│ Location: DISCRIMINATOR                │
│                                         │
│ ✓ Input: (state, action, next_state)   │
│ ✓ Output: scalar reward                │
│ ✓ Same architecture for ALL h          │
│                                         │
│ ✗ NO h parameter                       │
│ ✗ NO h-related attributes              │
│ ✗ NO depth conditioning                │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Observations (State)                    │
│ Dimension: 89                           │
│                                         │
│ ✓ 0-35: Black pieces                   │
│ ✓ 36-71: White pieces                  │
│ ✓ 72-88: Van Opheusden features        │
│                                         │
│ ✗ NO depth information                 │
│ ✗ NO h label                           │
│ ✗ NO augmented features                │
└─────────────────────────────────────────┘
```

---

**Document**: PHASE2_VALIDATION_CHECKLIST.md
**Purpose**: Prevent principle violations during Phase 2 implementation
**Status**: Active - Use for every implementation step
**Next**: Begin Step 1 with full validation
