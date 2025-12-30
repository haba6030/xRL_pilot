# AIRL 구조 비교: 4-in-a-Row vs Pedestrian

**미팅 자료 (2025-12-30)**

---

## 📋 요약 비교표

| 항목 | 4-in-a-Row (현재) | Pedestrian (기존) |
|------|------------------|-------------------|
| **Library** | Custom (discriminator) | `imitation` (공식 AIRL) |
| **Approach** | Multi-step IK → Discriminator | PPO + AIRL (adversarial) |
| **Environment** | `FourInARowEnv` (custom) | `PedestrianEnv` + Wrappers |
| **Horizon 처리** | Variable (자연스러운 종료) | **Fixed horizon** (absorbing state) |
| **Expert Data** | (state, action) pairs | Trajectories (obs, acts, rews) |
| **h 모델링** | Explicit h={1,2,3,4} models | Implicit (PPO의 gamma) |
| **Reward Learning** | ❌ 안함 (discriminator만) | ✅ AIRL로 학습 |
| **Policy Learning** | ❌ 안함 (분석만) | ✅ PPO로 학습 |
| **목적** | h 추정 + expertise 분석 | Reward 복원 + policy 학습 |

---

## 🏗️ Architecture 상세 비교

### 1. Data Pipeline

#### Pedestrian (기존)
```python
# analysis/irl/util.py

def load_traj(subjId, fixed_horizon=True):
    """Expert trajectories 로드"""
    full_log_list, max_traj_size, _ = load_subject_play_log(subjId)
    expert_trajectories = []
    
    for full_log in full_log_list:
        observations = full_log["observations"]  # (T+1, obs_dim)
        actions = full_log["actions"]            # (T,)
        rewards = full_log["rewards"]            # (T,)
        infos = [...]                            # (T,)
        
        if fixed_horizon:
            # Fixed horizon로 변환 (absorbing state 추가)
            traj = create_fixed_horizon_trajectory(
                observations, actions, rewards, infos, max_traj_size
            )
        else:
            traj = TrajectoryWithRew(
                obs=observations,
                acts=actions,
                rews=rewards,
                infos=infos,
                terminal=True
            )
        expert_trajectories.append(traj)
    
    return expert_trajectories  # imitation 라이브러리 형식
```

**특징**:
- ✅ `TrajectoryWithRew` 객체 사용 (imitation 표준)
- ✅ Fixed horizon 지원 (absorbing state)
- ✅ Rewards 포함 (full trajectory)
- ✅ Episode metadata 보존

---

#### 4-in-a-Row (현재)
```python
# preprocess_multistep_ik_data.py

def create_multistep_ik_pairs(trajectories, h):
    """Multi-step IK 페어 생성"""
    pairs = []
    
    for traj in trajectories:
        obs = traj['observations']  # (T+1, 89)
        acts = traj['actions']      # (T,)
        
        for t in range(len(acts)):
            if t + h < len(obs):
                pair = {
                    'state_current': obs[t],      # (89,)
                    'state_future': obs[t + h],   # (89,)
                    'action': acts[t],            # scalar
                    'h': h,
                    'game_id': traj['game_id'],
                    't': t
                }
                pairs.append(pair)
    
    return pairs  # List of dicts
```

**특징**:
- ❌ Custom 형식 (dict)
- ❌ Rewards 없음 (state-action pairs만)
- ✅ Multi-step structure (h 명시적)
- ❌ Variable horizon (자연스러운 종료)

---

### 2. Environment Wrapper

#### Pedestrian: FixedHorizonEnvWrapper

```python
# analysis/irl/util.py

class FixedHorizonEnvWrapper(gym.Wrapper):
    """
    Add absorbing indicator feature for all observations.
    
    Observation: [original_obs (113-dim), absorbing_flag (1-dim)] = 114-dim
    
    - absorbing_flag = 0: 정상 진행 중
    - absorbing_flag = 1: 에피소드 종료됨 (absorbing state)
    """
    
    def __init__(self, env, max_traj_size: int):
        super().__init__(env)
        self.max_traj_size = max_traj_size  # 모든 trajectory 동일 길이
        self.cur_step = 0
        self.is_absorbing = False
        
        # Observation space 확장: +1 for absorbing flag
        obs_space_size = len(env.observation_space.low) + 1
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, [0.0]]),
            high=np.concatenate([env.observation_space.high, [1.0]]),
            dtype=env.observation_space.dtype
        )
        
        # Absorbing state 정의
        self._absorbing_obs = np.concatenate([
            np.zeros(obs_space_size - 1),  # 원래 obs는 0
            [1.0]                           # absorbing flag = 1
        ])
    
    def step(self, action):
        self.cur_step += 1
        
        if not self.is_absorbing:
            # 정상 진행
            obs, rew, terminated, truncated, info = self.env.step(action)
            obs = np.concatenate([obs, [0.0]])  # absorbing_flag = 0
            
            if terminated or truncated:
                self.is_absorbing = True  # 다음부터 absorbing
        else:
            # Absorbing state에서 계속
            obs = self._absorbing_obs  # [0, 0, ..., 0, 1]
            rew = 0.0
            info = {}
        
        # max_traj_size에 도달했을 때만 truncated=True
        truncated = self.cur_step >= self.max_traj_size
        
        return obs, rew, False, truncated, info
```

**핵심 아이디어**:
1. **고정 길이**: 모든 trajectory를 `max_traj_size`로 패딩
2. **Absorbing state**: 종료 후에도 transition 계속 (obs=[0,...,0,1], rew=0)
3. **Discriminator 혜택**: 길이로 h를 구분 못함 (모두 같은 길이)

**참고 문헌**: [Variable Horizon Considered Harmful](https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html)

---

#### Pedestrian: InfiniteHorizonEnvWrapper

```python
# analysis/irl/airl.py

class InfiniteHorizonEnvWrapper(gym.Wrapper):
    """
    Training 중 에피소드가 끝나도 자동으로 리셋하여 계속 진행
    
    PPO가 계속 학습할 수 있도록 함
    """
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            # 자동 리셋 (다음 seed로)
            if self.env.cur_seed is not None:
                obs, info = self.reset(self.env.cur_seed + 1)
            else:
                obs, info = self.reset()
        
        # 항상 done=False 반환 (무한 horizon)
        return obs, rew, False, False, info
```

**사용 이유**: PPO는 finite horizon에서 학습하므로, 무한처럼 보이게 만듦

---

#### 4-in-a-Row: 현재 상태

**문제점**:
```python
# env.py에서
def step(self, action):
    # ... game logic ...
    
    # 게임이 끝나면 done=True
    if self.winner is not None or len(self.get_legal_actions()) == 0:
        done = True
        
    return obs, reward, done, info
```

❌ **Variable horizon**: 에피소드 길이가 제각각
- h=1 평균: ~17 스텝
- h=4 평균: ~26 스텝
→ Discriminator가 길이로 h를 구분할 수 있음 (confounding!)

**해결책**: Pedestrian처럼 FixedHorizonWrapper 필요
```python
# fourinarow_airl/fixed_horizon_wrapper.py (이미 구현됨!)

class FixedHorizonWrapper(gym.Wrapper):
    """
    Observation: [original_obs (89-dim), absorbing_flag (1-dim)] = 90-dim
    
    모든 에피소드를 max_episode_length로 패딩
    """
    def __init__(self, env, max_episode_length=36):
        # ... pedestrian과 동일한 로직 ...
```

✅ **이미 구현했음** (`fourinarow_airl/fixed_horizon_wrapper.py`)

---

### 3. AIRL Training

#### Pedestrian: 전체 AIRL 파이프라인

```python
# analysis/irl/airl.py

def train_AIRL(subjId, ...):
    """완전한 AIRL training"""
    
    # 1. Environment 설정
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    # 2. Generator (PPO) 초기화
    gen_algo = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_lr,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        gamma=ppo_gamma,  # ← 이게 사실상 "planning horizon"
        policy_kwargs={
            "net_arch": ppo_net_arch,
            "activation_fn": ppo_activation
        }
    )
    
    # 3. Reward Network 초기화
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hid_sizes=reward_net_hid_sizes,
        activation=reward_net_activation
    )
    
    # 4. AIRL Trainer 초기화
    airl_trainer = airl.AIRL(
        demonstrations=expert_trajectories,  # Expert data
        demo_batch_size=demo_batch_size,
        gen_algo=gen_algo,                   # PPO
        reward_net=reward_net,               # Reward function
        allow_variable_horizon=False         # Fixed horizon 사용!
    )
    
    # 5. Training loop
    airl_trainer.train(
        total_timesteps=gen_train_timesteps,
        n_disc_updates_per_round=n_disc_updates_per_round
    )
    
    # 6. 저장
    gen_algo.save(f"{save_dir}/generator.zip")
    torch.save(reward_net.state_dict(), f"{save_dir}/reward_net.pt")
    
    return gen_algo, reward_net
```

**구성요소**:
1. **Generator**: PPO (policy learner)
2. **Discriminator**: AIRL reward network
3. **Expert data**: Fixed-horizon trajectories
4. **Training**: Adversarial (PPO vs Discriminator)

---

#### 4-in-a-Row: 현재 접근

```python
# 현재 우리가 한 것

# 1. Multi-step IK 데이터 생성
pairs_h1 = create_multistep_ik_pairs(trajectories, h=1)
pairs_h4 = create_multistep_ik_pairs(trajectories, h=4)

# 2. h-specific 모델 학습
model_h1 = train_inverse_model(pairs_h1)  # LogisticRegression
model_h4 = train_inverse_model(pairs_h4)

# 3. Trajectory 생성 (rollout)
trajs_h1 = generate_trajectories(model_h1, h=1)
trajs_h4 = generate_trajectories(model_h4, h=4)

# 4. Discriminator 학습
discriminator = train_discriminator(trajs_h1, trajs_h4)

# 5. Human h 추정
h_estimates = discriminator.predict(human_data)
```

**차이점**:
- ❌ AIRL 사용 안함 (단순 discriminator)
- ❌ Reward 학습 안함
- ❌ Policy 학습 안함 (PPO 없음)
- ✅ h를 명시적으로 모델링
- ✅ Multi-step structure

---

### 4. 핵심 차이점 요약

| 측면 | Pedestrian | 4-in-a-Row |
|------|-----------|------------|
| **목적** | Reward 복원 + Policy 학습 | h 추정 + Expertise 분석 |
| **AIRL 사용** | ✅ Full AIRL (imitation lib) | ❌ Discriminator만 |
| **Reward learning** | ✅ BasicRewardNet | ❌ 없음 |
| **Policy learning** | ✅ PPO | ❌ 없음 |
| **h 모델링** | ❌ Implicit (gamma) | ✅ Explicit (h=1,2,3,4) |
| **Fixed horizon** | ✅ 114-dim (113+1) | ✅ 90-dim (89+1) |
| **Library** | `imitation` (공식) | Custom (직접 구현) |
| **Complexity** | 높음 (full RL pipeline) | 중간 (supervised learning) |

---

## 🔄 우리가 Pedestrian에서 가져온 것

### 1. Fixed Horizon Wrapper ✅

```python
# fourinarow_airl/fixed_horizon_wrapper.py

class FixedHorizonWrapper(gym.Wrapper):
    """
    Pedestrian의 FixedHorizonEnvWrapper를 4-in-a-row에 맞게 수정
    
    차이점:
    - Pedestrian: 114-dim (113 + 1)
    - 4-in-a-row: 90-dim (89 + 1)
    """
    def __init__(self, env, max_episode_length=36):
        # ... 동일한 로직 ...
```

✅ **적용 완료**: Phase 0에서 사용 중

---

### 2. Trajectory Format

**Pedestrian**:
```python
TrajectoryWithRew(
    obs=np.array(...),   # (T, obs_dim)
    acts=np.array(...),  # (T,)
    rews=np.array(...),  # (T,)
    infos=[...],         # (T,)
    terminal=True
)
```

**4-in-a-Row (현재)**:
```python
{
    'states': np.array(...),   # (T, 89)
    'actions': np.array(...),  # (T,)
    'h': 1,
    'episode': 0
}
```

❌ **아직 적용 안함**: Rewards 없음, infos 없음

---

### 3. VecNormalize

**Pedestrian**:
```python
env = VecNormalize(env, norm_obs=False, norm_reward=True)
```

**4-in-a-Row**:
```python
# 사용 안함 (아직)
```

❌ **미적용**: PPO 사용 안하므로 필요 없음

---

## 💡 Pedestrian 방식을 4-in-a-Row에 적용하려면?

### Option 1: Full AIRL (Pedestrian 방식 그대로)

```python
# fourinarow_airl/train_full_airl.py (새로 만들기)

def train_full_airl_h_aware(h_value):
    """
    h-aware AIRL: h마다 별도의 AIRL 학습
    """
    
    # 1. Environment with fixed horizon
    env = FourInARowEnv()
    env = FixedHorizonWrapper(env, max_episode_length=36)
    env = InfiniteHorizonEnvWrapper(env)  # PPO용
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    # 2. Expert trajectories (h-specific)
    expert_trajs = load_expert_trajectories_h(h=h_value)
    
    # 3. Generator (PPO)
    gen_algo = PPO("MlpPolicy", env, gamma=0.99, ...)
    
    # 4. Reward network
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    
    # 5. AIRL trainer
    airl_trainer = airl.AIRL(
        demonstrations=expert_trajs,
        gen_algo=gen_algo,
        reward_net=reward_net,
        allow_variable_horizon=False  # Fixed horizon!
    )
    
    # 6. Train
    airl_trainer.train(total_timesteps=100000)
    
    return gen_algo, reward_net
```

**장점**:
- ✅ 공식 AIRL 사용 (검증된 방법)
- ✅ Reward 복원 가능
- ✅ Policy 학습 가능
- ✅ Pedestrian 코드 거의 그대로 사용

**단점**:
- ❌ 매우 복잡함 (PPO + AIRL)
- ❌ 학습 시간 오래 걸림 (각 h마다 10만 timesteps)
- ❌ 우리 목적(h 추정)에 과한 방법

---

### Option 2: Discriminator Only (현재 방식 유지)

```python
# 현재 우리 방법 (이미 구현됨)

# 간단하고 빠름
# h 추정에 충분함
# Reward/policy 학습은 필요 없음
```

**장점**:
- ✅ 간단함
- ✅ 빠름 (discriminator만 학습)
- ✅ 목적에 충분 (h 추정)
- ✅ 이미 작동함 (93.8% 정확도)

**단점**:
- ❌ AIRL이라고 부르기 어려움 (discriminator만)
- ❌ Reward 복원 안됨

---

### Option 3: Hybrid (추천) ⭐

```python
# 현재 방법 유지 + Pedestrian 요소 추가

# 1. Fixed horizon wrapper 사용 (✅ 이미 적용)
# 2. Trajectory format 통일 (TrajectoryWithRew)
# 3. Discriminator는 현재 방식 유지
# 4. 나중에 필요하면 Full AIRL로 확장 가능
```

**구체적으로**:
```python
# fourinarow_airl/util.py (새로 만들기)

from imitation.data.types import TrajectoryWithRew

def convert_to_imitation_format(our_trajectory):
    """우리 형식 → imitation 형식"""
    return TrajectoryWithRew(
        obs=our_trajectory['states'],
        acts=our_trajectory['actions'],
        rews=np.zeros(len(our_trajectory['actions'])),  # dummy
        infos=None,
        terminal=True
    )

def load_expert_trajectories_h(h_value):
    """Pedestrian 스타일로 expert data 로드"""
    # 우리 데이터 로드
    trajs = load_our_trajectories(h=h_value)
    
    # imitation 형식으로 변환
    return [convert_to_imitation_format(t) for t in trajs]
```

---

## 📊 코드 구조 비교

### Pedestrian 구조

```
project_pedestrian/
├── pedestrian_env/
│   └── envs/
│       └── pedestrian_env.py      # Environment 정의
│
└── analysis/
    ├── irl/
    │   ├── airl.py                # AIRL training (main)
    │   ├── util.py                # Fixed horizon wrapper, load_traj
    │   └── saved/                 # 학습된 모델들
    │       ├── generator.zip      # PPO policy
    │       └── reward_net.pt      # Reward network
    │
    └── util.py                     # Data loading utilities
```

**특징**:
- Separation of concerns (env / training / utils)
- `imitation` 라이브러리 의존
- Complete RL pipeline

---

### 4-in-a-Row 구조 (현재)

```
fourinarow_airl/
├── env.py                              # Environment 정의
├── features.py                         # van Opheusden features
│
├── preprocess_multistep_ik_data.py     # Multi-step IK 페어 생성
├── train_separate_h_models.py          # h-specific 모델 학습
├── generate_trajectories_separate_h.py # Trajectory 생성 (rollout)
│
├── train_multiclass_discriminator.py   # Discriminator 학습
├── estimate_player_h_multiclass.py     # h 추정 (random rollout)
├── estimate_player_h_rollout_free.py   # h 추정 (rollout-free) ⭐
│
├── fixed_horizon_wrapper.py            # ← Pedestrian에서 가져옴!
│
├── data/
│   ├── multistep_ik/                   # IK 페어
│   └── separate_h_trajectories/        # 생성된 trajectories
│
└── models/
    ├── separate_h/                     # h-specific 모델
    └── multiclass_discriminator.pt     # Discriminator
```

**특징**:
- Custom pipeline (imitation 미사용)
- Multi-step IK 중심
- Discriminator만 학습 (no PPO)

---

## 🎯 핵심 교훈 및 권고사항

### 우리가 Pedestrian에서 배운 것

1. **Fixed Horizon이 필수다**
   - ✅ Variable horizon은 confounding 만듦
   - ✅ Absorbing state로 해결
   - ✅ 우리도 적용함 (`fixed_horizon_wrapper.py`)

2. **`imitation` 라이브러리가 편하다**
   - ✅ 검증된 AIRL 구현
   - ✅ 표준 trajectory 형식
   - ❌ 하지만 우리 목적엔 과할 수 있음

3. **Environment wrapper가 중요하다**
   - ✅ FixedHorizonWrapper
   - ✅ InfiniteHorizonWrapper (PPO용)
   - ✅ VecNormalize

---

### 우리 방식의 장점

1. **h를 명시적으로 모델링**
   - Pedestrian: gamma로만 implicit
   - 우리: h={1,2,3,4} 명시적 구분

2. **간단하고 빠름**
   - Pedestrian: Full AIRL (복잡)
   - 우리: Discriminator만 (간단)

3. **목적에 충분함**
   - Pedestrian: Reward 복원 필요
   - 우리: h 추정이 목표 → discriminator면 충분

---

### 미팅에서 논의할 점

#### 질문 1: Full AIRL로 확장할 필요가 있나?

**찬성**:
- Reward 복원 가능
- Pedestrian과 일관성
- Paper에서 "AIRL" 이름 사용 가능

**반대**:
- 우리 목적(h 추정)에 과함
- 이미 작동함 (rollout-free, AUC=0.84)
- 시간 많이 걸림

**권고**: 현재 방식 유지, 나중에 필요하면 확장

---

#### 질문 2: Pedestrian 코드 재사용?

**재사용 가능**:
- ✅ FixedHorizonWrapper (이미 사용 중)
- ✅ TrajectoryWithRew format
- ✅ VecNormalize (나중에)

**재사용 어려움**:
- ❌ AIRL trainer (목적 다름)
- ❌ PPO (policy 학습 안함)

**권고**: Wrapper와 format만 가져오기

---

#### 질문 3: 보행자 과제 적용 시?

**Option A**: Pedestrian 방식 그대로
```python
# 보행자는 이미 AIRL 인프라 있음
# h-aware version만 추가하면 됨

def train_pedestrian_h_aware(subjId, h_value):
    # Pedestrian 코드 기반
    # h마다 별도 AIRL 학습
```

**Option B**: 4-in-a-row 방식 적용
```python
# Multi-step IK + Discriminator
# 간단하고 빠름
# 하지만 Pedestrian 기존 인프라 버림
```

**권고**: Option A (기존 인프라 활용)

---

## 📋 액션 아이템 (미팅 후)

### Immediate (1주)
1. ✅ Fixed horizon 계속 사용
2. ✅ 현재 discriminator 방식 유지
3. ⬜ Trajectory format 통일 (TrajectoryWithRew)? → 논의 필요

### Short-term (2-3주)
1. ⬜ Full AIRL 구현? → 논의 필요
2. ⬜ Pedestrian에 h-aware 적용? → 논의 필요

### Long-term (1-2개월)
1. ⬜ 보행자 과제 확장 전략 결정

---

## 📚 참고 자료

**Pedestrian 프로젝트**:
- `project_pedestrian/analysis/irl/airl.py`
- `project_pedestrian/analysis/irl/util.py`

**4-in-a-Row 프로젝트**:
- `fourinarow_airl/fixed_horizon_wrapper.py` ← Pedestrian에서 가져옴
- `fourinarow_airl/estimate_player_h_rollout_free.py` ← 우리의 혁신

**문헌**:
- [Variable Horizon Considered Harmful](https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html)
- [Imitation Learning Library](https://imitation.readthedocs.io/)

---

**최종 업데이트**: 2025-12-30
**미팅 준비 완료**: 비교 완료, 논의 포인트 정리
