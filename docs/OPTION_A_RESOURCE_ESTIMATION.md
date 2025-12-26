# Option A 실행 리소스 및 시간 예상

## 시스템 사양

**현재 시스템**:
- **Model**: MacBook Pro M4 Max
- **CPU**: 14 cores (Apple Silicon)
- **Memory**: 36 GB RAM
- **GPU**: Integrated (Apple M4 Max GPU)

---

## Option A 실행 파라미터

### 기본 설정

```python
# 각 h별 학습 설정
h_values = [1, 2, 4, 8]  # 4개 실험
total_timesteps = 50000  # Option A 권장값
demo_batch_size = 64
n_disc_updates_per_round = 4
gen_train_timesteps = 2048
```

### 데이터 생성

```python
# Expert data (BFS 생성)
num_expert_episodes = 100  # h당
avg_episode_length = 15    # 약 15 moves per game
```

---

## 리소스 예상

### 1. 메모리 사용량

#### **Expert Data (BFS 생성)**
```
단일 trajectory:
- observations: (T+1, 89) * 4 bytes = ~6 KB per game
- actions: (T,) * 8 bytes = ~120 bytes per game
- Total per game: ~6-10 KB

100 episodes * 4 depths = 400 games
Total expert data: ~2.5 MB
```
**메모리 사용**: **< 10 MB** (무시 가능)

---

#### **AIRL Training**
```
PPO policy network:
- MlpPolicy [64, 64] layers
- Input: 89-dim
- Output: 36-dim (actions)
- Parameters: ~10K weights * 4 bytes = ~40 KB

Reward network:
- BasicRewardNet [64, 64]
- Input: (89 + 36 + 89) = 214-dim
- Output: 1-dim
- Parameters: ~20K weights * 4 bytes = ~80 KB

Replay buffer:
- gen_train_timesteps = 2048 transitions
- (obs, action, reward, next_obs) per transition
- 2048 * (89 + 1 + 1 + 89) * 4 bytes = ~1.5 MB

Total per iteration: ~2 MB
```

**Peak 메모리 사용**: **~500 MB - 1 GB** (단일 h)

**모든 h 동시 실행 시**: **~2-4 GB**
- 귀하의 시스템(36 GB)에서 **여유 있음** ✅

---

### 2. CPU 사용량

#### **BFS Data Generation** (C++ wrapper)
```
- Single-threaded BFS per game
- 100 episodes * 4 depths = 400 games
- Avg ~1-2 seconds per game
- Total: ~10-15 minutes (CPU bound)
```

#### **AIRL Training**
```
PyTorch operations:
- MLP forward/backward passes
- Apple Silicon MPS acceleration 가능
- Multi-core 활용 (14 cores)

Per iteration:
- Generator rollout: 2048 timesteps
  → ~5-10 seconds (환경 시뮬레이션)
- Discriminator update: 4 updates * 64 batch
  → ~2-3 seconds (네트워크 학습)
- Generator update (PPO): 10 epochs
  → ~5-10 seconds

Total per iteration: ~15-25 seconds
```

**CPU 활용률**: 평균 **50-70%** (14 cores 중 7-10 cores 활용)

---

### 3. GPU 사용량

**Apple M4 Max GPU**:
- PyTorch MPS backend 사용 가능
- 작은 네트워크이므로 GPU benefit 제한적
- CPU만 사용해도 충분히 빠름

**권장**: CPU만 사용 (MPS는 선택사항)

---

## 실행 시간 예상

### 단일 h 실험

#### **Option A (Pure NN, 50K timesteps)**

```
Total iterations = total_timesteps / gen_train_timesteps
                 = 50000 / 2048
                 = ~24 iterations

Time per iteration: ~20 seconds (평균)

Total training time per h:
= 24 iterations * 20 sec/iter
= 480 seconds
= ~8 minutes
```

**단일 h 학습**: **~8-12 분**

---

#### **전체 파이프라인 (단일 h)**

```
1. BFS Expert Data 생성: ~3 분
2. AIRL Training: ~10 분
3. Evaluation: ~2 분
────────────────────────────
Total per h: ~15 분
```

---

### 전체 실험 (모든 h)

#### **Sequential 실행** (h=1,2,4,8 순차)

```
4 depths * 15 min/depth = 60 분
```

**총 소요 시간**: **약 1 시간** ⏱️

---

#### **Parallel 실행** (가능하면)

```python
# 4개 h를 병렬로 실행 (멀티프로세싱)
# 각 h는 독립적이므로 가능

# 메모리: 4 GB (충분함)
# CPU: 14 cores / 4 processes = ~3.5 cores per process
```

**총 소요 시간**: **약 15-20 분** ⚡
(병렬 실행 시 약 **75% 시간 단축**)

---

## 상세 시간 분석

### BFS Data Generation (per h)

| 항목 | 시간 |
|-----|------|
| 100 episodes 생성 | 2-3 분 |
| Trajectory 저장 | < 10 초 |
| **Total** | **~3 분** |

---

### AIRL Training (per h, 50K timesteps)

| Iteration | 작업 | 시간 |
|-----------|------|------|
| 1 | Generator rollout (2048 steps) | 5-8 초 |
| 1 | Discriminator update (4x) | 2-3 초 |
| 1 | Generator update (PPO) | 5-8 초 |
| **1** | **Iteration total** | **~15-20 초** |

```
24 iterations * 18 sec (평균) = 432 sec = ~7 분
```

**AIRL 학습**: **7-10 분**

---

### Evaluation (per h)

| 작업 | 시간 |
|-----|------|
| 50 episodes 생성 | 1-2 분 |
| Metrics 계산 | < 30 초 |
| **Total** | **~2 분** |

---

## 최적화 옵션

### 1. Timesteps 조정

**빠른 테스트**:
```python
total_timesteps = 10000  # 50K → 10K
# 시간: ~2-3 분 per h
# 총: ~15 분 (4 depths)
```

**중간 설정**:
```python
total_timesteps = 25000  # 50K → 25K
# 시간: ~5 분 per h
# 총: ~30 분 (4 depths)
```

**권장 설정** (Option A):
```python
total_timesteps = 50000  # 원래대로
# 시간: ~10 분 per h
# 총: ~1 시간 (4 depths)
```

---

### 2. Expert Episodes 조정

**최소 설정**:
```python
num_expert_episodes = 50  # 100 → 50
# BFS 생성: 1-2 분 per h
```

**권장 설정**:
```python
num_expert_episodes = 100  # 원래대로
# BFS 생성: 2-3 분 per h
```

**충분한 설정**:
```python
num_expert_episodes = 200  # 더 많은 데이터
# BFS 생성: 5-6 분 per h
```

---

### 3. Batch Size 조정

**빠른 학습** (품질 저하 가능):
```python
demo_batch_size = 32      # 64 → 32
gen_train_timesteps = 1024  # 2048 → 1024
# 시간: ~5 분 per h
```

**균형 설정** (권장):
```python
demo_batch_size = 64       # 원래대로
gen_train_timesteps = 2048  # 원래대로
# 시간: ~10 분 per h
```

---

## 실행 전략

### 전략 1: Sequential 실행 (안전함)

```bash
# h=1,2,4,8 순차 실행
for h in 1 2 4 8; do
    python3 fourinarow_airl/train_airl_pure_nn.py \
        --h $h \
        --total_timesteps 50000 \
        --output_dir models/airl_pure_nn_results
done

# 총 시간: ~1 시간
# 메모리: ~1 GB peak
# CPU: 50-70% 활용
```

**장점**: 안정적, 메모리 부담 없음
**단점**: 시간이 오래 걸림

---

### 전략 2: Parallel 실행 (빠름)

```bash
# 4개 h를 동시에 실행 (백그라운드)
for h in 1 2 4 8; do
    python3 fourinarow_airl/train_airl_pure_nn.py \
        --h $h \
        --total_timesteps 50000 \
        --output_dir models/airl_pure_nn_results &
done

wait  # 모든 작업 완료 대기

# 총 시간: ~15-20 분
# 메모리: ~4 GB peak
# CPU: 90-100% 활용
```

**장점**: 빠름 (75% 시간 단축)
**단점**: 메모리 사용 증가, CPU 부하 높음

**귀하의 시스템**: 14 cores, 36 GB RAM → **병렬 실행 가능** ✅

---

### 전략 3: 2개씩 병렬 (절충안)

```bash
# h=1,2 먼저
python3 fourinarow_airl/train_airl_pure_nn.py --h 1 --total_timesteps 50000 &
python3 fourinarow_airl/train_airl_pure_nn.py --h 2 --total_timesteps 50000 &
wait

# h=4,8 나중에
python3 fourinarow_airl/train_airl_pure_nn.py --h 4 --total_timesteps 50000 &
python3 fourinarow_airl/train_airl_pure_nn.py --h 8 --total_timesteps 50000 &
wait

# 총 시간: ~30 분
# 메모리: ~2 GB peak
# CPU: 70-80% 활용
```

**장점**: 시간 단축 + 안정성
**권장**: **이 방법 추천** ⭐

---

## 예상 시간표 (권장 설정)

### Sequential 실행

| 시간 | 작업 |
|------|------|
| 0:00 | 시작 |
| 0:03 | h=1 BFS 데이터 생성 완료 |
| 0:13 | h=1 AIRL 학습 완료 |
| 0:15 | h=1 평가 완료 |
| 0:18 | h=2 BFS 데이터 생성 완료 |
| 0:28 | h=2 AIRL 학습 완료 |
| 0:30 | h=2 평가 완료 |
| 0:33 | h=4 BFS 데이터 생성 완료 |
| 0:43 | h=4 AIRL 학습 완료 |
| 0:45 | h=4 평가 완료 |
| 0:48 | h=8 BFS 데이터 생성 완료 |
| 0:58 | h=8 AIRL 학습 완료 |
| **1:00** | **전체 완료** ✅ |

---

### Parallel 실행 (2개씩)

| 시간 | 작업 |
|------|------|
| 0:00 | h=1,2 동시 시작 |
| 0:03 | h=1,2 BFS 데이터 완료 |
| 0:13 | h=1,2 AIRL 학습 완료 |
| 0:15 | h=1,2 평가 완료 |
| 0:15 | h=4,8 동시 시작 |
| 0:18 | h=4,8 BFS 데이터 완료 |
| 0:28 | h=4,8 AIRL 학습 완료 |
| **0:30** | **전체 완료** ✅ |

---

## 디스크 사용량

### 저장되는 파일들

```
models/airl_pure_nn_results/
├── airl_pure_generator_h1.zip    ~2 MB
├── airl_pure_reward_h1.pt        ~200 KB
├── airl_pure_metadata_h1.pkl     ~10 KB
├── airl_pure_generator_h2.zip    ~2 MB
├── airl_pure_reward_h2.pt        ~200 KB
├── airl_pure_metadata_h2.pkl     ~10 KB
├── airl_pure_generator_h4.zip    ~2 MB
├── airl_pure_reward_h4.pt        ~200 KB
├── airl_pure_metadata_h4.pkl     ~10 KB
├── airl_pure_generator_h8.zip    ~2 MB
├── airl_pure_reward_h8.pt        ~200 KB
└── airl_pure_metadata_h8.pkl     ~10 KB

Total: ~9 MB
```

**디스크 사용**: **< 10 MB** (무시 가능)

---

## 모니터링 팁

### 실행 중 모니터링

```bash
# Terminal 1: 학습 실행
python3 fourinarow_airl/train_airl_pure_nn.py --h 4 --total_timesteps 50000

# Terminal 2: 리소스 모니터링
# CPU/메모리 확인
top -pid $(pgrep -f train_airl_pure_nn)

# 또는 htop (설치 필요)
htop
```

### Tensorboard 로그

```bash
# 학습 중 progress 확인
tensorboard --logdir tensorboard_logs/ppo_pure_h4/

# 브라우저: http://localhost:6006
```

---

## 비교: Option A vs Option B

### 시간 비교

| 항목 | Option A | Option B |
|-----|---------|---------|
| **BFS 데이터** | 3 분 | 3 분 |
| **BC 학습** | - | 5 분 |
| **PPO 생성** | - | 1 분 |
| **AIRL 학습** | 10 분 (50K) | 3 분 (10K) |
| **Total per h** | **~13 분** | **~12 분** |

**총 시간 (4 depths)**:
- **Option A**: ~1 시간 (sequential)
- **Option B**: ~50 분 (sequential)

**차이**: 큰 차이 없음! (단, Option A는 더 긴 학습 권장)

---

### 리소스 비교

| 항목 | Option A | Option B |
|-----|---------|---------|
| **메모리** | ~1 GB | ~1.5 GB |
| **CPU** | 50-70% | 60-80% |
| **디스크** | 9 MB | 15 MB |

**차이**: 거의 동일

---

## 권장 실행 계획

### 🎯 권장 설정

```bash
# 2개씩 병렬 실행 (절충안)
# 총 30분 소요

# Step 1: h=1,2
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 1 \
    --total_timesteps 50000 \
    --demo_batch_size 64 \
    --output_dir models/airl_pure_nn_results &

python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 2 \
    --total_timesteps 50000 \
    --demo_batch_size 64 \
    --output_dir models/airl_pure_nn_results &

wait

# Step 2: h=4,8
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 4 \
    --total_timesteps 50000 \
    --demo_batch_size 64 \
    --output_dir models/airl_pure_nn_results &

python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 8 \
    --total_timesteps 50000 \
    --demo_batch_size 64 \
    --output_dir models/airl_pure_nn_results &

wait

echo "전체 학습 완료!"
```

**예상 시간**: **~30 분**
**메모리**: **~2 GB**
**CPU**: **70-80%**

---

## 빠른 테스트 (권장)

### 먼저 단일 h로 테스트

```bash
# h=4로 빠른 테스트 (10K timesteps)
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 4 \
    --total_timesteps 10000 \
    --output_dir models/airl_pure_nn_test

# 예상 시간: ~3 분
# 성공하면 전체 실험 진행
```

---

## 요약

### ✅ 귀하의 시스템 (MacBook Pro M4 Max)

**충분히 빠르고 강력합니다!**

| 항목 | 예상 |
|-----|------|
| **메모리** | 36 GB 중 ~2-4 GB 사용 (여유 있음) |
| **CPU** | 14 cores 중 7-12 cores 활용 (충분함) |
| **시간** | Sequential: ~1 시간 / Parallel: ~30 분 |
| **디스크** | ~10 MB (무시 가능) |

### 🎯 권장 실행 방식

1. **빠른 테스트**: 단일 h (10K timesteps) → **3 분**
2. **본 실험**: 2개씩 병렬 (50K timesteps) → **30 분**
3. **여유 있으면**: 순차 실행 → **1 시간**

### ⚡ 최적화 팁

- **병렬 실행**: 2개씩 → 시간 50% 단축
- **Timesteps 조정**: 25K로 줄이면 → 시간 50% 단축
- **MPS 사용**: GPU 가속 (옵션) → 10-20% 속도 향상

**결론**: **30분-1시간** 안에 Option A 전체 실험 완료 가능! 🚀
