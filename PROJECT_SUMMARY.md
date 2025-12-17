# Planning-Aware IRL/AIRL 연구 프로젝트 요약

**프로젝트명**: Planning-Aware IRL/AIRL for Expertise, Clinical Traits, and Neural Links

**시작일**: 2024-12-17
**최종 업데이트**: 2024-12-17

**팀원 온보딩 및 프로젝트 팔로우업을 위한 종합 문서**

---

## 📋 목차

1. [연구 목적 및 동기](#연구-목적-및-동기)
2. [핵심 가정 및 가설](#핵심-가정-및-가설)
3. [연구 대상 및 지표](#연구-대상-및-지표)
4. [연구 방법론](#연구-방법론)
5. [진행 현황](#진행-현황)
6. [주요 발견](#주요-발견)
7. [다음 단계](#다음-단계)
8. [참고 자료](#참고-자료)

---

## 🎯 연구 목적 및 동기

### 연구 질문

**Q1**: Planning depth가 expertise와 clinical traits를 설명하는 독립적인 요인인가?

**Q2**: Planning을 명시적으로 모델링하면 IRL/AIRL의 reward identifiability가 개선되는가?

**Q3**: Planning parameters가 neural mechanisms와 연결되는가?

### 동기

기존 IRL/AIRL 연구는 **reward function만 추론**하지만, 실제 인간 행동은:
- **Planning mechanisms**에 의해 형성됨 (얼마나 깊이 탐색하는가)
- **Individual differences**가 큼 (전문가 vs 초보자)
- **Cognitive constraints**를 받음 (시간, 작업 기억 등)

### 이론적 배경

1. **van Opheusden et al. (2023)**:
   - Expertise ↔ deeper planning (전문가는 더 깊게 계획)
   - 4-in-a-row 게임에서 검증

2. **Yao et al. (2024)**:
   - Planning horizon은 latent confounder
   - 이를 무시하면 reward identifiability 깨짐

3. **Mhammedi et al. (2023)**:
   - Multi-step inverse 관점
   - Planning을 explicit multi-step factor로 모델링

### 본 연구의 차별점

**기존**: Planning depth를 implicit하게 가정
**본 연구**: Planning을 **explicit, inferable, manipulable mechanism**으로 다룸

→ Expertise/clinical variability 설명 + IRL interpretability 개선

---

## 💡 핵심 가정 및 가설

### 가정

**A1. Planning as discrete parameter**
- Planning depth h ∈ {1, 2, 3, 4, 5}로 이산화 가능
- 각 개인은 선호하는 planning depth 존재

**A2. Planning-reward separability**
- Reward function과 planning mechanism은 독립적
- 동일한 reward에 대해 다른 planning depth 사용 가능

**A3. Identifiability**
- Planning depth, inverse temperature β, lapse rate는 식별 가능
- 적절한 regularization과 데이터로 분리 가능

### 가설

**H1. Expertise discrimination** (Primary)
- Planning depth h가 expertise (novice vs expert)를 유의미하게 구별
- 예상: Expert → higher h

**H2. Incremental value over baseline**
- Planning-aware model이 baseline (parameter-only)보다 예측력 높음
- 측정: AUC, log-likelihood improvement

**H3. Planning quality matters**
- Planning의 "깊이"보다 "효율성"이 중요
- Pruning threshold와 depth의 상호작용

**H4. Clinical relevance** (Exploratory)
- Anxiety/disorder severity ↔ planning parameters
- Planning depth가 clinical traits 설명에 기여

**H5. Neural correlates** (Exploratory)
- Planning parameters ↔ fMRI activity (e.g., PFC, striatum)
- Trial-wise regressors로 neural signatures 발견

---

## 📊 연구 대상 및 지표

### 데이터

**출처**: van Opheusden et al. (2023) 4-in-a-row dataset

**규모**:
- **참가자**: 40명 (human-vs-human)
- **Trials**: 67,331 trials
- **실험 조건**: learning, time pressure, eye tracking, generalization, fMRI

**Cross-validation**: 5-fold (참가자별)

### 주요 변수

#### 독립 변수 (Planning parameters)
- **h**: Planning depth (1-5 steps)
- **β**: Inverse temperature (choice stochasticity)
- **lapse**: Random choice probability
- **Pruning threshold**: Search tree pruning criterion
- **Feature drop rate**: Feature omission rate

#### 종속 변수 (Behavioral outcomes)
- **Choice accuracy**: Log-likelihood of observed actions
- **Response time**: Decision latency
- **Win rate**: Game outcome (Elo rating)

#### 분류 지표 (Expertise)
- **Composite score**:
  ```
  z(log-likelihood) + z(pruning threshold) - z(lapse rate)
  ```
- **Binary label**: Median split → Expert (1) vs Novice (0)

#### 성능 지표
- **Model comparison**: AIC, BIC, log-likelihood
- **Discrimination**: AUC-ROC, accuracy, confusion matrix
- **Correlation**: Pearson r, Spearman ρ

### 평가 기준

**Baseline**: Parameters only (pruning, lapse, log-likelihood)
**Target**: + Planning depth h

**Success criteria**:
- ΔLL (log-likelihood increase) > 0.05 bits/trial
- ΔAUC > 0.05
- p < 0.05 for h coefficient in logistic regression

---

## 🔬 연구 방법론

### Phase 1: Behavioral Modeling (진행 중)

#### Step 1.1: Data exploration ✅
- Raw data: 67K trials, 40 participants
- Model fits: 22 variants (main, ablations, alternatives)
- Response time distributions

#### Step 1.2: Expertise classification ✅
- Composite score from log-likelihood, pruning, lapse
- Binary label: 20 Expert, 20 Novice

#### Step 1.3: Planning depth analysis ⚠️ (불확실)
- `depth_by_session.txt` 사용 (30명 × 5 sessions)
- **문제**: 정확히 PV depth인지 불명확
- **발견**: Expert < Novice (역설적 결과, p=0.01)

#### Step 1.4: Discrimination test ✅
- Logistic regression: parameters → expertise
- **Baseline AUC**: 0.982 (거의 완벽)
- **With PV depth**: 0.987 (미세 개선)

#### Step 1.5: Model comparison ✅
- MCTS > No pruning > Main model > Fixed-depth
- **해석**: Stochastic stopping이 제한적일 수 있음

### Phase 2: Planning-Aware Modeling (예정)

#### Step 2.1: Fixed-h model implementation
```cpp
// Modify BFS to fix depth at h ∈ {1,2,3,4,5}
class heuristic_fixed_h : public heuristic {
    int fixed_depth;
    zet makemove_bfs_fixed_h(board, bool);
};
```

#### Step 2.2: Parameter fitting
- MATLAB BADS optimizer
- For each participant i, each h:
  - Optimize (β, lapse) given h
  - Compute log-likelihood
  - Select best h via AIC/BIC

#### Step 2.3: Model selection
- Compare h=1,2,3,4,5 per participant
- Aggregate: optimal h distribution
- Test: h ~ expertise

### Phase 3: AIRL Extension (예정)

#### Planning-aware AIRL algorithm
```python
for h in H:
    initialize reward_network r_φ
    initialize planner_policy π_θ(s|h)  # constrained by h

    repeat:
        # Discriminator: real vs fake trajectories
        φ ← update_discriminator(expert_trajs, rollouts(π_θ))

        # Generator: match expert under inferred reward
        θ ← update_planner(π_θ, reward=r_φ)

    score ← evaluate(r_φ, π_θ)

return best (h, r_φ, π_θ)
```

#### Evaluation
- Likelihood / imitation score
- OOD generalization (new boards)
- Turing test realism

### Phase 4: Clinical & Neural (탐색적)

- **Clinical**: Anxiety → planning parameters → behavior
- **fMRI**: Trial-wise regressors (value, uncertainty, planning proxy)
- **Individual differences**: Parameters ↔ ROI activity

---

## 📈 진행 현황

### ✅ 완료된 작업

#### 1. 환경 설정 (2024-12-17)
- [x] GitHub 저장소 clone (`xRL_pilot/`)
- [x] 폴더 구조 파악 (`FOLDER_STRUCTURE.md`)
- [x] CLAUDE.md 업데이트 (코드베이스 구조 추가)

#### 2. 데이터 재분석 (2024-12-17)
- [x] `data_reanalysis.py`: 기본 통계, 파라미터 분포
  - 40명, 67K trials
  - Expertise 복합 지표 생성 (z-score 기반)
  - 시각화 7개 생성

- [x] `model_comparison_analysis.py`: 모델 변형 비교
  - 8개 모델 log-likelihood 비교
  - MCTS (2.00) > Main (1.95) > Fixed-depth (1.94)
  - 참가자별 모델 선호도 분석

- [x] `immediate_analysis.py`: 즉시 분석
  - PV depth vs expertise: **Expert < Novice** (p=0.01) ⚠️
  - Discrimination: AUC 0.982 (baseline), 0.987 (with depth)
  - RT correlations: depth ↑ → RT ↑, LL ↓

#### 3. 검증 작업 (2024-12-17)
- [x] `verify_depth_variable.py`: depth 변수 검증
  - Raw vs -2 corrected: 방향 동일
  - depth_by_session.txt의 정체 불명확
  - 상관관계는 이론적으로 타당

### 🚧 진행 중인 작업

#### 1. Planning depth 정체 확인
- [ ] `compute_planning_depth` 바이너리 실행
- [ ] 원본 Peak 데이터 찾기 (`splits/` 디렉토리)
- [ ] PV depth 재계산 및 비교

#### 2. 참가자 매칭
- [ ] opendata 40명 ↔ depth 30명 대응 확인
- [ ] Learning notebook 150명과의 관계 파악

### 📝 대기 중인 작업

#### Phase 2: Fixed-h modeling
- [ ] C++ 코드 수정 (`heuristic_fixed_h` 클래스)
- [ ] MATLAB 피팅 파이프라인 수정
- [ ] 각 h별 log-likelihood 계산
- [ ] Optimal h distribution 분석

#### Phase 3: AIRL
- [ ] Python wrapper 환경 구축 (`dm_env`)
- [ ] 4-in-a-row 환경 래핑
- [ ] AIRL discriminator 구현 (PyTorch)
- [ ] Planning-constrained policy 구현

#### Phase 4: Clinical/fMRI
- [ ] fMRI 데이터 확인 및 전처리
- [ ] Trial-wise regressors 생성
- [ ] GLM analysis

---

## 🔍 주요 발견

### 1. 역설적 Planning Depth 패턴 ⚠️

**예상**: Expert → deeper planning (van Opheusden, 2023)

**실제**:
```
Expert:  PV depth = 6.23 ± 1.30
Novice:  PV depth = 7.29 ± 0.55
p = 0.011 (유의미)
```

**해석**:
- ❌ Expertise ≠ simply "deeper planning"
- ✅ Expertise = **efficient planning** (적은 depth로 좋은 결과)
- Novice는 비효율적으로 깊게 탐색하지만 성능 나쁨

**증거**:
- Depth ↑ → Log-likelihood ↓ (r = -0.50, p < 0.01)
- Depth ↑ → Response time ↑ (r = +0.36, p < 0.05)
- Expert: 높은 pruning threshold (효율적 가지치기)

### 2. 거의 완벽한 Baseline Discrimination

**기존 파라미터만으로도 expertise 구별 거의 완벽**:
- AUC = 0.982 (parameters only)
- AUC = 0.987 (+ PV depth)
- Δ = 0.005 (미미한 개선)

**Feature importance**:
1. Log-likelihood (+1.76) ← 가장 중요
2. Pruning threshold (+1.46)
3. PV depth (-0.59) ← 음수! (깊을수록 Novice)
4. Lapse rate (-0.37)

**함의**:
- PV depth 추가해도 성능 개선 미미
- **하지만**: depth의 coefficient 방향이 중요한 정보 제공
- Planning depth를 단독으로가 아니라 **interaction term**으로 모델링 필요

### 3. 모델 변형 비교

**Log-likelihood 순위** (높을수록 좋음):
1. MCTS (2.00)
2. No pruning (2.00)
3. No feature drop (2.00)
4. Main model (1.95)
5. Fixed depth (1.94)

**해석**:
- Stochastic stopping (gamma parameter)이 너무 제한적
- Feature dropping이 오히려 성능 저하 유발
- Fixed depth의 성능 저하: 유연성 부족

### 4. Response Time 패턴

**주요 상관관계**:
- RT ↔ Median RT: r = +0.78 (당연)
- RT ↔ PV depth: r = +0.36 (깊게 탐색 = 느림)
- RT ↔ Log-likelihood: r = -0.19 (좋은 성능 = 빠름)

**Planning depth와 다른 변수**:
- Depth ↔ Log-likelihood: r = -0.50 ⚠️ (깊게 탐색 = 나쁜 성능)
- Depth ↔ Center weight: r = +0.55
- Depth ↔ RT: r = +0.36

---

## 🚀 다음 단계

### 즉시 실행 (1-2주)

#### 1. Planning Depth 검증 (최우선)
```bash
# C++ 바이너리 실행
cd "xRL_pilot/Model code"
./compute_planning_depth --data ../data_hvh.txt --output pv_depth_test.txt

# Python으로 비교
python compare_depth_files.py
```

**목표**: depth_by_session.txt가 정확히 무엇인지 확인

#### 2. Learning Trajectory 분석
```python
# 동일 참가자의 초기 vs 후기 게임 비교
for participant in participants:
    early_games = games[0:20]
    late_games = games[80:100]

    compare_pv_depth(early, late)
    # 예상: U-shaped? 초기↑ → 중기↑ → 후기↓ (효율화)
```

**목표**: Experience에 따른 planning depth 변화 추적

#### 3. Pruning Efficiency Metric
```python
efficiency = log_likelihood / pv_depth
# "적은 탐색으로 좋은 결과" = 높은 효율성
```

**가설**: Expert의 efficiency > Novice

### 단기 목표 (1개월)

#### 1. Fixed-h Model 구현
- [ ] `heuristic_fixed_h.cpp` 작성
- [ ] Compile 및 테스트
- [ ] MATLAB wrapper 수정

#### 2. Parameter Fitting
- [ ] 각 참가자별 h ∈ {1,2,3,4,5} 피팅
- [ ] Optimal h distribution
- [ ] h ~ expertise 검정

#### 3. Interaction Term Modeling
```python
# Planning depth × pruning quality
model = LogisticRegression()
X = pd.DataFrame({
    'h': planning_depth,
    'pruning': pruning_threshold,
    'h_x_pruning': planning_depth * pruning_threshold  # interaction
})
model.fit(X, expertise_label)
```

### 중기 목표 (2-3개월)

#### 1. Planning-Aware AIRL
- [ ] Python wrapper 환경 구축
- [ ] AIRL baseline 구현
- [ ] Planning-constrained policy
- [ ] Toy problem 검증

#### 2. Parameter Recovery Simulation
```python
# Ground truth h로 synthetic data 생성
for h_true in [1,2,3,4,5]:
    simulate_trajectories(h=h_true, ...)
    h_recovered = fit_model(trajectories)
    recovery_rate = (h_recovered == h_true).mean()
```

**목표**: Identifiability 검증

### 장기 목표 (6개월+)

#### 1. Clinical Extension
- [ ] Clinical trait 데이터 수집 설계
- [ ] Anxiety/disorder severity 측정
- [ ] Planning parameters ↔ clinical traits

#### 2. fMRI Analysis
- [ ] Trial-wise regressors (value, uncertainty, planning)
- [ ] GLM analysis
- [ ] ROI-based correlations

---

## 📚 참고 자료

### 논문

1. **van Opheusden, B., et al. (2023)**. "Expertise increases planning depth in human gameplay." *Nature*.
   - 위치: `papers/` (찾기)
   - 핵심: Expertise ↔ deeper planning, BFS with heuristic evaluation

2. **Yao, W., et al. (2024)**. "Planning horizon as a latent confounder in IRL."
   - 위치: `papers/Yao(2024)_IRLandPlanning.pdf`
   - 핵심: Planning horizon → reward identifiability

3. **Mhammedi, Z., et al. (2023)**. "RL for multi-step inverse kinematics."
   - 위치: `papers/Mhammedi(2023)_RLmultiInvKinematics.pdf`
   - 핵심: Multi-step inverse perspective

### 코드베이스

- **GitHub**: https://github.com/haba6030/xRL_pilot
- **핵심 파일**:
  - `Model code/bfs.cpp`: BFS + `get_depth_of_pv()`
  - `Model code/heuristic.cpp`: 17 feature weights
  - `Model code/matlab wrapper/fit_model.m`: BADS fitting
  - `Analysis notebooks/learning.ipynb`: PV depth 분석

### 내부 문서

- **CLAUDE.md**: 연구 계획 + 코드 구조
- **FOLDER_STRUCTURE.md**: 전체 폴더 구조
- **이 문서**: 프로젝트 종합 요약

---

## 🤝 팀원 온보딩 가이드

### 신규 팀원을 위한 체크리스트

#### Day 1: 환경 설정
- [ ] GitHub 저장소 clone
- [ ] Python 환경 설정 (`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`)
- [ ] Jupyter notebook 실행 확인
- [ ] 문서 읽기: README.md → FOLDER_STRUCTURE.md → 이 문서

#### Day 2-3: 데이터 탐색
- [ ] `opendata/` 파일들 확인
- [ ] `data_reanalysis.py` 실행 및 출력 확인
- [ ] `immediate_analysis.py` 실행 및 결과 해석
- [ ] 생성된 PNG 파일들 확인

#### Day 4-5: 코드 이해
- [ ] `xRL_pilot/Model code/` C++ 코드 훑어보기
- [ ] `bfs.cpp`의 `get_depth_of_pv()` 이해
- [ ] `heuristic.h` 파라미터 구조 파악
- [ ] `Analysis notebooks/learning.ipynb` 실행 (가능하다면)

#### Week 2: 실습
- [ ] 새로운 분석 아이디어 구현
- [ ] 기존 스크립트 수정 및 확장
- [ ] 첫 팀 미팅에서 질문 및 논의

### 자주 묻는 질문 (FAQ)

**Q1: depth_by_session.txt가 정확히 무엇인가요?**
- A: 현재 불명확합니다. 30명 참가자의 planning depth 관련 지표이지만, 정확히 PV depth인지 다른 metric인지 확인 중입니다.

**Q2: 왜 Expert가 Novice보다 planning depth가 낮나요?**
- A: 예상과 반대되는 결과입니다. 해석은 "Expert는 효율적으로 얕게 탐색"입니다. Pruning을 잘해서 불필요한 깊은 탐색을 하지 않습니다.

**Q3: Baseline AUC가 이미 0.98인데 개선 여지가 있나요?**
- A: 직접적인 discrimination 성능 개선은 어렵지만, planning depth의 **해석 가능성**과 **interaction effect** 분석에 의의가 있습니다. 또한 AIRL에서 reward identifiability 개선이 목표입니다.

**Q4: C++ 코드를 수정해야 하나요?**
- A: Phase 2에서 `heuristic_fixed_h` 클래스를 추가해야 합니다. C++ 경험이 있다면 도움이 되지만, 기존 코드 패턴을 따라하면 됩니다.

**Q5: MATLAB이 필요한가요?**
- A: Parameter fitting에 MATLAB + BADS optimizer가 사용되지만, Python으로 대체 가능합니다 (scipy.optimize, GPyOpt 등).

---

## 📧 연락처 및 협업

**프로젝트 리드**: [이름]
**GitHub**: https://github.com/haba6030/xRL_pilot
**문서 업데이트**: 주요 발견이나 방법 변경 시 이 문서를 업데이트해주세요.

**버전 관리**:
- 분석 스크립트는 날짜별로 백업 (`analysis_YYYYMMDD.py`)
- 주요 결과는 `results/` 디렉토리에 저장
- Git commit 메시지에 진행 상황 명시

---

**마지막 업데이트**: 2024-12-17
**다음 리뷰 예정**: Phase 2 시작 시 (Fixed-h model 구현 후)
