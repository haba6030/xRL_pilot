# depth h의 연구 목적 및 전체 프로세스

## ✅ 정확한 이해 확인

**귀하의 이해가 100% 맞습니다!**

---

## 🎯 depth h의 세 가지 연구 목적

### 1. **Expert Data 생성: 다양한 Planning Style 생성**

```python
# 깊게 보는 expert (전략적, 신중함)
expert_deep = generate_BFS_trajectories(h=8, num_episodes=100)
# → 장기 전략, 함정 설치, 수비적

# 얕게 보는 expert (충동적, 빠름)
expert_shallow = generate_BFS_trajectories(h=1, num_episodes=100)
# → 단기 이익, 즉각 반응, 공격적

# 중간 깊이 expert들
expert_h2 = generate_BFS_trajectories(h=2, ...)
expert_h4 = generate_BFS_trajectories(h=4, ...)
```

**목적**:
- 다양한 "planning style"을 가진 expert 생성
- 각 h는 다른 **인지 전략**을 표현
- 실제 human expert의 다양성을 시뮬레이션

---

### 2. **Performance Evaluation: Elo Rating으로 최적 h 찾기**

```python
# Step 1: 각 h로 policy 학습
policies = {}
for h in [1, 2, 4, 8]:
    expert_data = generate_BFS_trajectories(h=h)
    policies[h] = train_airl(expert_data)

# Step 2: Human expert와 비교 (Elo rating)
human_expert_elo = compute_elo(human_expert_data)

for h, policy in policies.items():
    policy_elo = compute_elo(policy)

    print(f"h={h}: Elo = {policy_elo}")
    print(f"  Distance from human: {abs(policy_elo - human_expert_elo)}")

# Step 3: 가장 가까운 h 선택
best_h = argmin(abs(policy_elo - human_expert_elo) for h, policy in policies.items())
print(f"Best matching depth: h={best_h}")
# → "Human expert는 대략 h={best_h} 정도로 계획하는 것 같다"
```

**목적**:
- **어떤 h가 실제 human expert를 가장 잘 모방하는가?**
- Elo rating으로 정량적 평가
- "Human expert의 planning depth 추정"

---

### 3. **Clinical Trait Mapping: 임상 특성에 따른 Planning Depth**

```python
# Phase 1: Human expert의 planning depth 추정 (위에서 함)
# 결과: 전문가들은 대략 h=4~6 정도

# Phase 2: Clinical trait별 planning depth 조사
participants = load_clinical_data()

for participant in participants:
    # 임상 측정
    anxiety_score = participant.anxiety_score
    expertise = participant.elo_rating

    # 행동 데이터
    behavior_data = participant.game_trajectories

    # Planning depth 추정
    estimated_h = estimate_planning_depth(behavior_data)

    # 관계 분석
    correlations[participant.id] = {
        'anxiety': anxiety_score,
        'expertise': expertise,
        'estimated_h': estimated_h
    }

# 분석
# "불안이 높을수록 planning depth가 낮은가?"
# "전문가일수록 planning depth가 높은가?"
plot_correlation(anxiety_scores, estimated_h_values)
plot_correlation(expertise_levels, estimated_h_values)
```

**목적**:
- **Clinical trait (불안, 우울 등)과 planning depth 연결**
- "불안한 사람은 얕게 계획하는가?"
- "전문가는 깊게 계획하는가?"
- **개인차를 planning mechanism으로 설명**

---

## 📊 전체 연구 프로세스 (순서대로)

### **Phase 1: Synthetic Expert 생성 및 AIRL 학습**

```
Step 1: 다양한 h로 BFS expert 생성
├── h=1 expert (shallow planning)
├── h=2 expert
├── h=4 expert
└── h=8 expert (deep planning)

Step 2: 각 h마다 AIRL 학습
├── Option A: Pure NN → policy_h1, policy_h2, ...
└── Option B: BC-initialized → policy_h1, policy_h2, ...

Step 3: 성능 평가
└── Elo rating, win rate, action distribution
```

### **Phase 2: Human Expert 분석**

```
Step 1: Human expert data 로드
└── opendata/raw_data.csv

Step 2: 각 h policy와 human 비교
├── Elo rating 계산
├── KL divergence
└── Behavioral similarity

Step 3: Best matching h 찾기
└── "Human expert ≈ h=?" 추정
```

### **Phase 3: Clinical Trait 분석**

```
Step 1: Participant-level analysis
├── 각 참가자의 planning depth 추정
└── Clinical trait 측정치 수집

Step 2: Correlation analysis
├── Anxiety ↔ Planning depth
├── Expertise ↔ Planning depth
└── Other traits ↔ Planning depth

Step 3: Mechanism explanation
└── "Clinical trait → Planning depth → Behavior"
   (not just "Clinical trait → Behavior")
```

---

## 🔬 구체적 연구 질문과 depth h

### **연구 질문 1: "Human expert는 얼마나 깊게 계획하는가?"**

**방법**:
```python
# 1. 각 h로 policy 학습
for h in [1, 2, 4, 6, 8, 10]:
    policy[h] = train_airl(expert_data=BFS(h))

# 2. Human expert와 비교
human_data = load_expert_trajectories('opendata/raw_data.csv')

for h, policy in policies.items():
    similarity[h] = compute_similarity(policy, human_data)

# 3. 최고 유사도 h 찾기
best_h = argmax(similarity)
print(f"Human expert planning depth: approximately h={best_h}")
```

**기대 결과**: "Human expert는 대략 h=4~6 정도로 계획"

---

### **연구 질문 2: "Planning depth가 expertise를 설명하는가?"**

**방법**:
```python
# 1. Expertise 분류 (Elo 기반)
experts = participants[elo > threshold_expert]
novices = participants[elo < threshold_novice]

# 2. 각 그룹의 planning depth 추정
expert_depths = [estimate_h(p.data) for p in experts]
novice_depths = [estimate_h(p.data) for p in novices]

# 3. 통계 검정
t_test(expert_depths, novice_depths)
# "Experts have significantly higher planning depth"
```

**기대 결과**: "Experts: h=6~8, Novices: h=2~4"

---

### **연구 질문 3: "불안이 planning depth에 영향을 주는가?"**

**방법**:
```python
# 1. Clinical data 수집
for participant in participants:
    anxiety = participant.anxiety_score  # e.g., STAI
    estimated_h = estimate_planning_depth(participant.data)

    data.append({
        'participant_id': participant.id,
        'anxiety': anxiety,
        'planning_depth': estimated_h
    })

# 2. Correlation 분석
correlation = pearsonr(anxiety_scores, planning_depths)

# 3. Regression 분석
model = LinearRegression()
model.fit(anxiety_scores, planning_depths)
# "1 SD increase in anxiety → -0.5 decrease in planning depth"
```

**기대 결과**: "높은 불안 → 낮은 planning depth (단기적 사고)"

---

## 🎯 depth h를 사용하는 이유 정리

### **1. Computational Mechanism 제공**

```
기존 접근:
  "전문가가 초보자보다 잘한다" (관찰)
  → 왜? 어떻게? (black box)

우리 접근:
  "전문가는 h=8로 계획, 초보자는 h=2로 계획"
  → Planning depth라는 구체적 mechanism 제공
```

### **2. Manipulable Variable**

```
기존:
  "이 사람은 전문가다" (고정된 레이블)

우리:
  "이 사람의 planning depth는 h=6이다"
  → h를 조작해서 다른 행동 시뮬레이션 가능
  → "만약 h=2로 계획했다면?" 반사실 추론
```

### **3. Explainable AI**

```
기존 IRL:
  "Expert의 reward function을 복원했습니다"
  → 하지만 reward가 복잡하면 해석 어려움

Planning-aware AIRL:
  "Expert는 이 reward를 h=6 depth로 최적화합니다"
  → Reward와 Planning을 분리
  → 더 해석 가능
```

---

## 📋 실험 체크리스트

### ✅ **Phase 1: Infrastructure (완료)**
- [x] Environment 구현
- [x] BFS wrapper
- [x] Data loader
- [x] AIRL pipeline (Option A & B)

### 🔄 **Phase 2: Depth Sweep (진행 예정)**
- [ ] h=1 expert 생성 및 AIRL 학습
- [ ] h=2 expert 생성 및 AIRL 학습
- [ ] h=4 expert 생성 및 AIRL 학습
- [ ] h=8 expert 생성 및 AIRL 학습
- [ ] Policy 간 비교 (action distribution, win rate)

### 🔄 **Phase 3: Human Expert Matching (진행 예정)**
- [ ] Human expert data 분석
- [ ] 각 h policy와 비교 (Elo rating)
- [ ] Best matching h 선택
- [ ] "Human planning depth" 추정

### 🔄 **Phase 4: Clinical Trait Analysis (진행 예정)**
- [ ] Participant-level planning depth 추정
- [ ] Anxiety ↔ Planning depth 분석
- [ ] Expertise ↔ Planning depth 분석
- [ ] Regression 모델 구축

### 🔄 **Phase 5: Neural Correlates (탐색적)**
- [ ] fMRI data 수집 (if available)
- [ ] Planning depth ↔ Brain activity 분석
- [ ] Individual differences in neural planning

---

## 🎓 핵심 통찰

### **Planning depth h는:**

1. **Synthetic expert 생성 도구**
   - 다양한 planning style 시뮬레이션
   - BFS(h=1) vs BFS(h=8)

2. **Human behavior 분석 도구**
   - "이 사람은 얼마나 깊게 계획하는가?"
   - Elo rating으로 정량적 평가

3. **Individual difference 설명 도구**
   - Expertise: planning depth로 설명
   - Clinical trait: planning depth로 매개

4. **Intervention 설계 도구**
   - "Planning depth를 늘리면 성능 향상?"
   - "불안 감소 → planning depth 증가?"

---

## 📝 요약 (한 문장씩)

1. **Expert 생성**: "h를 바꿔서 다양한 planning style expert를 만든다"
2. **성능 평가**: "Elo rating으로 어떤 h가 human expert와 가장 비슷한지 찾는다"
3. **Clinical 연결**: "나중에 clinical trait에 따라 planning depth를 조정한다"

---

## 🔗 다른 문서와의 연결

이 내용을 다음 문서들에 통합:
- ✅ `AIRL_COMPLETE_GUIDE.md` (전체 가이드에 목적 섹션 추가)
- ✅ `OPTION_A_DEPTH_H_EXPLAINED.md` (depth h 역할 설명)
- ✅ `CLAUDE.md` (프로젝트 전체 계획)

---

**귀하의 이해가 정확합니다!** 이것이 바로 planning-aware IRL의 핵심 아이디어입니다. 🎯
