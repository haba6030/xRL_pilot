# 문서 품질 검토 및 개선 방안

**검토일**: 2025-12-25
**목적**: 연구실 구성원들이 프로젝트를 이해하고 논의하기 위한 문서 정리
**대상**: GitHub/Notion 공유용 문서

---

## 📋 현재 문서 현황

### 총 19개 문서 존재

**문제점**:
- ❌ 너무 많은 문서 (19개) → 어디서 시작해야 할지 불명확
- ❌ 중복된 내용 (AIRL_COMPLETE_GUIDE vs GYMNASIUM_AND_AIRL_GUIDE)
- ❌ 일부 outdated 내용 (OPTION_A_VS_B, OPTION_DIFFERENCE_SIMPLE)
- ❌ 명확한 문서 계층 구조 부재
- ❌ 신규 멤버를 위한 시작점 불명확

---

## ✅ 권장 문서 구조 (3-Tier)

### 📌 Tier 1: 시작 문서 (필수 읽기)

**목적**: 프로젝트 전체 개요 파악 (15-20분)

```
1. README.md                    # 프로젝트 소개, Quick Start
   ├─ 연구 목적 및 배경
   ├─ 핵심 아이디어 (Planning-Aware IRL)
   ├─ 주요 결과 (현재까지)
   └─ 다음 단계로 가는 링크

2. PROJECT_OVERVIEW.md (신규 생성 권장)
   ├─ 연구 질문 (Research Questions)
   ├─ 접근 방법 (Methodology Overview)
   ├─ 데이터 (4-in-a-row expert data)
   ├─ 주요 발견 (Key Findings)
   └─ 현재 상태 (Current Status)
```

### 📚 Tier 2: 핵심 문서 (깊이 이해)

**목적**: 구현 및 이론 이해 (1-2시간)

```
3. PLANNING_DEPTH_PRINCIPLES.md    # 핵심 원칙 (이론)
   └─ h는 POLICY에만 존재, reward는 depth-agnostic

4. IMPLEMENTATION_SUMMARY.md        # 구현 요약 (Steps A-E)
   ├─ Step A: Training data generation
   ├─ Step B: Behavior Cloning
   ├─ Step C: PPO generator
   ├─ Step D: Reward network
   └─ Step E: AIRL training

5. AIRL_DESIGN.md                   # 설계 문서
   ├─ Environment 구조
   ├─ State/Action representation
   ├─ AIRL 적용 전략
   └─ 학습 절차 (실제 구현 반영)

6. IMPLEMENTATION_NOTES.md          # 기술 세부사항
   ├─ Library 버전 및 설정
   ├─ API 사용법 (imitation, SB3)
   ├─ 주요 이슈 및 해결책
   └─ Training metrics 해석
```

### 🔧 Tier 3: 참고 문서 (필요시)

**목적**: 특정 주제 깊이 파기

```
7. PHASE2_PROGRESS.md               # 진행 상황 (실시간 업데이트)
8. GYMNASIUM_AND_AIRL_GUIDE.md      # Gymnasium 환경 이해
9. DEPTH_INTEGRATION_DETAILED.md    # Depth integration 상세
10. PHASE2_VALIDATION_CHECKLIST.md  # 검증 프로토콜
```

### 🗑️ 삭제/통합 권장

```
❌ AIRL_COMPLETE_GUIDE.md          → AIRL_DESIGN.md와 중복
❌ OPTION_A_VS_B.md                 → Outdated (BC approach 선택됨)
❌ OPTION_DIFFERENCE_SIMPLE.md      → Outdated
❌ IMPLEMENTATION_STATUS.md         → IMPLEMENTATION_SUMMARY.md와 중복
❌ RESPONSE_TO_FEEDBACK.md          → 필요시 별도 폴더로 이동
❌ RESEARCH_DISCUSSION.md           → 필요시 별도 폴더로 이동
❌ DEPTH_VARIABLE_VERIFICATION.md   → 검증 완료, 아카이브
❌ FOLDER_STRUCTURE.md              → README에 통합
```

---

## 📂 권장 폴더 구조

```
xRL_pilot/
├── README.md                          # 프로젝트 진입점
├── PROJECT_OVERVIEW.md                # 연구 개요 (신규 생성)
│
├── docs/                              # 핵심 문서
│   ├── 1_PRINCIPLES.md               # 핵심 원칙 (← PLANNING_DEPTH_PRINCIPLES.md)
│   ├── 2_DESIGN.md                   # 설계 (← AIRL_DESIGN.md)
│   ├── 3_IMPLEMENTATION.md           # 구현 (← IMPLEMENTATION_SUMMARY.md)
│   └── 4_TECHNICAL_NOTES.md          # 기술 (← IMPLEMENTATION_NOTES.md)
│
├── progress/                          # 진행 상황
│   ├── CURRENT_STATUS.md             # 현재 상태 (← PHASE2_PROGRESS.md)
│   └── VALIDATION_CHECKLIST.md       # 검증 (← PHASE2_VALIDATION_CHECKLIST.md)
│
├── guides/                            # 상세 가이드
│   ├── gymnasium_guide.md            # (← GYMNASIUM_AND_AIRL_GUIDE.md)
│   └── depth_integration.md          # (← DEPTH_INTEGRATION_DETAILED.md)
│
├── archive/                           # 아카이브
│   ├── design_options/               # 설계 옵션 논의
│   │   ├── option_a_vs_b.md
│   │   └── option_difference.md
│   └── old_discussions/              # 과거 논의
│       ├── research_discussion.md
│       └── response_to_feedback.md
│
├── fourinarow_airl/                   # 코드
│   ├── README.md                     # 코드 사용법
│   ├── generate_training_data.py
│   ├── train_bc.py
│   ├── create_ppo_generator.py
│   ├── create_reward_net.py
│   └── train_airl.py
│
└── CLAUDE.md                          # Claude 지시사항 (유지)
```

---

## 📝 각 문서별 개선 사항

### 1. README.md (신규 작성 필요)

**현재 문제**: 기존 README가 없거나 outdated

**권장 구조**:
```markdown
# Planning-Aware AIRL for 4-in-a-Row

## 🎯 연구 목적
Planning depth를 explicit하게 모델링하여 IRL의 reward identifiability 향상

## 🔑 핵심 아이디어
- Planning depth h는 **POLICY에만** 존재
- Reward network는 **완전히 depth-agnostic**
- h별 학습 후 비교 → 최적 h 추정

## 📊 현재 상태
- ✅ Steps A-E 완료 (71%)
- 📝 Step F 준비 중 (Multi-depth comparison)
- 8/8 validation checkpoints passed

## 🚀 Quick Start
1. Environment setup
2. Run test: `python3 train_airl.py --test`
3. Full training: `python3 train_airl.py --total_timesteps 50000`

## 📚 문서
- [연구 개요](PROJECT_OVERVIEW.md) - 시작은 여기서
- [핵심 원칙](docs/1_PRINCIPLES.md) - 이론적 배경
- [구현 요약](docs/3_IMPLEMENTATION.md) - 코드 이해

## 👥 Team
- PI: [이름]
- Contributors: [이름들]

## 📄 References
- van Opheusden et al. (2023) - 4-in-a-row expertise
- Yao et al. (2024) - Planning horizon in IRL
```

---

### 2. PROJECT_OVERVIEW.md (신규 생성)

**목적**: 연구 전체 그림 제공

```markdown
# 연구 개요: Planning-Aware AIRL

## 1. 연구 질문

**Q1**: Planning depth가 다른 사람들의 행동을 같은 reward function으로 설명할 수 있는가?

**Q2**: Planning depth를 explicit하게 모델링하면 IRL의 reward identifiability가 향상되는가?

**Q3**: 4-in-a-row expert들의 최적 planning depth는?

## 2. 데이터

- **출처**: van Opheusden et al. (2023)
- **내용**: 40명 참가자, 67K trials
- **게임**: 4-in-a-row (6×6 board)
- **특징**: Expert vs Novice, Elo rating 있음

## 3. 방법론

### 3.1 기존 접근 (van Opheusden)
- BFS + heuristic weights
- Planning depth h는 고정 parameter

### 3.2 우리 접근 (Planning-Aware IRL)
- h별로 **별도 generator** 학습 (h ∈ {1,2,4,8})
- **공통 reward network** (depth-agnostic)
- AIRL로 adversarial learning

### 3.3 핵심 원칙
**h는 POLICY에만 존재, Reward는 depth-agnostic**

| Component | h 존재? |
|-----------|---------|
| DepthLimitedPolicy | ✅ YES |
| Observations | ❌ NO |
| Reward Network | ❌ NO |

## 4. 구현 (Steps A-E)

```
A. Generate h-specific training data
   └─ DepthLimitedPolicy(h) → trajectories

B. Behavior Cloning
   └─ Neural network mimics h-specific behavior

C. Wrap BC with PPO
   └─ BC policy → PPO generator

D. Create reward network
   └─ Depth-agnostic discriminator

E. AIRL training
   └─ h-specific generator + depth-agnostic reward
```

## 5. 주요 결과 (현재까지)

- ✅ 모든 validation checkpoints 통과 (8/8)
- ✅ AIRL training 성공적으로 완료
- 📊 Multi-depth comparison 진행 중

## 6. 다음 단계

1. All depths training (h=1,2,4,8)
2. Discriminator metrics 비교
3. Best h 선택
4. Expert behavior 분석

## 7. 참고 문헌

- van Opheusden et al. (2023). Expertise increases planning depth in human gameplay. *Nature*
- Yao et al. (2024). Planning in Inverse Reinforcement Learning
```

---

### 3. AIRL_DESIGN.md 개선

**현재 문제**: 일부 초기 설계와 실제 구현이 혼재

**개선 방안**:
```markdown
# AIRL 설계 문서

## ⚠️ 주의
- 이 문서는 **초기 설계**를 포함합니다
- **실제 구현**은 일부 다릅니다 (Section C 참조)
- 최신 구현은 IMPLEMENTATION_SUMMARY.md 참조

## 목차
1. Pedestrian 프로젝트 분석 (참고용)
2. 4-in-a-row 적용 가능성
3. 제안 설계
4. Planning-Aware AIRL (이론)
5. 구현 로드맵
6. 예상 결과
7. 미해결 이슈
8. 최종 판단
9. Implementation Notes → IMPLEMENTATION_NOTES.md 참조

## Section C: 학습 절차 (Actual Implementation)

**주요 변경사항**:
| 설계 | 실제 구현 |
|------|-----------|
| BFS Distillation | BC (Behavior Cloning) |
| h-conditioned reward | Completely depth-agnostic |

[실제 코드 예시...]
```

---

### 4. IMPLEMENTATION_SUMMARY.md 개선

**현재**: 좋음, 약간의 구조 개선만 필요

**개선**:
```markdown
# 구현 요약

**Status**: 5/7 steps complete (71%)
**Checkpoints**: 8/8 passed ✅

## 📖 빠른 이해

### 전체 Pipeline (한눈에)

```
Expert Data (h=?)
    ↓
┌─────────────────────────────────┐
│  For each h ∈ {1, 2, 4, 8}      │
│                                   │
│  Step A: DepthLimitedPolicy(h)   │
│     ↓                             │
│  Step B: BC → Neural Policy      │
│     ↓                             │
│  Step C: BC → PPO Generator      │
│                                   │
└─────────────────────────────────┘
    ↓
Step D: Depth-AGNOSTIC Reward Net
    ↓
Step E: AIRL Training
    ├─ Generator (h-specific)
    └─ Discriminator (depth-agnostic)
    ↓
Step F: Compare h values
    ↓
Step G: Best h selection
```

### 핵심 원칙 (다시 강조)

✅ h는 POLICY에만
❌ h는 REWARD에 없음

[나머지 기존 내용...]
```

---

### 5. IMPLEMENTATION_NOTES.md 개선

**현재**: 매우 좋음, 소폭 개선

**추가 권장**:
```markdown
## 목차
1. [Environment Setup](#1-environment-setup)
2. [BasicRewardNet Usage](#2-basicrewardnet-usage)
3. [imitation 1.0.1 API](#3-imitation-101-api)
4. [Architecture Matching](#4-architecture-matching)
5. [Data Formats](#5-data-formats)
6. [AIRL Metrics](#6-airl-training-metrics) ⭐ 중요!
7. [Common Issues](#9-common-issues-and-solutions)
8. [Training Tips](#10-training-recommendations)

## 6. AIRL Training Metrics ⭐

### Discriminator Metrics 해석 (매우 중요!)

**❌ 잘못된 해석**:
```
disc_acc_expert = 1.0  → "완벽하게 학습됨!"
disc_acc_gen = 0.0     → "완벽하게 학습됨!"
```

**✅ 올바른 해석**:
```python
# Well-trained (목표)
disc_acc_expert ≈ 0.5  # Generator가 discriminator를 속임
disc_acc_gen ≈ 0.5     # 좋은 imitation
disc_acc ≈ 0.5         # Balanced

# Undertrained (현재)
disc_acc_expert = 1.0  # Discriminator가 너무 강함
disc_acc_gen = 0.0     # Generator가 약함
→ total_timesteps 증가 필요!
```

### 시각화로 이해하기

```
Training Progress:
─────────────────────────────────────
Early     disc_acc_expert: 1.0 ━━━━━━━━━━
(Bad)     disc_acc_gen:    0.0

Mid       disc_acc_expert: 0.7 ━━━━━━━
          disc_acc_gen:    0.3 ━━━

Good      disc_acc_expert: 0.5 ━━━━━
(Target)  disc_acc_gen:    0.5 ━━━━━
─────────────────────────────────────
```
```

---

### 6. PLANNING_DEPTH_PRINCIPLES.md

**현재**: 이론적으로 잘 정리됨

**추가 권장**: 시각적 예시
```markdown
## 핵심 원칙 (Visual)

### ❌ 잘못된 접근 (h를 reward에 넣음)

```
┌─────────────────────────────────┐
│  Discriminator (Reward Network) │
│                                  │
│  Input: (state, action, h)      │  ← h 포함 (잘못됨!)
│  Output: reward                  │
└─────────────────────────────────┘
```

### ✅ 올바른 접근 (h는 policy에만)

```
┌─────────────────────────────────┐
│  Generator (Policy)              │
│                                  │
│  DepthLimitedPolicy(h=2)         │  ← h는 여기만!
│  Input: state (89-dim)           │
│  Output: action                  │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  Discriminator (Reward Network) │
│                                  │
│  Input: (state, action)          │  ← h 없음!
│  Output: reward                  │
└─────────────────────────────────┘
```

### 왜 이렇게 해야 하는가?

**이유 1: Reward Identifiability**
- h를 reward에 넣으면 confounding
- 같은 behavior를 다른 (h, reward) 조합으로 설명 가능

**이유 2: Generalization**
- Depth-agnostic reward는 모든 h에 적용 가능
- Transfer learning 용이

**이유 3: Interpretability**
- Reward는 "무엇이 좋은가" (what)
- Planning depth는 "어떻게 생각하는가" (how)
```

---

## 🎨 GitHub/Notion 최적화

### GitHub용 개선사항

1. **README에 Badges 추가**
```markdown
![Progress](https://img.shields.io/badge/Progress-71%25-green)
![Checkpoints](https://img.shields.io/badge/Checkpoints-8%2F8-success)
![Status](https://img.shields.io/badge/Status-Active-blue)
```

2. **Wiki 활용**
- Main docs → Wiki로 이동
- README는 간결하게 유지

3. **Issues/Projects 활용**
- Step F, G를 Issues로 tracking
- Project board로 진행상황 시각화

### Notion용 개선사항

1. **Database 구조**
```
📊 Progress Tracker
├─ Steps (A-G)
│  ├─ Status (Done/In Progress/Pending)
│  ├─ Files
│  └─ Checkpoints
├─ Validation Checkpoints (1-8)
└─ Metrics (disc_acc 등)
```

2. **Toggle Blocks 활용**
```
▶ Step A: Generate Training Data
  └─ [상세 내용...]

▶ Step B: Behavior Cloning
  └─ [상세 내용...]
```

3. **Callout Boxes**
```
💡 핵심 원칙
h는 POLICY에만 존재!

⚠️ 주의사항
disc_acc = 0.5가 목표!

✅ 완료
All checkpoints passed
```

---

## 🚀 실행 권장사항

### Phase 1: 문서 정리 (우선순위)

**즉시 실행**:
1. ✅ README.md 새로 작성
2. ✅ PROJECT_OVERVIEW.md 생성
3. ✅ 폴더 구조 정리 (`docs/`, `progress/`, `archive/`)
4. ✅ 중복/outdated 문서 삭제

**다음 주**:
5. 각 핵심 문서에 목차 및 내부 링크 추가
6. 시각적 다이어그램 추가 (원칙, pipeline)
7. GitHub Wiki 설정

### Phase 2: 접근성 향상

1. **신규 멤버 Onboarding 가이드** 작성
   - 30분 Quick Start
   - 1시간 Deep Dive
   - 첫 실험 실행하기

2. **FAQ 섹션** 추가
   - "h를 reward에 넣으면 안 되는 이유?"
   - "disc_acc = 0.5가 왜 좋은가?"
   - "BC vs BFS Distillation?"

3. **Troubleshooting Guide**
   - OpenMP 에러
   - Variable horizon 에러
   - Tensor dimension 에러

---

## 📊 문서 품질 체크리스트

### 필수 요소

각 문서는 다음을 포함해야 함:

- [ ] **명확한 제목 및 목적** (첫 3줄에)
- [ ] **최종 업데이트 날짜**
- [ ] **관련 문서 링크** (See also)
- [ ] **예제 코드** (해당 시)
- [ ] **시각적 요소** (다이어그램, 표)
- [ ] **현재 상태 표시** (✅/🔄/❌)

### 가독성 체크

- [ ] 제목 계층 구조 (H1 → H2 → H3)
- [ ] 코드 블록에 언어 명시
- [ ] 중요 내용은 Callout/Bold
- [ ] 너무 긴 문단 분리 (3-5줄)
- [ ] Technical term 한/영 병기

---

## 💡 최종 권장사항

### 우선순위 1 (이번 주 내)

1. **README.md 작성** - 프로젝트 진입점
2. **PROJECT_OVERVIEW.md 생성** - 연구 전체 그림
3. **폴더 정리** - docs/, progress/, archive/
4. **중복 문서 삭제** - 8개 문서 제거

### 우선순위 2 (다음 주)

5. AIRL_DESIGN.md에 "주의" 섹션 추가
6. IMPLEMENTATION_NOTES.md에 시각화 추가
7. 각 문서에 목차 추가
8. GitHub Wiki 설정

### 우선순위 3 (여유 있을 때)

9. Notion database 구축
10. Onboarding guide 작성
11. FAQ 섹션 작성
12. Video tutorial 제작 (optional)

---

**작성자**: Claude
**검토 필요**: 연구실 PI 및 주요 멤버
**다음 검토**: Step F 완료 후
