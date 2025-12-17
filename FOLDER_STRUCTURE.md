# xRL_pilot 프로젝트 폴더 구조

프로젝트 전체 구조를 명확하게 파악하기 위한 문서입니다.

**마지막 업데이트**: 2024-12-17

---

## 📁 최상위 구조

```
xRL_pilot/
├── opendata/              # 실험 데이터 (CSV)
├── papers/                # 참고 논문 (PDF)
├── xRL_pilot/            # van Opheusden (2023) 코드베이스 (GitHub clone)
├── CLAUDE.md             # 프로젝트 가이드 (연구 계획 + 코드 구조)
├── FOLDER_STRUCTURE.md   # 이 파일
└── *.py                  # 데이터 분석 스크립트들
```

---

## 📊 `opendata/` - 실험 데이터

**용도**: van Opheusden (2023) 논문의 실험 데이터 및 모델 피팅 결과

### 파일 목록

#### 원본 행동 데이터
```
raw_data.csv                                    (7.5 MB)
└── 67,331 trials from 40 participants
    └── 컬럼: black_pieces, white_pieces, move, color,
             response_time, participant, cross-validation group,
             experiment, time limit, session number
```

#### 모델 피팅 결과 (22개 모델 변형)

**주요 모델**:
- `model_fits_main_model.csv` (189 KB) - **전체 feature 모델**
- `model_fits_mcts.csv` (182 KB) - Monte Carlo Tree Search 비교
- `model_fits_optimal_weights.csv` (145 KB) - Oracle weights

**Planning 관련 변형**:
- `model_fits_fixed_depth.csv` (186 KB)
- `model_fits_fixed_iterations.csv` (198 KB)
- `model_fits_fixed_branching.csv` (196 KB)

**Ablation 모델** (feature 제거 실험):
```
model_fits_no_pruning.csv               (179 KB)
model_fits_no_tree.csv                  (165 KB)
model_fits_no_feature_drop.csv          (187 KB)
model_fits_no_value_noise.csv           (188 KB)
model_fits_no_center.csv                (175 KB)
model_fits_no_connected_2-in-a-row.csv  (197 KB)
model_fits_no_unconnected_2-in-a-row.csv(202 KB)
model_fits_no_3-in-a-row.csv            (181 KB)
model_fits_no_4-in-a-row.csv            (184 KB)
model_fits_no_active_scaling.csv        (190 KB)
```

**Feature 변형**:
```
model_fits_orientation-dependent_weights.csv   (239 KB)
model_fits_orientation-dependent_dropping.csv  (238 KB)
model_fits_type-dependent_dropping.csv         (236 KB)
model_fits_tile_dropping.csv                   (194 KB)
model_fits_opponent_scaling.csv                (206 KB)
model_fits_triangle.csv                        (196 KB)
```

### 데이터 특성
- **참가자**: 40명
- **Cross-validation 그룹**: 1-5
- **실험 타입**: human-vs-human, learning, time pressure, eye tracking, generalization
- **파라미터**: pruning threshold, stopping probability, feature drop rate, lapse rate,
               active scaling constant, center weight, 2/3/4-in-a-row weights

---

## 📄 `papers/` - 참고 논문

```
Yao(2024)_IRLandPlanning.pdf            (1.7 MB)
└── Planning horizon as latent confounder in IRL

Mhammedi(2023)_RLmultiInvKinematics.pdf (726 KB)
└── Multi-step inverse perspective
```

---

## 🔬 `xRL_pilot/` - van Opheusden (2023) 코드베이스

**GitHub**: https://github.com/haba6030/xRL_pilot (forked)

### 하위 구조

```
xRL_pilot/
├── Model code/              # C++ 핵심 구현
├── Analysis notebooks/      # Jupyter 분석 노트북
├── Experiment code/         # 실험 웹 인터페이스
└── data_hvh.txt            # Human vs Human 게임 데이터
```

---

## 💻 `xRL_pilot/Model code/` - C++ 구현

**용도**: 4-in-a-row 게임 엔진 및 planning 알고리즘

### 핵심 파일

#### 게임 로직
```
board.h                     (9.2 KB)
└── 6×6 보드 표현 (uint64 bitboard)
    └── 36 positions, black/white pieces

board_list.h               (3.4 KB)
└── 보드 상태 리스트 관리
```

#### Planning 알고리즘
```
bfs.h / bfs.cpp            (1.3 KB / 4.5 KB)
└── Best-First Search
    ├── get_depth_of_pv()      # Principal Variation depth ⭐
    ├── get_mean_depth()       # 평균 탐색 depth
    └── get_sum_depth()        # 총 depth 합

dfs.cpp                    (4.7 KB)
└── Depth-First Search (비교용)

mcts.cpp / mcts.h
└── Monte Carlo Tree Search
```

#### Heuristic 평가
```
heuristic.h / heuristic.cpp    (11 KB)
└── 17 feature weights
    ├── center_weight
    ├── connected/unconnected 2-in-a-row
    ├── 3-in-a-row, 4-in-a-row
    ├── w_act[17], w_pass[17]  # Active/passive weights
    └── delta[17]              # Feature drop rates

features.cpp               (37 KB)
└── Feature extraction 구현

features_all.cpp          (37 KB)
└── 모든 feature variants
```

#### Planning Depth 계산 바이너리들 ⚠️
```
compute_planning_depth                 (실행 파일)
compute_planning_depth.cpp
compute_planning_depth_fixed_branch.cpp
compute_planning_depth_fixed_depth.cpp
compute_planning_depth_fixed_iters.cpp
compute_planning_depth_nonoise.cpp
```
**→ 이 바이너리들이 PV depth를 계산합니다!**

#### 모델 변형들
```
heuristic_drop.h/cpp       # Feature dropping variant
heuristic_fixed_branch.h/cpp
heuristic_fixed_iters.h/cpp
heuristic_nhp.h/cpp        # No heuristic pruning
```

#### 데이터 구조
```
data_struct.h / data_struct.cpp
└── 실험 데이터 로딩/저장

data_hvh.cpp              (314 KB)
└── Human vs Human 게임 데이터 (hardcoded)
```

#### 빌드 관련
```
fourinarow.cbp            # Code::Blocks 프로젝트
fourinarow.dll            (183 KB)
libfourinarow.a           (69 KB)
```

### Wrappers

#### MATLAB Wrapper
```
matlab wrapper/
├── bads/                 # Bayesian Adaptive Direct Search optimizer
├── fit_model.m           # 모델 피팅 메인
├── cross_val.m           # Cross-validation
├── estimate_loglik_mex.cpp
└── auto_fit.sh           # 자동 피팅 스크립트
```

#### Python Wrapper
```
Python wrapper/
├── dm_env/               # DeepMind Environment API
├── Fourinarow environment.ipynb
└── python_wrapper.cpp    # C++ ↔ Python bridge
```

#### JavaScript Wrapper
```
js wrapper/
└── 웹 실험용
```

---

## 📓 `xRL_pilot/Analysis notebooks/` - 분석 노트북

### 메인 노트북
```
learning.ipynb            (148 KB)
└── **Experience에 따른 변화 분석** ⭐
    ├── Elo rating calculation (block 20/40/60/80/100)
    ├── Principal Variation depth 분석
    └── Parameter trajectories
```

### `old/` 디렉토리 - 이전 분석들

**중요 파일**:
```
depth_by_session.txt      (3.7 KB) ⚠️
└── 30 participants × 5 sessions
    └── 현재 사용 중이지만 정체 불명!
    └── PV depth인지 다른 depth metric인지 확실하지 않음

Expertise.ipynb           (620 KB)
└── Expertise 분석

Learning.ipynb            (36 KB)
└── 학습 곡선 분석

fmri.ipynb                (942 KB)
fmri_4inarow_scripts.ipynb(3.2 MB)
└── fMRI 데이터 분석

Eye movements.ipynb       (185 KB)
└── Eye tracking 분석

Freechoice.ipynb          (1.1 MB)
Generalization.ipynb      (436 KB)
Opening moves.ipynb       (565 KB)
Param_corrs.ipynb         (99 KB)
```

### `new/` 디렉토리 - 새 분석들

```
Model comparison.ipynb           (1.5 MB)
Learning and time pressure analysis.ipynb (1.8 MB)
Create splits.ipynb              (1.5 MB)
└── Cross-validation splits 생성

Eye tracking preprocessing.ipynb (487 KB)
Eye tracking feature analysis.ipynb (525 KB)
Parameter tradeoffs and reliability.ipynb (289 KB)

params_peak_final.txt            (5.2 MB) ⭐
└── 모든 참가자의 최종 파라미터
```

**Bayesian Elo 계산**:
```
Run Bayeselo.ipynb
bayeselo.exe
```

---

## 🧪 `xRL_pilot/Experiment code/` - 실험 인터페이스

```
Parse fourinarow data.ipynb  (2.2 MB)
└── 실험 데이터 파싱

static/                      # 웹 리소스 (CSS, JS, images)
templates/                   # HTML 템플릿
```

---

## 📊 루트 디렉토리 - 분석 스크립트 (새로 생성)

### Python 분석 스크립트들

```
data_reanalysis.py           (11 KB)
└── opendata/ CSV 파일 재분석
    ├── 파라미터 분포
    ├── Expertise 분류 (복합 지표)
    └── 시각화

model_comparison_analysis.py (9.7 KB)
└── 모델 변형 비교
    ├── Log-likelihood 비교
    └── 통계적 유의성 검정

immediate_analysis.py        (18 KB)
└── 즉시 분석 (PV depth, Discrimination, RT)
    ├── PV depth vs expertise
    ├── Logistic regression
    └── RT-parameter correlations

verify_depth_variable.py     (6.6 KB)
└── depth_by_session.txt 검증
```

### 생성된 데이터 파일

```
analysis_participant_with_expertise.csv (7.5 KB)
└── 40명 참가자별:
    ├── 평균 파라미터
    ├── Expertise score & label
    └── Response time 통계

analysis_participant_summary.csv (6.1 KB)
analysis_model_comparison_by_participant.csv (2.1 KB)
analysis_summary.json (773 B)
immediate_discrimination_results.csv (1.4 KB)
```

### 시각화 결과 (PNG)

```
analysis_parameter_distributions.png
analysis_ll_by_participant.png
analysis_response_time.png
analysis_parameter_correlations.png
analysis_expertise_distribution.png
analysis_model_comparison.png
analysis_planning_models_parameters.png
analysis_participant_model_comparison.png

immediate_pv_depth_analysis.png
immediate_expertise_discrimination.png
immediate_rt_correlation.png
verify_depth_comparison.png
```

---

## ⚠️ 현재 불명확한 점

### 1. `depth_by_session.txt`의 정체

**위치**: `xRL_pilot/Analysis notebooks/old/depth_by_session.txt`

**문제**:
- 30명 × 5 sessions 데이터
- 값 범위: 2.74 ~ 10.37
- Learning notebook에서 PV depth는 `-2` 보정 사용
- 이 파일이 정확히 무엇인지 확실하지 않음

**가능성**:
1. PV depth + 2 (보정 전)
2. Mean depth (다른 metric)
3. 이미 보정된 PV depth

**검증 방법**:
```bash
cd "xRL_pilot/Model code"
./compute_planning_depth [data_file]
# → pv_depth_X.txt 파일 생성하여 비교
```

### 2. 원본 Peak 데이터 위치

**Learning notebook에서 참조**:
```python
path = '/Users/ionatankuperwajs/Desktop/4-in-a-row/Data/peak/splits/'
```

**현재 저장소**: 이 경로가 존재하지 않음

**필요**:
- `splits/` 디렉토리 구조 파악
- 각 참가자별 `pv_depth_X.txt` 파일 위치

### 3. 참가자 매칭

**opendata**: 40명 (participant 1-40)
**depth_by_session.txt**: 30명
**Learning notebook**: 150명 (Peak app users)

→ 어떤 참가자들이 서로 대응되는지 확인 필요

---

## 🎯 데이터 분석 워크플로우

### 현재까지 수행한 작업

```
1. opendata/ 로딩 및 기본 통계
   └── data_reanalysis.py

2. 모델 변형 비교
   └── model_comparison_analysis.py
       └── Main model vs Fixed-depth vs MCTS 등

3. Expertise 판별
   └── immediate_analysis.py
       ├── PV depth (depth_by_session.txt 사용)
       ├── Logistic regression (AUC=0.99)
       └── RT 상관관계

4. Depth 변수 검증
   └── verify_depth_variable.py
       └── Raw vs Corrected(-2) 비교
```

### 다음 단계

```
1. depth_by_session.txt 정체 확인
   ├── C++ 바이너리 실행하여 PV depth 재계산
   └── 또는 원본 Peak 데이터 찾기

2. 참가자 매칭 확인
   └── opendata 40명 ↔ depth 30명 ↔ Peak 150명

3. 고정 planning depth (h=1,2,3,4,5) 모델 구현
   └── C++ 코드 수정 + MATLAB 피팅
```

---

## 📚 주요 참고 자료

### 코드 이해를 위한 핵심 파일

1. **Planning 알고리즘**: `Model code/bfs.cpp` (line 60-70: `get_depth_of_pv()`)
2. **Heuristic 구조**: `Model code/heuristic.h` (line 30-50: parameters)
3. **모델 피팅**: `Model code/matlab wrapper/fit_model.m`
4. **PV depth 계산**: `Analysis notebooks/learning.ipynb` (cell 36)

### 데이터 형식

**Board state encoding**:
- 36-character binary string (6×6 board)
- `black_pieces`: "000000000001000000000000000000000000"
- `white_pieces`: "000000000000000000000000100000000000"

**Move encoding**:
- Integer 0-35 (36 positions)

---

## 🔄 업데이트 이력

- **2024-12-17**: 초기 생성
  - 전체 폴더 구조 스캔
  - 주요 파일 목록 및 용도 정리
  - 불명확한 점 명시 (depth_by_session.txt 등)
