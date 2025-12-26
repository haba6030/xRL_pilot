# 연구실 서버 배포 가이드

## 목차
1. [서버 사용 이유 및 장점](#1-서버-사용-이유-및-장점)
2. [배포 방법 비교](#2-배포-방법-비교)
3. [방법 1: Git을 통한 배포 (권장)](#3-방법-1-git을-통한-배포-권장)
4. [방법 2: 직접 전송 (rsync/scp)](#4-방법-2-직접-전송-rsyncscp)
5. [서버 환경 설정](#5-서버-환경-설정)
6. [실행 방법 (서버)](#6-실행-방법-서버)
7. [결과 동기화](#7-결과-동기화)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. 서버 사용 이유 및 장점

### 로컬 (MacBook M4 Max) vs 서버 비교

| 항목 | 로컬 (MacBook) | 서버 |
|-----|---------------|------|
| **실행 시간** | 30분-1시간 | 20-40분 (GPU 시) |
| **컴퓨터 점유** | 점유됨 (작업 불가) | 백그라운드 실행 (작업 가능) |
| **전력 소모** | 배터리 소모 | 서버 전력 |
| **안정성** | 노트북 끄면 중단 | 계속 실행 |
| **장기 실험** | 부적합 | 적합 (tmux/screen) |
| **GPU** | M4 Max (MPS) | CUDA GPU (더 빠를 수 있음) |

### 🎯 서버 사용이 유리한 경우

✅ **장시간 실험** (여러 h를 순차 실행, 1-2시간 이상)
✅ **여러 실험 병렬** (다양한 하이퍼파라미터)
✅ **맥북을 다른 작업에 사용**
✅ **안정적인 실행 환경** (중간에 끄지 않음)
✅ **결과 백업** (서버에 자동 저장)

### ⚠️ 로컬 실행이 나은 경우

- 빠른 테스트 (10분 이내)
- 코드 디버깅
- 실시간 모니터링 필요
- 서버 접근 불가

---

## 2. 배포 방법 비교

### 방법 1: Git을 통한 배포 (권장 ⭐)

**장점**:
- ✅ 버전 관리
- ✅ 깔끔한 동기화
- ✅ 변경사항만 전송
- ✅ 협업 용이

**단점**:
- ⚠️ 대용량 데이터 파일 부적합 (opendata, models)
- ⚠️ Git LFS 필요할 수 있음

**적합한 경우**:
- 코드 위주 프로젝트
- 데이터는 서버에서 별도 생성
- 팀 협업

---

### 방법 2: rsync/scp를 통한 직접 전송

**장점**:
- ✅ 모든 파일 전송 가능
- ✅ 빠른 초기 설정
- ✅ Git 없이 사용 가능

**단점**:
- ⚠️ 버전 관리 없음
- ⚠️ 매번 수동 동기화
- ⚠️ 전체 파일 전송 (느림)

**적합한 경우**:
- 일회성 실험
- 데이터 포함 전체 전송
- Git 사용 불가

---

### 방법 3: 하이브리드 (권장 for 이 프로젝트)

**전략**:
1. **코드**: Git으로 관리
2. **데이터**: rsync 또는 서버에서 직접 생성
3. **결과**: rsync로 동기화

---

## 3. 방법 1: Git을 통한 배포 (권장)

### 현재 상태 확인

```bash
# 로컬 (MacBook)
cd /Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot

# Git 상태 확인
git status
git remote -v
# origin: https://github.com/haba6030/xRL_pilot
```

### 3.1 로컬에서 변경사항 커밋 & Push

```bash
# 현재 작업 커밋
git add docs/
git add fourinarow_airl/
git add *.py
git add CLAUDE.md README.md

# 커밋
git commit -m "Add Option A implementation and documentation

- Add pure NN AIRL training (Option A)
- Add resource estimation guide
- Add depth h research purpose doc
- Update documentation"

# GitHub에 push
git push origin main
```

### 3.2 서버에서 Clone

```bash
# 서버 SSH 접속
ssh your_username@server_address

# 작업 디렉토리 이동
cd ~/projects  # 또는 원하는 위치

# Git clone
git clone https://github.com/haba6030/xRL_pilot.git
cd xRL_pilot

# 확인
ls -la
```

### 3.3 업데이트 (나중에 변경사항 반영)

```bash
# 로컬에서 변경 후 push
git add .
git commit -m "Update code"
git push origin main

# 서버에서 pull
cd ~/projects/xRL_pilot
git pull origin main
```

---

## 4. 방법 2: 직접 전송 (rsync/scp)

### 4.1 전체 프로젝트 전송

```bash
# 로컬 (MacBook)에서 실행
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'models/' \
    --exclude '.DS_Store' \
    /Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/ \
    your_username@server_address:~/projects/xRL_pilot/

# 예상 시간: 5-10분 (네트워크 속도에 따라)
# 전송 크기: ~166MB (전체) 또는 ~20MB (코드만)
```

### 4.2 특정 폴더만 전송

```bash
# 코드만 전송 (빠름)
rsync -avz --progress \
    fourinarow_airl/ \
    your_username@server_address:~/projects/xRL_pilot/fourinarow_airl/

# 문서
rsync -avz --progress \
    docs/ \
    your_username@server_address:~/projects/xRL_pilot/docs/

# 데이터 (opendata)
rsync -avz --progress \
    opendata/ \
    your_username@server_address:~/projects/xRL_pilot/opendata/
```

### 4.3 scp 사용 (간단한 파일 전송)

```bash
# 단일 파일
scp train_script.py your_username@server_address:~/projects/xRL_pilot/

# 폴더
scp -r fourinarow_airl/ your_username@server_address:~/projects/xRL_pilot/
```

---

## 5. 서버 환경 설정

### 5.1 Python 환경 설정

#### Option A: Conda 사용 (권장)

```bash
# 서버 SSH 접속 후
cd ~/projects/xRL_pilot

# Conda environment 생성
conda create -n xrl_pilot python=3.9 -y
conda activate xrl_pilot

# 필수 패키지 설치
pip install gymnasium
pip install stable-baselines3
pip install imitation
pip install torch torchvision
pip install numpy pandas matplotlib scipy

# 추가 패키지
pip install tensorboard

# 설치 확인
python -c "import gymnasium; import stable_baselines3; import imitation; print('All packages installed!')"
```

#### Option B: venv 사용

```bash
# Python venv 생성
python3.9 -m venv venv
source venv/bin/activate

# 패키지 설치 (위와 동일)
pip install gymnasium stable-baselines3 imitation torch numpy pandas matplotlib scipy tensorboard
```

#### Option C: requirements.txt 사용

```bash
# 로컬에서 requirements.txt 생성
pip freeze > requirements.txt

# 서버에 전송
scp requirements.txt your_username@server_address:~/projects/xRL_pilot/

# 서버에서 설치
cd ~/projects/xRL_pilot
conda activate xrl_pilot
pip install -r requirements.txt
```

---

### 5.2 C++ BFS Wrapper 컴파일 (필요 시)

**중요**: Python wrapper는 이미 작동하지만, 성능 향상을 위해 C++ 재컴파일 가능

```bash
cd ~/projects/xRL_pilot/xRL_pilot/Model\ code

# C++ 컴파일러 확인
g++ --version

# 컴파일 (예시)
g++ -O3 -shared -fPIC \
    -o libfourinarow.so \
    heuristic.cpp bfs.cpp board.cpp \
    -std=c++11

# Python wrapper 테스트
cd ~/projects/xRL_pilot
python -c "from fourinarow_airl.bfs_wrapper import BFSPolicy; print('BFS wrapper OK!')"
```

**참고**: 이미 Python 구현이 있으므로 C++ 컴파일 실패해도 실행 가능

---

### 5.3 데이터 준비

#### Option 1: 서버에서 직접 생성 (권장)

```bash
# BFS expert data 생성
cd ~/projects/xRL_pilot

conda activate xrl_pilot

python3 fourinarow_airl/generate_training_data.py \
    --h 4 \
    --num_episodes 100 \
    --output training_data/depth_h4.pkl

# 각 h에 대해 반복
for h in 1 2 4 8; do
    python3 fourinarow_airl/generate_training_data.py \
        --h $h \
        --num_episodes 100
done
```

#### Option 2: 로컬에서 전송

```bash
# 로컬 (MacBook)
rsync -avz --progress \
    opendata/ \
    your_username@server_address:~/projects/xRL_pilot/opendata/

rsync -avz --progress \
    training_data/ \
    your_username@server_address:~/projects/xRL_pilot/training_data/
```

---

## 6. 실행 방법 (서버)

### 6.1 즉시 실행 (간단한 테스트)

```bash
# SSH 접속
ssh your_username@server_address

# 환경 활성화
cd ~/projects/xRL_pilot
conda activate xrl_pilot

# 테스트 실행
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 4 \
    --total_timesteps 10000 \
    --output_dir models/airl_pure_nn_results

# 주의: SSH 연결 끊기면 종료됨!
```

---

### 6.2 백그라운드 실행 (tmux 사용, 권장 ⭐)

**장점**: SSH 끊겨도 계속 실행, 재접속 가능

```bash
# tmux 세션 시작
tmux new -s xrl_training

# 환경 활성화
cd ~/projects/xRL_pilot
conda activate xrl_pilot

# 실행
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 4 \
    --total_timesteps 50000 \
    --output_dir models/airl_pure_nn_results

# tmux에서 나가기 (실행은 계속됨)
# Ctrl+B, D (detach)

# 나중에 다시 접속
tmux attach -t xrl_training

# 세션 목록 보기
tmux ls

# 세션 종료
tmux kill-session -t xrl_training
```

---

### 6.3 모든 h 실험 자동화 (스크립트)

#### 순차 실행

```bash
# train_all_h.sh 생성
cat > train_all_h.sh << 'EOF'
#!/bin/bash

# Conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh  # 경로 확인 필요
conda activate xrl_pilot

cd ~/projects/xRL_pilot

for h in 1 2 4 8; do
    echo "========================================="
    echo "Training h=$h"
    echo "========================================="

    python3 fourinarow_airl/train_airl_pure_nn.py \
        --h $h \
        --total_timesteps 50000 \
        --output_dir models/airl_pure_nn_results \
        2>&1 | tee logs/train_h${h}.log

    echo "h=$h completed at $(date)"
done

echo "All training complete!"
EOF

# 실행 권한
chmod +x train_all_h.sh

# tmux에서 실행
tmux new -s xrl_training
./train_all_h.sh
# Ctrl+B, D (detach)
```

#### 병렬 실행

```bash
# train_all_h_parallel.sh
cat > train_all_h_parallel.sh << 'EOF'
#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xrl_pilot

cd ~/projects/xRL_pilot

# h=1,2 병렬
python3 fourinarow_airl/train_airl_pure_nn.py --h 1 --total_timesteps 50000 &
python3 fourinarow_airl/train_airl_pure_nn.py --h 2 --total_timesteps 50000 &
wait

# h=4,8 병렬
python3 fourinarow_airl/train_airl_pure_nn.py --h 4 --total_timesteps 50000 &
python3 fourinarow_airl/train_airl_pure_nn.py --h 8 --total_timesteps 50000 &
wait

echo "All training complete!"
EOF

chmod +x train_all_h_parallel.sh

# 실행
tmux new -s xrl_training
./train_all_h_parallel.sh
```

---

### 6.4 nohup 사용 (tmux 없을 때)

```bash
# 백그라운드 실행
nohup python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 4 \
    --total_timesteps 50000 \
    > logs/train_h4.log 2>&1 &

# 프로세스 확인
ps aux | grep train_airl_pure_nn

# 로그 확인
tail -f logs/train_h4.log

# 종료
kill <PID>
```

---

## 7. 결과 동기화

### 7.1 서버 → 로컬 (결과 다운로드)

```bash
# 로컬 (MacBook)에서 실행

# 학습된 모델 다운로드
rsync -avz --progress \
    your_username@server_address:~/projects/xRL_pilot/models/airl_pure_nn_results/ \
    ./models/airl_pure_nn_results/

# 로그 다운로드
rsync -avz --progress \
    your_username@server_address:~/projects/xRL_pilot/logs/ \
    ./logs/

# 그림 다운로드
rsync -avz --progress \
    your_username@server_address:~/projects/xRL_pilot/figures/ \
    ./figures/
```

### 7.2 자동 동기화 (스크립트)

```bash
# sync_from_server.sh (로컬)
cat > sync_from_server.sh << 'EOF'
#!/bin/bash

SERVER="your_username@server_address"
REMOTE_DIR="~/projects/xRL_pilot"
LOCAL_DIR="/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot"

echo "Syncing models..."
rsync -avz --progress $SERVER:$REMOTE_DIR/models/ $LOCAL_DIR/models/

echo "Syncing logs..."
rsync -avz --progress $SERVER:$REMOTE_DIR/logs/ $LOCAL_DIR/logs/

echo "Syncing figures..."
rsync -avz --progress $SERVER:$REMOTE_DIR/figures/ $LOCAL_DIR/figures/

echo "Sync complete!"
EOF

chmod +x sync_from_server.sh

# 사용
./sync_from_server.sh
```

---

## 8. 트러블슈팅

### 문제 1: SSH 접속 안 됨

```bash
# SSH 키 확인
ls -la ~/.ssh/

# SSH 키 생성 (없으면)
ssh-keygen -t rsa -b 4096

# 공개키를 서버에 복사
ssh-copy-id your_username@server_address

# 또는 수동으로
cat ~/.ssh/id_rsa.pub
# → 서버의 ~/.ssh/authorized_keys에 추가
```

---

### 문제 2: Conda 환경 활성화 안 됨

```bash
# conda init 실행
conda init bash  # 또는 zsh

# .bashrc 또는 .zshrc 재로드
source ~/.bashrc

# 수동 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xrl_pilot
```

---

### 문제 3: 패키지 import 에러

```bash
# Python 경로 확인
which python
python --version

# 패키지 설치 확인
pip list | grep -E "gymnasium|imitation|torch"

# 재설치
pip install --upgrade --force-reinstall imitation stable-baselines3
```

---

### 문제 4: 메모리 부족

```bash
# 메모리 확인
free -h

# 프로세스별 메모리 사용
top -o %MEM

# 해결책: 병렬 실행 줄이기
# 4개 → 2개 또는 순차 실행
```

---

### 문제 5: GPU 사용 안 됨

```bash
# CUDA 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"

# CPU로 강제 실행
export CUDA_VISIBLE_DEVICES=""
python3 train_airl_pure_nn.py ...
```

---

## 9. 권장 워크플로우

### 초기 설정 (한 번만)

```bash
# 1. 로컬에서 Git push
git add .
git commit -m "Initial commit"
git push origin main

# 2. 서버에서 clone
ssh server
git clone https://github.com/haba6030/xRL_pilot.git
cd xRL_pilot

# 3. 환경 설정
conda create -n xrl_pilot python=3.9
conda activate xrl_pilot
pip install gymnasium stable-baselines3 imitation torch numpy pandas matplotlib

# 4. 데이터 생성
for h in 1 2 4 8; do
    python3 fourinarow_airl/generate_training_data.py --h $h --num_episodes 100
done
```

---

### 일반적인 작업 흐름

```bash
# 1. 로컬에서 코드 수정
# (MacBook에서 개발)

# 2. Git push
git add .
git commit -m "Update code"
git push origin main

# 3. 서버에서 pull
ssh server
cd ~/projects/xRL_pilot
git pull origin main

# 4. tmux에서 실행
tmux new -s experiment
conda activate xrl_pilot
./train_all_h.sh
# Ctrl+B, D

# 5. 로컬로 결과 동기화
# (로컬 MacBook)
./sync_from_server.sh

# 6. 로컬에서 분석
python3 compare_option_a_vs_b.py
```

---

## 10. 체크리스트

### 서버 배포 전 확인

- [ ] Git repository 최신 상태 (git push)
- [ ] 필요한 파일 모두 포함 (코드, 문서, 스크립트)
- [ ] .gitignore 설정 (모델, 로그 제외)
- [ ] 서버 SSH 접속 확인
- [ ] 서버 디스크 공간 확인 (최소 1GB 여유)

### 서버 설정 확인

- [ ] Python 3.9 설치
- [ ] Conda 또는 venv 환경 생성
- [ ] 필수 패키지 설치 (imitation, stable-baselines3, torch)
- [ ] tmux 또는 screen 설치
- [ ] 프로젝트 디렉토리 생성

### 실행 전 확인

- [ ] 환경 활성화 (conda activate xrl_pilot)
- [ ] 테스트 실행 성공 (단일 h, 10K timesteps)
- [ ] 로그 디렉토리 생성 (mkdir -p logs)
- [ ] 출력 디렉토리 생성 (mkdir -p models/airl_pure_nn_results)

---

## 요약

### 🎯 권장 방법 (하이브리드)

1. **코드**: Git으로 관리 (push/pull)
2. **데이터**: 서버에서 직접 생성 (BFS trajectories)
3. **실행**: tmux에서 백그라운드 실행
4. **결과**: rsync로 로컬에 동기화

### ⏱️ 예상 시간

| 작업 | 시간 |
|-----|------|
| 초기 설정 (환경, 데이터) | 20-30분 |
| 코드 push/pull | < 1분 |
| 실행 (4 depths) | 30분-1시간 |
| 결과 동기화 | 1-2분 |

### 💡 팁

- **tmux 사용 필수** (SSH 끊겨도 계속 실행)
- **로그 파일 확인** (`tail -f logs/train_h4.log`)
- **정기적으로 결과 동기화** (실험 중간에도)
- **Git으로 코드만 관리** (모델/데이터는 .gitignore)

---

**준비되셨으면 바로 시작하세요!** 🚀
