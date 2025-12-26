# 스크립트 디렉토리

서버 배포 및 실험 자동화를 위한 스크립트 모음입니다.

## 📜 스크립트 목록

### 1. `deploy_to_server.sh` - 서버 배포
```bash
./scripts/deploy_to_server.sh server.university.edu your_username
```

**기능**:
- SSH 연결 테스트
- Git 변경사항 확인 및 push
- 서버에 Git clone/pull
- 데이터 전송 (선택)
- 환경 확인

**사용 시나리오**: 최초 서버 설정 또는 코드 업데이트

---

### 2. `train_all_h_server.sh` - 전체 실험 실행 (서버용)
```bash
# 서버에서 실행
./scripts/train_all_h_server.sh sequential  # 순차 (1시간)
./scripts/train_all_h_server.sh parallel    # 병렬 (30분)
```

**기능**:
- h=1,2,4,8 모두 학습
- 로그 자동 저장
- Sequential/Parallel 모드 선택
- Conda 환경 자동 활성화

**권장**: tmux에서 실행

---

### 3. `sync_from_server.sh` - 결과 동기화
```bash
# 로컬(MacBook)에서 실행
./scripts/sync_from_server.sh server.university.edu your_username
```

**기능**:
- 학습된 모델 다운로드
- 로그 파일 다운로드
- 그림 파일 다운로드
- TensorBoard 로그 다운로드

**사용 시나리오**: 실험 완료 후 결과 분석

---

## 🚀 전체 워크플로우

### Step 1: 초기 배포 (한 번만)
```bash
# 로컬(MacBook)
cd /Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot
./scripts/deploy_to_server.sh server.edu username

# 서버
ssh username@server.edu
cd ~/projects/xRL_pilot

# 환경 설정
conda create -n xrl_pilot python=3.9 -y
conda activate xrl_pilot
pip install gymnasium stable-baselines3 imitation torch numpy pandas matplotlib
```

### Step 2: 실험 실행
```bash
# 서버
tmux new -s xrl_training
cd ~/projects/xRL_pilot
./scripts/train_all_h_server.sh parallel
# Ctrl+B, D (detach)

# 진행 상황 확인
tmux attach -t xrl_training
tail -f logs/train_h4_*.log
```

### Step 3: 결과 동기화
```bash
# 로컬(MacBook)
./scripts/sync_from_server.sh server.edu username

# 분석
python3 compare_option_a_vs_b.py --h 4
```

---

## 💡 팁

### tmux 사용법
```bash
# 새 세션
tmux new -s session_name

# Detach (나가기, 백그라운드 실행)
Ctrl+B, D

# 재접속
tmux attach -t session_name

# 세션 목록
tmux ls

# 세션 종료
tmux kill-session -t session_name
```

### 로그 실시간 확인
```bash
# 최신 로그 파일
tail -f logs/train_h4_*.log

# 여러 로그 동시
tail -f logs/train_h*.log
```

### 프로세스 확인
```bash
# 실행 중인 Python 프로세스
ps aux | grep train_airl_pure_nn

# 메모리 사용량
top -o %MEM
```

---

## 🔧 커스터마이징

### timesteps 조정
```bash
# train_all_h_server.sh 수정
--total_timesteps 50000  # → 25000 (더 빠름)
```

### 특정 h만 실행
```bash
# 서버
python3 fourinarow_airl/train_airl_pure_nn.py \
    --h 4 \
    --total_timesteps 50000 \
    --output_dir models/airl_pure_nn_results
```

### 배치 크기 조정
```bash
--demo_batch_size 64  # → 32 (메모리 절약)
```

---

## 📊 예상 시간 및 리소스

| 모드 | 시간 | 메모리 | CPU |
|-----|------|--------|-----|
| Sequential | ~1시간 | ~1GB | 50-70% |
| Parallel (2개씩) | ~30분 | ~2GB | 70-80% |
| Parallel (4개) | ~20분 | ~4GB | 90-100% |

---

## ⚠️ 주의사항

1. **SSH 키 설정 필수**
   ```bash
   ssh-copy-id username@server.edu
   ```

2. **서버 디스크 공간 확인**
   ```bash
   df -h ~/projects
   # 최소 1GB 여유 필요
   ```

3. **tmux/screen 사용**
   - SSH 끊겨도 계속 실행
   - 장시간 실험 필수

4. **정기적 동기화**
   - 실험 중간에도 sync 추천
   - 백업 용도

---

## 📝 체크리스트

### 배포 전
- [ ] Git 변경사항 커밋
- [ ] SSH 접속 확인
- [ ] 서버 디스크 공간 확인

### 실행 전
- [ ] conda 환경 활성화
- [ ] 로그 디렉토리 생성
- [ ] tmux 세션 시작

### 실행 후
- [ ] 로그 파일 확인
- [ ] 결과 동기화
- [ ] 분석 실행

---

**문서**: `/docs/SERVER_DEPLOYMENT_GUIDE.md` 참고
