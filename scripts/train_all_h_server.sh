#!/bin/bash
# 서버에서 모든 h 실험 실행
# 사용법: ./scripts/train_all_h_server.sh [sequential|parallel]

set -e

# 색상
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODE=${1:-sequential}  # 기본값: sequential

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Option A 전체 실험 실행${NC}"
echo -e "${GREEN}Mode: $MODE${NC}"
echo -e "${GREEN}========================================${NC}"

# 로그 디렉토리 생성
mkdir -p logs
mkdir -p models/airl_pure_nn_results

# Conda 활성화 (경로는 서버에 맞게 수정 필요)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate xrl_pilot || {
    echo -e "${YELLOW}⚠ conda 환경 활성화 실패${NC}"
    echo "수동으로 활성화하세요: conda activate xrl_pilot"
    exit 1
}

echo -e "${GREEN}✓ 환경 활성화 완료${NC}"
echo "Python: $(which python)"
echo "시작 시간: $(date)"

if [ "$MODE" = "sequential" ]; then
    # 순차 실행
    echo -e "\n${GREEN}순차 실행 모드 (약 1시간 예상)${NC}"

    for h in 1 2 4 8; do
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}Training h=$h${NC}"
        echo -e "${GREEN}========================================${NC}"

        python3 fourinarow_airl/train_airl_pure_nn.py \
            --h $h \
            --total_timesteps 50000 \
            --demo_batch_size 64 \
            --output_dir models/airl_pure_nn_results \
            2>&1 | tee logs/train_h${h}_$(date +%Y%m%d_%H%M%S).log

        echo -e "${GREEN}✓ h=$h 완료: $(date)${NC}"
    done

elif [ "$MODE" = "parallel" ]; then
    # 병렬 실행 (2개씩)
    echo -e "\n${GREEN}병렬 실행 모드 (약 30분 예상)${NC}"

    echo -e "\n${YELLOW}h=1,2 병렬 실행...${NC}"
    python3 fourinarow_airl/train_airl_pure_nn.py \
        --h 1 --total_timesteps 50000 --output_dir models/airl_pure_nn_results \
        > logs/train_h1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    python3 fourinarow_airl/train_airl_pure_nn.py \
        --h 2 --total_timesteps 50000 --output_dir models/airl_pure_nn_results \
        > logs/train_h2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    wait
    echo -e "${GREEN}✓ h=1,2 완료${NC}"

    echo -e "\n${YELLOW}h=4,8 병렬 실행...${NC}"
    python3 fourinarow_airl/train_airl_pure_nn.py \
        --h 4 --total_timesteps 50000 --output_dir models/airl_pure_nn_results \
        > logs/train_h4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    python3 fourinarow_airl/train_airl_pure_nn.py \
        --h 8 --total_timesteps 50000 --output_dir models/airl_pure_nn_results \
        > logs/train_h8_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    wait
    echo -e "${GREEN}✓ h=4,8 완료${NC}"

else
    echo "잘못된 모드: $MODE"
    echo "사용법: $0 [sequential|parallel]"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}모든 실험 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "종료 시간: $(date)"
echo ""
echo "결과 파일:"
ls -lh models/airl_pure_nn_results/
echo ""
echo "로그 파일:"
ls -lh logs/
