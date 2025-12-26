#!/bin/bash
# 서버에서 결과 동기화
# 사용법: ./scripts/sync_from_server.sh [server_address] [username]

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}서버 결과 동기화${NC}"
echo -e "${GREEN}========================================${NC}"

# 파라미터
if [ $# -lt 2 ]; then
    echo "사용법: $0 <server_address> <username>"
    echo "예시: $0 server.university.edu your_username"
    exit 1
fi

SERVER_ADDRESS=$1
USERNAME=$2
SERVER="$USERNAME@$SERVER_ADDRESS"
REMOTE_DIR="~/projects/xRL_pilot"
LOCAL_DIR="$(pwd)"

echo -e "${YELLOW}서버: $SERVER${NC}"
echo -e "${YELLOW}로컬 디렉토리: $LOCAL_DIR${NC}"

# 1. 모델 동기화
echo -e "\n${GREEN}[1/4] 모델 동기화...${NC}"
rsync -avz --progress \
    --exclude '*.tmp' \
    $SERVER:$REMOTE_DIR/models/airl_pure_nn_results/ \
    $LOCAL_DIR/models/airl_pure_nn_results/

# 2. 로그 동기화
echo -e "\n${GREEN}[2/4] 로그 동기화...${NC}"
rsync -avz --progress \
    $SERVER:$REMOTE_DIR/logs/ \
    $LOCAL_DIR/logs/

# 3. 그림 동기화
echo -e "\n${GREEN}[3/4] 그림 동기화...${NC}"
if ssh $SERVER "[ -d $REMOTE_DIR/figures ]"; then
    rsync -avz --progress \
        $SERVER:$REMOTE_DIR/figures/ \
        $LOCAL_DIR/figures/
fi

# 4. TensorBoard 로그
echo -e "\n${GREEN}[4/4] TensorBoard 로그 동기화...${NC}"
if ssh $SERVER "[ -d $REMOTE_DIR/tensorboard_logs ]"; then
    rsync -avz --progress \
        $SERVER:$REMOTE_DIR/tensorboard_logs/ \
        $LOCAL_DIR/tensorboard_logs/
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}동기화 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "동기화된 파일:"
echo "- 모델: models/airl_pure_nn_results/"
echo "- 로그: logs/"
echo "- 그림: figures/"
echo "- TensorBoard: tensorboard_logs/"
echo ""
echo "다음 단계: 분석 실행"
echo "  python3 compare_option_a_vs_b.py --h 4"
