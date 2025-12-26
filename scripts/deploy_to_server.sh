#!/bin/bash
# 서버 배포 스크립트
# 사용법: ./scripts/deploy_to_server.sh [server_address] [username]

set -e  # 에러 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}서버 배포 스크립트${NC}"
echo -e "${GREEN}========================================${NC}"

# 파라미터 확인
if [ $# -lt 2 ]; then
    echo -e "${RED}사용법: $0 <server_address> <username>${NC}"
    echo "예시: $0 server.university.edu your_username"
    exit 1
fi

SERVER_ADDRESS=$1
USERNAME=$2
SERVER="$USERNAME@$SERVER_ADDRESS"
REMOTE_DIR="~/projects/xRL_pilot"

echo -e "${YELLOW}서버: $SERVER${NC}"
echo -e "${YELLOW}원격 디렉토리: $REMOTE_DIR${NC}"

# 1. SSH 연결 테스트
echo -e "\n${GREEN}[1/5] SSH 연결 테스트...${NC}"
if ssh -o BatchMode=yes -o ConnectTimeout=5 $SERVER "echo '연결 성공'" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH 연결 성공${NC}"
else
    echo -e "${RED}✗ SSH 연결 실패${NC}"
    echo "SSH 키 설정이 필요할 수 있습니다:"
    echo "  ssh-copy-id $SERVER"
    exit 1
fi

# 2. 로컬 변경사항 확인
echo -e "\n${GREEN}[2/5] Git 상태 확인...${NC}"
if git status --porcelain | grep -q .; then
    echo -e "${YELLOW}⚠ 커밋되지 않은 변경사항이 있습니다:${NC}"
    git status --short

    read -p "변경사항을 커밋하고 push하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add docs/ fourinarow_airl/ scripts/ *.py CLAUDE.md README.md
        git commit -m "Update before server deployment $(date +%Y-%m-%d)"
        git push origin main
        echo -e "${GREEN}✓ Git push 완료${NC}"
    else
        echo -e "${YELLOW}⚠ 변경사항을 push하지 않았습니다${NC}"
    fi
else
    echo -e "${GREEN}✓ 모든 변경사항이 커밋됨${NC}"
fi

# 3. 서버에 디렉토리 확인/생성
echo -e "\n${GREEN}[3/5] 서버 디렉토리 확인...${NC}"
if ssh $SERVER "[ -d $REMOTE_DIR ]"; then
    echo -e "${YELLOW}⚠ 디렉토리가 이미 존재합니다${NC}"
    read -p "Git pull로 업데이트하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ssh $SERVER "cd $REMOTE_DIR && git pull origin main"
        echo -e "${GREEN}✓ Git pull 완료${NC}"
    fi
else
    echo "새로운 디렉토리를 생성합니다..."
    ssh $SERVER "mkdir -p ~/projects"
    ssh $SERVER "cd ~/projects && git clone https://github.com/haba6030/xRL_pilot.git"
    echo -e "${GREEN}✓ Git clone 완료${NC}"
fi

# 4. 데이터 전송 (선택)
echo -e "\n${GREEN}[4/5] 데이터 전송...${NC}"
read -p "opendata 폴더를 전송하시겠습니까? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "opendata 전송 중... (크기: 12MB)"
    rsync -avz --progress \
        --exclude '.DS_Store' \
        opendata/ \
        $SERVER:$REMOTE_DIR/opendata/
    echo -e "${GREEN}✓ opendata 전송 완료${NC}"
else
    echo -e "${YELLOW}⚠ opendata는 서버에서 직접 생성하세요${NC}"
fi

# 5. 환경 설정 확인
echo -e "\n${GREEN}[5/5] Python 환경 확인...${NC}"
if ssh $SERVER "conda env list | grep -q xrl_pilot"; then
    echo -e "${GREEN}✓ conda 환경 'xrl_pilot' 존재${NC}"
else
    echo -e "${YELLOW}⚠ conda 환경 'xrl_pilot'이 없습니다${NC}"
    echo "서버에서 다음 명령을 실행하세요:"
    echo "  conda create -n xrl_pilot python=3.9 -y"
    echo "  conda activate xrl_pilot"
    echo "  pip install gymnasium stable-baselines3 imitation torch numpy pandas matplotlib"
fi

# 완료
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}배포 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n다음 단계:"
echo "1. 서버 접속:"
echo -e "   ${YELLOW}ssh $SERVER${NC}"
echo "2. 환경 활성화:"
echo -e "   ${YELLOW}conda activate xrl_pilot${NC}"
echo "3. 실행:"
echo -e "   ${YELLOW}cd $REMOTE_DIR${NC}"
echo -e "   ${YELLOW}tmux new -s xrl_training${NC}"
echo -e "   ${YELLOW}./scripts/train_all_h_server.sh${NC}"
echo ""
