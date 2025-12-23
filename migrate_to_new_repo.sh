#!/bin/bash
# 새 독립 저장소로 마이그레이션 스크립트

echo "=========================================="
echo "Fork 관계 끊고 새 저장소로 마이그레이션"
echo "=========================================="
echo ""

# 1. 먼저 GitHub에서 새 저장소 생성하세요
echo "Step 1: GitHub에서 새 저장소 생성"
echo "  → https://github.com/new"
echo "  → Repository name: xRL_pilot_research"
echo "  → Public"
echo "  → README, .gitignore, license 모두 체크 안 함"
echo ""
read -p "새 저장소를 생성했나요? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "먼저 GitHub에서 새 저장소를 생성해주세요."
    exit 1
fi

# 2. 새 저장소 URL 입력
echo ""
echo "Step 2: 새 저장소 URL 입력"
echo "  예시: https://github.com/haba6030/xRL_pilot_research.git"
read -p "새 저장소 URL: " NEW_REPO_URL

# 3. Remote 변경
echo ""
echo "Step 3: Remote 변경 중..."
git remote remove origin
git remote add origin "$NEW_REPO_URL"

# 4. 확인
echo ""
echo "Step 4: Remote 확인"
git remote -v

# 5. Push
echo ""
read -p "새 저장소에 push하시겠습니까? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push -u origin main
    echo ""
    echo "✅ 완료!"
    echo "새 저장소: $NEW_REPO_URL"
    echo ""
    echo "기존 fork 저장소(fourinarow)는 GitHub에서 삭제해도 됩니다:"
    echo "  → https://github.com/haba6030/fourinarow"
    echo "  → Settings → Danger Zone → Delete this repository"
fi

echo ""
echo "=========================================="
echo "마이그레이션 완료"
echo "=========================================="
