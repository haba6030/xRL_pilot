# van Opheusden (2023)과 우리 연구의 비교

**Planning Depth 개념의 차이와 Expertise의 본질**

**날짜**: 2025-12-29

이 문서는 van Opheusden et al. (2023)의 발견과 우리 연구 결과를 비교하고, 표면적 차이를 어떻게 조화시킬 수 있는지 설명합니다.

## 핵심 질문

van Opheusden (2023)과 우리 연구 모두 planning과 expertise를 다루는데, 왜 결과가 다른가?

답: 서로 다른 planning depth 개념을 측정했기 때문입니다.
- van Opheusden: PV (Principal Variation) depth = 탐색 트리 깊이
- 우리: Behavioral h = 결정 관련 lookahead horizon

두 발견 모두 정확하며, 함께 expertise의 본질을 밝힙니다.

## 결과 직접 비교

van Opheusden (2023) 결과:

| 메트릭 | Expert | Novice | 차이 | 통계 | 해석 |
|--------|--------|--------|------|------|------|
| PV depth | 6.23 ± 1.30 | 7.29 ± 0.55 | -1.06 | r=-0.50, p<0.01 | Expert가 낮음 |
| Pruning threshold | 높음 | 낮음 | + | 유의 | 효율적 가지치기 |
| Search iterations | 적음 | 많음 | - | 유의 | 효율적 탐색 |

해석: "전문가는 더 효율적으로 계획한다 (깊이 탐색하지 않음)"

우리 연구 결과:

| 메트릭 | Expert | Novice | 차이 | 통계 | 해석 |
|--------|--------|--------|------|------|------|
| Behavioral h | 1.77 ± 0.12 | 1.77 ± 0.13 | 0.00 | r=-0.01, p=0.94 | 차이 없음 |
| van Opheusden features | 높음 | 낮음 | + | AUC=0.84 | 품질 차이 |
| h 분포 | P(h=1)=47% | P(h=1)=47% | 0% | p>0.8 | 동일한 분포 |

해석: "전문가와 초보자는 같은 behavioral horizon을 사용한다"

## 차이점 원인: 다른 개념 측정

PV Depth (van Opheusden): Best-First Search에서 탐색한 최장 경로. 탐색 트리의 구조적 특성이며 계산 노력의 지표입니다. Pruning과 stopping에 의해 결정되며, 실제 선택한 action과 무관할 수 있습니다.

예시: PV depth = 7은 깊이 7까지 탐색했다는 의미이지만, 실제로는 1-2 스텝만 결정적이었을 수 있습니다. 나머지는 검증을 위한 탐색입니다.

Behavioral h (우리 연구): 행동을 설명하는 미래 horizon. 실제 관찰된 action이 어느 시간 horizon의 state information으로 가장 잘 예측되는지를 측정합니다.

측정 방법:
```python
# 각 h에 대해 likelihood 계산
for h in [1, 2, 3, 4]:
    state_future = game_record[t+h]  # 실제 인간의 미래
    likelihood[h] = model_h.predict_proba(state_t, state_future)[action_t]

# 가장 높은 likelihood를 주는 h가 behavioral h
estimated_h = argmax(likelihood)
```

특징: 행동 데이터에서 직접 추론되며, 실제 선택에 영향을 준 temporal horizon을 반영합니다.

## 두 개념의 관계

PV depth와 behavioral h는 다른 것을 측정합니다:

PV depth = behavioral h + verification depth

분해 예시:
- PV depth = 7 (트리에서 7 스텝 탐색)
- Behavioral h = 2 (실제로 처음 2 스텝이 선택을 결정)
- Verification depth = 5 (나머지 5 스텝은 검증용)

Expert vs Novice:

Novice:
- 낮은 heuristics → 많은 검증 필요 → 높은 PV depth (7.29)
- 과제 요구사항 → behavioral h = 1.77

Expert:
- 높은 heuristics → 적은 검증 필요 → 낮은 PV depth (6.23)
- 과제 요구사항 → behavioral h = 1.77

PV depth는 expertise에 따라 변하지만 (더 나은 heuristics → 효율적 pruning), behavioral h는 변하지 않습니다 (과제 요구사항은 동일).

## 통합된 Expertise 모델

두 연구를 종합하면 expertise의 메커니즘이 명확해집니다:

Novice 행동:
1. 낮은 품질의 heuristics (van Opheusden features 낮음)
2. 불확실성 보상을 위한 깊은 탐색 (PV depth = 7.29)
3. 과제 요구사항에 따른 decision horizon (behavioral h = 1.77)
4. 결과: 비효율적이지만 올바른 depth

Expert 행동:
1. 높은 품질의 heuristics (van Opheusden features 높음)
2. 신뢰할 수 있는 heuristics로 얕은 탐색 (PV depth = 6.23)
3. 과제 요구사항에 따른 decision horizon (behavioral h = 1.77)
4. 결과: 효율적이며 올바른 depth

핵심 통찰: Expertise는 heuristic 품질에서 나오며, planning depth (behavioral h)가 아닙니다. 더 나은 heuristics는 효율적 탐색을 가능케 하지만 (낮은 PV depth), decision-relevant horizon은 변하지 않습니다 (동일한 behavioral h).

## 임상 및 IRL 함의

IRL의 경우:
- Planning depth h를 latent confounder로 모델링하되 expertise marker로 사용하지 마세요
- Heuristic 품질 (van Opheusden features)이 expertise를 예측합니다
- PV depth와 behavioral h를 혼동하지 마세요

인지과학의 경우:
- PV depth: 계산 효율성 지표 (expertise와 음의 상관)
- Behavioral h: 결정 horizon (expertise와 무관)
- Heuristic 품질: 실제 expertise의 원천

임상 응용의 경우:
- 평균 planning depth를 증가시키는 것이 아니라
- Heuristic 품질을 향상시키는 것에 집중하세요
- Context-적응적 planning (언제 깊게 vs 얕게)이 중요할 수 있습니다

## 요약

van Opheusden의 발견 (Expert PV depth < Novice PV depth)과 우리 발견 (Expert h = Novice h)은 모순이 아니라 상보적입니다.

PV depth는 tree exploration breadth를 측정하며 expertise와 음의 상관을 보입니다 (더 나은 heuristics → 효율적 pruning). Behavioral h는 decision horizon을 측정하며 expertise와 무관합니다 (과제 요구사항은 모두에게 동일).

둘 다 같은 결론을 지지합니다: Expertise는 더 나은 heuristics (무엇을 평가하는가)로부터 나오며, 더 깊은 planning (얼마나 앞을 보는가)이 아닙니다.

**마지막 업데이트**: 2025-12-31
