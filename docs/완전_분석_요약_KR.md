# 통합 분석: Planning Depth와 Expertise

**날짜**: 2025-12-29

이 문서는 세 가지 h 추정 방법(random rollout, rollout-free, opponent model)의 결과를 통합하고, planning depth와 van Opheusden의 heuristic features를 전문성 예측 측면에서 비교합니다.

## 핵심 질문

4-in-a-row 게임에서 planning depth h가 전문성을 식별할 수 있는가?

답변: 아니오. Planning depth는 행동으로부터 식별 가능하지만 전문성과는 무관합니다. Van Opheusden의 heuristic features (17차원 보드 평가 지표)는 84% AUC로 전문성을 예측하는 반면, planning depth는 chance level (53% AUC)입니다.

## 완전 결과

h 추정 방법 비교:

| 방법 | 평균 E[h] | Expert h | Novice h | Elo 상관 | 승률 상관 | Expertise AUC |
|------|-----------|----------|----------|----------|-----------|---------------|
| Random Rollout | 2.87 ± 0.08 | 2.80 | 2.84 | -0.12 (ns) | -0.43** | ~0.53 |
| Rollout-Free | 1.78 ± 0.12 | 1.77 | 1.77 | -0.01 (ns) | +0.08 (ns) | 0.53 |
| Opponent Model | 2.62 ± 0.10 | 2.61 | 2.63 | -0.03 (ns) | +0.06 (ns) | ~0.53 |

세 방법 모두 h가 전문성을 예측하지 못함을 보입니다.

Feature-based 대안:

| 방법 | 차원 | 개별 r | 결합 AUC | 해석 |
|------|------|--------|----------|------|
| van Opheusden Features | 17-dim | 0.035 (평균) | 0.84 | 강력한 예측력 |
| Planning Depth h | 1-dim | 0.012 | 0.53 | Chance level |

Features는 다변량 패턴을 통해 전문성을 예측합니다. 개별 features는 약하지만 (평균 |r| = 0.035, 0/17 유의함), 이들의 조합은 전문가와 비전문가를 강하게 구분합니다.

## 발견 1: Rollout 방법이 중요함

Random rollout은 h를 +1.09 스텝 (38% bias) 과대평가합니다. 이는 훈련이 실제 인간 게임 continuation(제약적, 전략적)을 사용하는 반면 추론은 random moves로 미래를 시뮬레이션(다양함, 탐색적)하기 때문입니다. 더 긴 horizon 모델이 다양한 미래로부터 불균형하게 이득을 보아 P(h=4)를 부풀립니다.

분포 변화:
- Random Rollout: P(h=1)=13%, P(h=2)=23%, P(h=3)=30%, P(h=4)=35%
- Rollout-Free: P(h=1)=47%, P(h=2)=24%, P(h=3)=19%, P(h=4)=10%

Rollout-free 방법은 훈련과 추론 모두에서 실제 게임 미래를 사용하여 이 artifact를 제거합니다. Opponent model rollout은 bias를 줄이지만 완전히 제거하지는 못합니다 (rollout-free 대비 +0.84 스텝).

권장사항: 실제 미래가 데이터에서 관찰 가능할 때는 rollout-free 방법을 사용하세요.

## 발견 2: 인간은 근시안적으로 계획함

Rollout-free 추정치: E[h] = 1.78 ± 0.12

분포:
- 47%의 수: h=1 (반응적, 즉각 대응)
- 24%의 수: h=2 (단기 계획)
- 19%의 수: h=3 (중기 계획)
- 10%의 수: h=4 (장기 계획)

이는 플레이어가 search tree에서 6-7 스텝을 탐색한다는 van Opheusden의 발견(PV depth)보다 훨씬 얕습니다. 이 차이는 tree exploration breadth(얼마나 넓게 탐색하는가)와 decision horizon(어디까지가 선택을 결정하는가) 간 구분을 반영합니다.

PV depth = behavioral h + verification depth

예시: 선택이 좋은지 검증하기 위해 6 스텝 탐색 (PV = 6), 하지만 어떤 수를 둘지 결정하는 것은 처음 2 스텝 (h = 2). 나머지 4 스텝은 pruning과 검증이지 의사결정이 아닙니다.

## 발견 3: Expertise Paradox는 강건함

Planning depth는 세 방법 모두에서 전문성과 관계가 없습니다:
- Random rollout: r = -0.12, p = 0.47, Expert h = Novice h
- Rollout-free: r = -0.01, p = 0.94, Expert h = Novice h
- Opponent model: r = -0.03, p = 0.86, Expert h = Novice h

이 null result는 rollout artifact 제거 후에도 지속됩니다. 이는 진짜 현상을 나타냅니다: h는 이 과제에서 전문성과 직교합니다.

## 발견 4: Features가 Expertise를 강하게 예측함

Van Opheusden의 17차원 features는 전문가 분류에서 AUC = 0.84를 달성하며, h의 AUC = 0.53 (chance level)과 비교됩니다. +0.31의 차이는 58.5% 향상을 나타냅니다.

개별 features는 약하지만 (Elo와 평균 |r| = 0.035, 0/17이 p < 0.05에서 유의함), 다변량 조합은 강합니다. 이는 전문성이 어떤 단일 측면에서의 우수함이 아니라 여러 heuristic 차원에 걸친 균형잡힌 프로필을 반영함을 나타냅니다.

상관 크기 상위 features:
1. 4-in-a-row horizontal: r = -0.244, p = 0.130 (유의하지 않음)
2. Connected 2-in-a-row diag1: r = +0.054, p = 0.742
3. 3-in-a-row diag1: r = +0.049, p = 0.765

가장 강한 개별 feature조차 유의성에 도달하지 못합니다. 전문성은 패턴에 관한 것이지 개별 구성요소가 아닙니다.

## van Opheusden et al. (2023)과의 조화

van Opheusden은 "Expertise increases planning depth in human gameplay"라고 보고했지만, 실제 발견은 expert PV depth (6.23 스텝)가 novice PV depth (7.29 스텝)보다 낮다는 것이며, r = -0.50, p < 0.01입니다. 그들은 전문가가 얕은 트리로 더 효율적으로 탐색한다고 결론지었습니다.

우리의 발견: Expert behavioral h = 1.769, Novice h = 1.768, r = -0.012, p = 0.94. Planning depth는 전문성과 관계가 없습니다.

두 발견 모두 호환 가능합니다. PV depth와 behavioral h가 다른 구성개념을 측정하기 때문입니다:

PV depth (van Opheusden):
- 계산 effort를 측정하는 search tree exploration metric
- 전문가에게 더 낮음 (더 나은 heuristics를 통한 효율적 pruning)

Behavioral h (우리 연구):
- 선택을 결정하는 거리를 측정하는 decision-relevant horizon
- 기술 수준에 걸쳐 동일함 (과제 요구사항이 유사함)

둘 다 같은 결론을 지지합니다: 전문성은 더 나은 heuristics(무엇을 평가하는가)로부터 나오며, 더 깊은 planning(얼마나 앞을 보는가)이 아닙니다. 전문가는 더 나은 heuristics가 공격적 pruning을 가능케 하기 때문에 더 효율적으로 탐색하지만, 모두가 능숙하게 플레이하려면 대략 2 스텝 앞을 봐야 하기 때문에 decision-relevant horizon은 유사합니다.

## Expertise 메커니즘 모델

Novice 행동:
- 여러 차원에 걸친 낮은 heuristics
- 보상을 위한 깊은 탐색 (brute-force, 높은 PV depth)
- 낮은 성능
- h ≈ 1.8 (expert와 동일)

Expert 행동:
- 여러 차원에 걸친 고품질 heuristics
- 얕은 탐색으로 충분 (효율적 pruning, 낮은 PV depth)
- 높은 성능
- h ≈ 1.8 (novice와 동일)

핵심 통찰: h는 과제 요구사항에 의해 제어되며 전문성이 아닙니다. 전문성은 heuristic 품질에서 나타나며 planning depth가 아닙니다.

## IRL에 대한 함의

행동으로부터 보상을 추론할 때 h를 명시적으로 latent confounder로 모델링하세요. 다른 h 값이 구별 가능한 행동 패턴을 생성하므로 (93.8% discriminator accuracy), h를 모델링하지 않으면 보상 추정을 혼란시킬 수 있습니다.

그러나 h를 전문성 예측에 사용하지 마세요. 모든 방법에 걸쳐 상관이 본질적으로 0입니다. 대신 van Opheusden features (AUC = 0.84)를 사용하세요.

관찰 가능한 미래가 있는 실제 행동 데이터에 접근할 수 있을 때는 rollout-free 방법을 사용하세요. 이들은 distribution mismatch를 제거하고 (+38% bias 제거) Bayesian uncertainty quantification과 함께 편향되지 않은 h 추정을 제공합니다.

## 인지과학에 대한 함의

개별 features는 약하지만 조합은 강하다는 발견은 전문성이 어떤 단일 차원을 최대화하는 것이 아니라 균형잡힌 heuristic 프로필을 개발하는 것을 포함함을 시사합니다. 개입은 고립된 기술이 아니라 heuristic 품질을 광범위하게 목표로 해야 합니다.

Planning depth는 trait가 아니라 state-dependent로 보입니다. 모든 플레이어가 E[h] ≈ 1.77 주변에 군집하지만 (0.38 스텝의 좁은 플레이어 간 분산), 개별 수는 각 플레이어 내에서 크게 변합니다. 이는 h가 개인 정체성보다 게임 맥락에 따라 더 많이 변함을 시사합니다.

Planning depth가 context-dependent라면, 관련 질문은 "이 사람이 얼마나 앞을 계획하는가"가 아니라 "이 사람이 언제 깊게 vs. 얕게 계획하는가"입니다. 임상 적용(불안, ADHD)의 경우, 평균 h가 아니라 맥락에 대한 h의 민감도를 분석하세요.

## 요약

Planning depth 추정을 위한 세 방법은 rollout artifacts로 인해 다른 절대값(1.78 ~ 2.87)을 생성하지만, 모두 전문성과 동일한 null 관계를 보입니다. Rollout-free가 가장 정확한 방법입니다 (+38% bias 제거).

인간은 4-in-a-row에서 근시안적으로 계획합니다 (E[h] = 1.78, 47%의 수가 반응적 h=1 planning). 이는 tree exploration metrics (PV depth = 6-7)보다 훨씬 얕습니다. PV depth가 검증 breadth를 측정하는 반면 behavioral h는 decision horizon을 측정하기 때문입니다.

Planning depth는 전문성을 예측하지 못하지만 (AUC = 53%, chance level), van Opheusden의 heuristic features는 예측합니다 (AUC = 84%). 전문성은 planning depth가 아니라 다변량 heuristic 품질을 반영합니다. 이는 전문가가 더 낮은 PV depth를 갖는다는 van Opheusden의 발견과 조화됩니다: 둘 다 전문성이 효율적 탐색을 가능케 하는 더 나은 heuristics로부터 나온다는 관점을 지지합니다.

IRL의 경우, h를 latent confounder로 모델링하되 전문성 예측에는 features를 사용하세요 (h가 아님).

**마지막 업데이트**: 2025-12-31
