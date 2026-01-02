# Methods for additional analysis 
현재 방식에서의 어느정도 문제점을 생각해봤습니다.
지금은 실제 raw_data를 활용해서, 즉 실제 실험 데이터로 모델을 학습한 후, 이를 다시 동일한 데이터에 적용해서 bayesian probability를 업데이트하는 방식입니다.
이는 state들이 겹쳐있고, 베이지안 방법에 부합합니다. 그러나, 학습 데이터와 추론 데이터가 동일하다는 단점이 있습니다.

## 1. MusIK-Inspired
**METHOD**
- h를 마지막 \psi(t)로드 시 \psi(t-h), 즉 h step 전에 t를 고려했을 때 행동 확률분포를 활용하도록 하는 것입니다.
- 이러면 모든 h에 대해서 trajectory based action probability를 구하면서도 동시에 이를 활용하여 planning depth를 계산할 수 있겠죠. 
- 이 경우 기존 의사코드처럼 random rollout - probability computing - multistep inverse learning을 진행한다면 데이터 중복의 문제가 없을 것입니다.

### Pseudo-code
'''
Ψ(1) = ∅

for h = 2 to H:

    # IKDP(h)
    for t = h - 1 down to 1:

        D_t = ∅
        repeat n times:
            sample π_rollin ∈ Ψ(t)
            execute π_rollin to reach time t
            take random action a_t
            if t < h - 1:
                sample i ∈ S_h
                roll-out using \hat{π}(i, t+1) until time h
            observe (x_t, x_h, a_t, i)
            add to D_t

        learn inverse model f_t from D_t

        for each i ∈ S_h:
            define partial policy \hat{π}(i, t):
                at time t:
                    choose action using f_t
                at time t+1:
                    route to \hat{π}(·, t+1)

    Ψ(h) = { \hat{π}(i, 1) | i ∈ S_h }
'''

## 2. AIRL + PPO
- Yao(2024)에서 제시하였던 MPLP-IRL with two loop learning (finding r and fitting depth h)를 적용
- PPO 또는 generator를 학습한 후, 이를 이용하여 실제 agent (raw_data.csv)에 대해 AIRL로 discriminating

### Pseudo-code
'''
input: demonstrations from K experts: D_k, known dynamics T, horizon H
initialize Γ = {γ_k}  (or search space)

repeat (outer search over Γ):
    # 1) Reward inference (inner LP)
    r* = solve_LP_reward(Γ, expert_policies_or_estimates, distinguishable_states Ω_k)

    # 2) Evaluate objective g*(Γ)
    score = margin_objective(r*, Γ, Ω_k)

    update Γ to increase score   (grid search or Bayesian optimization)

output: (r*, Γ*)
'''

## TODO
- Reinforcement learning expert PhD의 관점에서, 현재 문제점에 대한 진단, 두 방법 선택 근거, 두 방법의 기대효과 및 실현 가능성을 고려해서 피드백을 부탁합니다. 
- 또한, 두 방법을 적용할 때 더 나은 방법이 있을지, 데이터 구조나 현재 목적에 맞게 고려하여 어떻게 적용할지 고민해봅시다. 