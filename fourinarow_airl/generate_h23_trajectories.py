"""
Generate h=2,3 Trajectories

Quick script to generate trajectories for h=2 and h=3
"""

import sys
import os

# Set h values
for h in [2, 3]:
    print(f"\n{'='*80}")
    print(f"Generating trajectories for h={h}")
    print(f"{'='*80}\n")

    # Run the main generation script
    os.system(f"python3 -c \"\nimport pickle\nimport numpy as np\nimport joblib\nfrom pathlib import Path\nimport copy\nimport sys\nsys.path.append('.')\nfrom env import FourInARowEnv\n\n# Load model\nmodel_path = Path('models/separate_h/model_h{h}.pkl')\ncheckpoint = joblib.load(model_path)\nmodel = checkpoint['model']\n\nprint(f'Loaded h={h} model')\n\n# Generate trajectories\ntrajectories = []\nall_actions = []\nrng = np.random.default_rng(42)\n\nfor ep in range(100):\n    if (ep + 1) % 10 == 0:\n        print(f'Episode {{ep+1}}/100')\n    \n    env = FourInARowEnv()\n    env.reset()\n    \n    observations = [env._get_observation()]\n    actions = []\n    \n    for step in range(36):\n        legal = env.get_legal_actions()\n        if len(legal) == 0:\n            break\n        \n        current_state = env._get_observation()\n        action_scores = np.zeros(36) - np.inf\n        \n        # Rollout for each action\n        for action in legal:\n            sim_env = copy.deepcopy(env)\n            sim_env.step(action)\n            \n            # Rollout h-1 more steps\n            for _ in range({h}-1):\n                sim_legal = sim_env.get_legal_actions()\n                if len(sim_legal) == 0:\n                    break\n                sim_env.step(rng.choice(sim_legal))\n            \n            future_state = sim_env._get_observation()\n            features = np.concatenate([current_state, future_state]).reshape(1, -1)\n            \n            probs = model.predict_proba(features)[0]\n            action_scores[action] = probs[action]\n        \n        # Softmax\n        legal_scores = action_scores[legal]\n        logits = np.log(legal_scores + 1e-10)\n        probs = np.exp(logits) / np.exp(logits).sum()\n        \n        chosen = rng.choice(legal, p=probs)\n        \n        obs, reward, terminated, truncated, info = env.step(chosen)\n        \n        actions.append(chosen)\n        observations.append(obs)\n        all_actions.append(chosen)\n        \n        if terminated or truncated:\n            break\n    \n    trajectories.append({{\n        'observations': observations,\n        'actions': actions,\n        'num_moves': len(actions)\n    }})\n\n# Save\noutput_dir = Path('data/separate_h_trajectories')\noutput_dir.mkdir(exist_ok=True, parents=True)\n\nwith open(output_dir / f'trajectories_h{h}.pkl', 'wb') as f:\n    pickle.dump(trajectories, f)\n\nwith open(output_dir / f'actions_h{h}.pkl', 'wb') as f:\n    pickle.dump(all_actions, f)\n\nprint(f'\\nSaved {{len(trajectories)}} episodes, {{len(all_actions)}} actions')\n\"")

print("\n" + "="*80)
print("DONE")
print("="*80)
