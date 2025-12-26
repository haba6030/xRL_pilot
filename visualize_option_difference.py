"""
Visualize Option A vs Option B Difference

Create visual diagrams showing the key differences between two approaches.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_comparison_diagram():
    """Create visual comparison diagram"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ═══════════════════════════════════════════════════════
    # Option A Diagram
    # ═══════════════════════════════════════════════════════
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Option A: Pure Neural Network\n(Pedestrian 방식)',
                  fontsize=16, fontweight='bold', pad=20)

    # Step 1: Random initialization
    box1 = FancyBboxPatch((0.5, 7), 3, 1.5,
                          boxstyle="round,pad=0.1",
                          edgecolor='red', facecolor='lightcoral',
                          linewidth=2)
    ax1.add_patch(box1)
    ax1.text(2, 7.75, 'Random Init\n신경망 (🎲)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow 1
    arrow1 = FancyArrowPatch((2, 7), (2, 5.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='darkred')
    ax1.add_patch(arrow1)
    ax1.text(3.5, 6.25, '아무것도 모름\n무작위 행동',
            fontsize=9, style='italic', color='darkred')

    # Step 2: AIRL training
    box2 = FancyBboxPatch((0.5, 4), 3, 1.5,
                          boxstyle="round,pad=0.1",
                          edgecolor='orange', facecolor='lightyellow',
                          linewidth=2)
    ax1.add_patch(box2)
    ax1.text(2, 4.75, 'AIRL 학습\n(50K-100K steps)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow 2
    arrow2 = FancyArrowPatch((2, 4), (2, 2.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='darkorange')
    ax1.add_patch(arrow2)
    ax1.text(3.8, 3.25, '순수 학습\nReward 신호만',
            fontsize=9, style='italic', color='darkorange')

    # Step 3: Final policy
    box3 = FancyBboxPatch((0.5, 1), 3, 1.5,
                          boxstyle="round,pad=0.1",
                          edgecolor='green', facecolor='lightgreen',
                          linewidth=2)
    ax1.add_patch(box3)
    ax1.text(2, 1.75, 'Expert 행동\n✅ 학습 완료',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Characteristics box
    char_box = FancyBboxPatch((5, 1), 4.5, 7.5,
                             boxstyle="round,pad=0.15",
                             edgecolor='gray', facecolor='white',
                             linewidth=2, linestyle='--', alpha=0.3)
    ax1.add_patch(char_box)

    characteristics_a = [
        "✅ 이론적 순수성",
        "✅ Domain knowledge 無",
        "✅ IRL 능력 검증",
        "⚠️  학습 느림 (50K+)",
        "⚠️  불안정 가능성",
        "⚠️  많은 시간 소요"
    ]

    for i, char in enumerate(characteristics_a):
        ax1.text(7.25, 7.5 - i*0.9, char, fontsize=10,
                ha='center', va='center')

    # ═══════════════════════════════════════════════════════
    # Option B Diagram
    # ═══════════════════════════════════════════════════════
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Option B: BFS Distillation\n(van Opheusden 활용)',
                  fontsize=16, fontweight='bold', pad=20)

    # Step 1: BFS
    box1b = FancyBboxPatch((0.5, 8.5), 3, 1,
                           boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='lavender',
                           linewidth=2)
    ax2.add_patch(box1b)
    ax2.text(2, 9, 'van Opheusden\nBFS (h=4)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow 1b
    arrow1b = FancyArrowPatch((2, 8.5), (2, 7.5),
                             arrowstyle='->', mutation_scale=30,
                             linewidth=3, color='purple')
    ax2.add_patch(arrow1b)
    ax2.text(3.8, 8, '행동 데이터\n생성',
            fontsize=9, style='italic', color='purple')

    # Step 2: BC training
    box2b = FancyBboxPatch((0.5, 6), 3, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor='blue', facecolor='lightblue',
                           linewidth=2)
    ax2.add_patch(box2b)
    ax2.text(2, 6.75, 'BC 학습\n(BFS 모방)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow 2b
    arrow2b = FancyArrowPatch((2, 6), (2, 4.5),
                             arrowstyle='->', mutation_scale=30,
                             linewidth=3, color='darkblue')
    ax2.add_patch(arrow2b)
    ax2.text(3.8, 5.25, 'BFS 흉내\n꽤 괜찮음',
            fontsize=9, style='italic', color='darkblue')

    # Step 3: AIRL fine-tuning
    box3b = FancyBboxPatch((0.5, 3), 3, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor='orange', facecolor='lightyellow',
                           linewidth=2)
    ax2.add_patch(box3b)
    ax2.text(2, 3.75, 'AIRL Fine-tune\n(10K steps)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow 3b
    arrow3b = FancyArrowPatch((2, 3), (2, 1.5),
                             arrowstyle='->', mutation_scale=30,
                             linewidth=3, color='darkorange')
    ax2.add_patch(arrow3b)
    ax2.text(3.8, 2.25, 'Reward로\n미세 조정',
            fontsize=9, style='italic', color='darkorange')

    # Step 4: Final policy
    box4b = FancyBboxPatch((0.5, 0), 3, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor='green', facecolor='lightgreen',
                           linewidth=2)
    ax2.add_patch(box4b)
    ax2.text(2, 0.75, 'Expert 행동\n✅ 학습 완료',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Characteristics box
    char_box_b = FancyBboxPatch((5, 0), 4.5, 9.5,
                               boxstyle="round,pad=0.15",
                               edgecolor='gray', facecolor='white',
                               linewidth=2, linestyle='--', alpha=0.3)
    ax2.add_patch(char_box_b)

    characteristics_b = [
        "✅ 빠른 학습 (10K)",
        "✅ 안정적 수렴",
        "✅ 실용적 성능",
        "✅ 시간 절약",
        "⚠️  Domain 의존적",
        "⚠️  순수 IRL 아님",
        "⚠️  BC/IRL 구분 애매"
    ]

    for i, char in enumerate(characteristics_b):
        ax2.text(7.25, 8.8 - i*0.9, char, fontsize=10,
                ha='center', va='center')

    plt.tight_layout()
    plt.savefig('figures/option_a_vs_b_diagram.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: figures/option_a_vs_b_diagram.png")
    plt.close()


def create_training_curve_comparison():
    """Create hypothetical training curve comparison"""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate hypothetical training curves
    timesteps = np.linspace(0, 100000, 1000)

    # Option A: Slow start, gradual improvement
    performance_a = 1 / (1 + np.exp(-(timesteps - 50000) / 10000))
    performance_a = performance_a * 0.9 + 0.05  # Scale to [0.05, 0.95]
    # Add noise
    np.random.seed(42)
    performance_a += np.random.normal(0, 0.02, len(timesteps))

    # Option B: Fast start (BC), quick improvement
    performance_b = 1 / (1 + np.exp(-(timesteps - 20000) / 5000))
    performance_b = performance_b * 0.85 + 0.35  # Scale to [0.35, 1.0] (starts higher)
    # Add noise
    np.random.seed(43)
    performance_b += np.random.normal(0, 0.015, len(timesteps))

    # Expert baseline
    expert_performance = 0.9

    # Plot
    ax.plot(timesteps, performance_a, label='Option A (Pure NN)',
           linewidth=2.5, color='coral')
    ax.plot(timesteps, performance_b, label='Option B (BC-initialized)',
           linewidth=2.5, color='skyblue')
    ax.axhline(expert_performance, color='green', linestyle='--',
              linewidth=2, label='Expert Performance', alpha=0.7)

    # Annotations
    ax.annotate('Random init\n(무작위 행동)', xy=(5000, 0.15),
               xytext=(15000, 0.05),
               arrowprops=dict(arrowstyle='->', color='coral', lw=2),
               fontsize=10, color='darkred')

    ax.annotate('BC warm start\n(이미 괜찮음)', xy=(5000, 0.45),
               xytext=(15000, 0.6),
               arrowprops=dict(arrowstyle='->', color='skyblue', lw=2),
               fontsize=10, color='darkblue')

    ax.annotate('Option A 수렴', xy=(70000, 0.88),
               xytext=(75000, 0.75),
               arrowprops=dict(arrowstyle='->', color='coral', lw=2),
               fontsize=10, color='darkred')

    ax.annotate('Option B 수렴', xy=(30000, 0.92),
               xytext=(40000, 1.05),
               arrowprops=dict(arrowstyle='->', color='skyblue', lw=2),
               fontsize=10, color='darkblue')

    # Labels and title
    ax.set_xlabel('Training Timesteps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance (Expert Similarity)', fontsize=13, fontweight='bold')
    ax.set_title('Option A vs Option B: Training Curves (Hypothetical)',
                fontsize=15, fontweight='bold', pad=15)

    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # Add text box
    textstr = 'Key Observations:\n' \
              '• Option B starts higher (BC)\n' \
              '• Option B converges faster (~30K)\n' \
              '• Option A needs more steps (~70K)\n' \
              '• Both reach similar final performance'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('figures/option_a_vs_b_training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: figures/option_a_vs_b_training_curves.png")
    plt.close()


def create_reward_network_diagram():
    """Emphasize that reward network is THE SAME for both options"""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, '중요: Reward Network는 둘 다 동일!',
           ha='center', fontsize=18, fontweight='bold', color='darkgreen')

    # Common reward network (top)
    reward_box = FancyBboxPatch((3, 7), 6, 1.5,
                               boxstyle="round,pad=0.15",
                               edgecolor='darkgreen', facecolor='lightgreen',
                               linewidth=3)
    ax.add_patch(reward_box)
    ax.text(6, 7.75, 'Reward Network (공통)\nBasicRewardNet(obs=89-dim, NO h)',
           ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrow splitting to two options
    arrow_left = FancyArrowPatch((4.5, 7), (3, 5.5),
                                arrowstyle='->', mutation_scale=30,
                                linewidth=3, color='coral')
    ax.add_patch(arrow_left)

    arrow_right = FancyArrowPatch((7.5, 7), (9, 5.5),
                                 arrowstyle='->', mutation_scale=30,
                                 linewidth=3, color='skyblue')
    ax.add_patch(arrow_right)

    # Option A generator
    gen_a_box = FancyBboxPatch((0.5, 4), 5, 1.5,
                              boxstyle="round,pad=0.1",
                              edgecolor='coral', facecolor='mistyrose',
                              linewidth=2)
    ax.add_patch(gen_a_box)
    ax.text(3, 4.75, 'Option A Generator\nPPO (Random Init, NO BC)',
           ha='center', va='center', fontsize=11, fontweight='bold')

    # Option B generator
    gen_b_box = FancyBboxPatch((6.5, 4), 5, 1.5,
                              boxstyle="round,pad=0.1",
                              edgecolor='skyblue', facecolor='aliceblue',
                              linewidth=2)
    ax.add_patch(gen_b_box)
    ax.text(9, 4.75, 'Option B Generator\nPPO (BC-initialized from BFS)',
           ha='center', va='center', fontsize=11, fontweight='bold')

    # AIRL training (both)
    airl_a_box = FancyBboxPatch((0.5, 2), 5, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='orange', facecolor='lightyellow',
                               linewidth=2)
    ax.add_patch(airl_a_box)
    ax.text(3, 2.75, 'AIRL Training\n50K-100K timesteps',
           ha='center', va='center', fontsize=11)

    airl_b_box = FancyBboxPatch((6.5, 2), 5, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='orange', facecolor='lightyellow',
                               linewidth=2)
    ax.add_patch(airl_b_box)
    ax.text(9, 2.75, 'AIRL Training\n10K timesteps',
           ha='center', va='center', fontsize=11)

    # Final policies
    final_a = FancyBboxPatch((0.5, 0.25), 5, 1,
                            boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen',
                            linewidth=2)
    ax.add_patch(final_a)
    ax.text(3, 0.75, 'Trained Policy A',
           ha='center', va='center', fontsize=10, fontweight='bold')

    final_b = FancyBboxPatch((6.5, 0.25), 5, 1,
                            boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen',
                            linewidth=2)
    ax.add_patch(final_b)
    ax.text(9, 0.75, 'Trained Policy B',
           ha='center', va='center', fontsize=10, fontweight='bold')

    # Key point box
    key_box = FancyBboxPatch((1, 5.8), 10, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor='darkgreen', facecolor='honeydew',
                            linewidth=2)
    ax.add_patch(key_box)
    ax.text(6, 6.2, '✅ 같은 Reward Network = 같은 "목표"  |  차이는 오직 Generator 초기화',
           ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    plt.savefig('figures/reward_network_same.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: figures/reward_network_same.png")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)

    print("Creating comparison diagrams...")
    create_comparison_diagram()
    create_training_curve_comparison()
    create_reward_network_diagram()

    print("\n" + "=" * 80)
    print("✅ All diagrams created successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. figures/option_a_vs_b_diagram.png")
    print("  2. figures/option_a_vs_b_training_curves.png")
    print("  3. figures/reward_network_same.png")
