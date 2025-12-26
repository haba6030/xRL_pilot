"""
Visualize Complete AIRL Pipeline

Create comprehensive diagrams showing the entire AIRL structure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np


def create_complete_pipeline():
    """Create complete AIRL pipeline diagram"""

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Complete AIRL Pipeline for 4-in-a-row',
                 fontsize=20, fontweight='bold', pad=20)

    # Step 0: Expert Data
    step0 = FancyBboxPatch((7, 12.5), 4, 1,
                           boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='lavender',
                           linewidth=3)
    ax.add_patch(step0)
    ax.text(9, 13, 'Step 0: Expert Data\n(Human or BFS)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow down
    arrow0 = FancyArrowPatch((9, 12.5), (9, 11.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='purple')
    ax.add_patch(arrow0)

    # Step 1: Environment
    step1 = FancyBboxPatch((6.5, 10.5), 5, 1,
                           boxstyle="round,pad=0.1",
                           edgecolor='blue', facecolor='lightblue',
                           linewidth=3)
    ax.add_patch(step1)
    ax.text(9, 11, 'Step 1: Environment Setup\nFourInARowEnv (89-dim obs)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow down
    arrow1 = FancyArrowPatch((9, 10.5), (9, 9.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='blue')
    ax.add_patch(arrow1)

    # Step 2: Generator (split into Option A and B)
    step2_box = FancyBboxPatch((3, 6.5), 12, 3,
                              boxstyle="round,pad=0.15",
                              edgecolor='gray', facecolor='white',
                              linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(step2_box)
    ax.text(9, 9.2, 'Step 2: Generator (Policy) - Choose Option!',
            ha='center', fontsize=12, fontweight='bold', color='red')

    # Split arrow
    arrow2a = FancyArrowPatch((9, 9), (5, 8.5),
                             arrowstyle='->', mutation_scale=25,
                             linewidth=2.5, color='coral')
    ax.add_patch(arrow2a)

    arrow2b = FancyArrowPatch((9, 9), (13, 8.5),
                             arrowstyle='->', mutation_scale=25,
                             linewidth=2.5, color='skyblue')
    ax.add_patch(arrow2b)

    # Option A
    optA = FancyBboxPatch((3.5, 7), 3, 1.5,
                         boxstyle="round,pad=0.1",
                         edgecolor='coral', facecolor='mistyrose',
                         linewidth=2.5)
    ax.add_patch(optA)
    ax.text(5, 7.75, 'Option A\nPure NN\n(Random Init)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Option B (multi-stage)
    optB_container = FancyBboxPatch((10, 6.8), 4.5, 1.9,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='skyblue', facecolor='aliceblue',
                                   linewidth=2.5)
    ax.add_patch(optB_container)

    ax.text(12.25, 8.5, 'Option B: BFS Distillation',
            ha='center', fontsize=9, fontweight='bold')

    # Option B stages
    ax.text(12.25, 8.0, '1. BFS(h) data', ha='center', fontsize=8)
    ax.text(12.25, 7.6, '2. BC training', ha='center', fontsize=8)
    ax.text(12.25, 7.2, '3. PPO wrapping', ha='center', fontsize=8)

    # Convergence arrows
    arrow_convA = FancyArrowPatch((5, 7), (7, 5.5),
                                 arrowstyle='->', mutation_scale=25,
                                 linewidth=2.5, color='coral')
    ax.add_patch(arrow_convA)

    arrow_convB = FancyArrowPatch((12.25, 6.8), (11, 5.5),
                                 arrowstyle='->', mutation_scale=25,
                                 linewidth=2.5, color='skyblue')
    ax.add_patch(arrow_convB)

    # Step 3: Reward Network
    step3 = FancyBboxPatch((6.5, 4.5), 5, 1,
                          boxstyle="round,pad=0.1",
                          edgecolor='green', facecolor='lightgreen',
                          linewidth=3)
    ax.add_patch(step3)
    ax.text(9, 5, 'Step 3: Reward Network\nBasicRewardNet (NO h!)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow down
    arrow3 = FancyArrowPatch((9, 4.5), (9, 3.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='green')
    ax.add_patch(arrow3)

    # Step 4: AIRL Training Loop
    step4_box = FancyBboxPatch((5, 1), 8, 2.5,
                              boxstyle="round,pad=0.15",
                              edgecolor='orange', facecolor='lightyellow',
                              linewidth=3)
    ax.add_patch(step4_box)
    ax.text(9, 3.2, 'Step 4: AIRL Training Loop',
            ha='center', fontsize=11, fontweight='bold')

    # AIRL loop details
    ax.text(9, 2.7, 'For each iteration:', ha='center', fontsize=9)
    ax.text(9, 2.4, '1. Generator rollout', ha='center', fontsize=8)
    ax.text(9, 2.1, '2. Discriminator update (Expert vs Gen)', ha='center', fontsize=8)
    ax.text(9, 1.8, '3. Generator update (using reward)', ha='center', fontsize=8)
    ax.text(9, 1.4, 'Option A: 50K-100K steps  |  Option B: 10K steps',
            ha='center', fontsize=8, style='italic', color='red')

    # Key information boxes
    key1 = FancyBboxPatch((0.5, 9), 2.5, 3.5,
                         boxstyle="round,pad=0.1",
                         edgecolor='gray', facecolor='wheat',
                         linewidth=2, alpha=0.7)
    ax.add_patch(key1)
    ax.text(1.75, 12.2, 'Key Points', ha='center', fontsize=10, fontweight='bold')
    key_text = [
        'depth h:',
        '- NOT in reward',
        '- NOT in obs',
        '- Only naming',
        '',
        'Reward:',
        '- depth-agnostic',
        '- Same arch',
        '- Separate train'
    ]
    for i, txt in enumerate(key_text):
        ax.text(1.75, 11.7 - i*0.35, txt, ha='center', fontsize=7)

    key2 = FancyBboxPatch((15, 9), 2.5, 3.5,
                         boxstyle="round,pad=0.1",
                         edgecolor='gray', facecolor='wheat',
                         linewidth=2, alpha=0.7)
    ax.add_patch(key2)
    ax.text(16.25, 12.2, 'Differences', ha='center', fontsize=10, fontweight='bold')
    diff_text = [
        'Option A:',
        '- Pure NN',
        '- Random init',
        '- Slow (50K+)',
        '',
        'Option B:',
        '- BC init',
        '- Fast (10K)',
        '- van Opheusden'
    ]
    for i, txt in enumerate(diff_text):
        ax.text(16.25, 11.7 - i*0.35, txt, ha='center', fontsize=7)

    # Bottom output
    output_box = FancyBboxPatch((6, 0.2), 6, 0.6,
                               boxstyle="round,pad=0.05",
                               edgecolor='darkgreen', facecolor='lightgreen',
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(9, 0.5, 'Output: Trained Policy + Learned Reward',
            ha='center', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/airl_complete_pipeline.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/airl_complete_pipeline.png")
    plt.close()


def create_depth_h_role_diagram():
    """Diagram showing where depth h is and isn't used"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Where h IS used
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Where depth h IS used',
                  fontsize=16, fontweight='bold', color='green', pad=20)

    uses = [
        ('Experiment Design', 'Each h={1,2,4,8} trained separately', 8.5),
        ('Option B: BFS Data', 'BFS(h=4) generates trajectories', 6.5),
        ('File Naming', 'airl_generator_h4.zip', 4.5),
        ('Metadata', 'Stored in metadata.pkl for reference', 2.5)
    ]

    for i, (title, desc, y) in enumerate(uses):
        box = FancyBboxPatch((1, y-0.4), 8, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen',
                            linewidth=2)
        ax1.add_patch(box)
        ax1.text(5, y+0.2, title, ha='center', fontsize=11, fontweight='bold')
        ax1.text(5, y-0.15, desc, ha='center', fontsize=9, style='italic')

    # Right: Where h is NOT used
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Where depth h is NOT used',
                  fontsize=16, fontweight='bold', color='red', pad=20)

    not_uses = [
        ('Reward Network', 'NO h parameter in architecture', 8.5),
        ('Observation Space', '89-dim: board + features (NO h)', 6.5),
        ('AIRL Algorithm', 'Standard AIRL (no depth)', 4.5),
        ('Policy Network', 'MLP with NO h input', 2.5)
    ]

    for i, (title, desc, y) in enumerate(not_uses):
        box = FancyBboxPatch((1, y-0.4), 8, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor='red', facecolor='mistyrose',
                            linewidth=2)
        ax2.add_patch(box)
        ax2.text(5, y+0.2, title, ha='center', fontsize=11, fontweight='bold')
        ax2.text(5, y-0.15, desc, ha='center', fontsize=9, style='italic')

    # Add big X mark on right side
    ax2.plot([1.5, 8.5], [8, 2], 'r-', linewidth=8, alpha=0.3)
    ax2.plot([1.5, 8.5], [2, 8], 'r-', linewidth=8, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/depth_h_role.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/depth_h_role.png")
    plt.close()


def create_implementation_status():
    """Current implementation status checklist"""

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Implementation Status Checklist',
                 fontsize=18, fontweight='bold', pad=20)

    # Define phases
    phases = [
        {
            'name': 'Phase 1: Infrastructure',
            'items': [
                ('Environment (env.py)', True),
                ('Features (features.py)', True),
                ('Data Loader (data_loader.py)', True),
                ('BFS Wrapper (bfs_wrapper.py)', True)
            ],
            'y_start': 8.5
        },
        {
            'name': 'Phase 2: Option B Pipeline',
            'items': [
                ('BFS Data Gen (generate_training_data.py)', True),
                ('BC Training (train_bc.py)', True),
                ('PPO Generator (create_ppo_generator.py)', True),
                ('Reward Network (create_reward_net.py)', True),
                ('AIRL Training (train_airl.py)', True)
            ],
            'y_start': 6.0
        },
        {
            'name': 'Phase 3: Option A Pipeline',
            'items': [
                ('Pure NN Generator (create_ppo_generator_pure_nn.py)', True),
                ('AIRL Training Pure (train_airl_pure_nn.py)', True)
            ],
            'y_start': 3.5
        },
        {
            'name': 'Phase 4: Evaluation',
            'items': [
                ('Comparison Script (compare_option_a_vs_b.py)', True),
                ('Visualization (visualize_*.py)', True),
                ('Documentation (*.md)', True)
            ],
            'y_start': 1.8
        },
        {
            'name': 'Phase 5: Experiments (TODO)',
            'items': [
                ('Option A Training (all h)', False),
                ('Option B Training (all h)', False),
                ('Performance Comparison', False),
                ('Depth Discrimination Test', False)
            ],
            'y_start': 0.1
        }
    ]

    x_start = 1
    for phase in phases:
        # Phase box
        num_items = len(phase['items'])
        box_height = num_items * 0.35 + 0.5

        phase_box = FancyBboxPatch((x_start, phase['y_start']), 12, box_height,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='blue', facecolor='aliceblue',
                                  linewidth=2, alpha=0.5)
        ax.add_patch(phase_box)

        # Phase name
        ax.text(x_start + 6, phase['y_start'] + box_height - 0.3,
                phase['name'],
                ha='center', fontsize=11, fontweight='bold')

        # Items
        for i, (item_name, completed) in enumerate(phase['items']):
            y = phase['y_start'] + box_height - 0.7 - i * 0.35

            # Checkbox
            checkbox_color = 'green' if completed else 'red'
            checkbox_text = '✓' if completed else '☐'

            ax.text(x_start + 0.3, y, checkbox_text,
                   fontsize=14, color=checkbox_color, fontweight='bold')

            # Item name
            ax.text(x_start + 0.8, y, item_name,
                   fontsize=9, va='center')

    # Summary box
    summary_box = FancyBboxPatch((0.5, 9), 13, 0.7,
                                boxstyle="round,pad=0.1",
                                edgecolor='darkgreen', facecolor='lightgreen',
                                linewidth=2)
    ax.add_patch(summary_box)

    total_items = sum(len(p['items']) for p in phases)
    completed_items = sum(sum(1 for _, done in p['items'] if done) for p in phases)

    ax.text(7, 9.35,
            f'Progress: {completed_items}/{total_items} Complete ({completed_items/total_items*100:.0f}%)',
            ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/implementation_status.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/implementation_status.png")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)

    print("Creating AIRL pipeline diagrams...")
    create_complete_pipeline()
    create_depth_h_role_diagram()
    create_implementation_status()

    print("\n" + "=" * 80)
    print("All pipeline diagrams created successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. figures/airl_complete_pipeline.png - Complete pipeline overview")
    print("  2. figures/depth_h_role.png - Where h is/isn't used")
    print("  3. figures/implementation_status.png - Current progress")
