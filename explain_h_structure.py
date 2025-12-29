"""
h 적용 구조 및 분석 방법 설명

이 스크립트는 우리 시스템에서 h가 어떻게 사용되고 분석되는지 시각화합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 12))

# ═══════════════════════════════════════════════════════
# Part 1: h의 역할 구조도
# ═══════════════════════════════════════════════════════
ax1 = plt.subplot(2, 2, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Part 1: h의 역할 구조', fontsize=14, fontweight='bold', pad=20)

# Expert Data Generation (h 사용)
box1 = FancyBboxPatch((0.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1", 
                       edgecolor='red', facecolor='lightcoral', linewidth=2)
ax1.add_patch(box1)
ax1.text(2, 8.8, 'BFS Policy', ha='center', fontsize=10, fontweight='bold')
ax1.text(2, 8.3, 'depth_limit=h', ha='center', fontsize=9, style='italic', color='red')

# Arrow to expert data
arrow1 = FancyArrowPatch((2, 7.5), (2, 6.5), arrowstyle='->', mutation_scale=20, 
                         linewidth=2, color='red')
ax1.add_patch(arrow1)

# Expert Data
box2 = FancyBboxPatch((0.5, 5), 3, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='orange', facecolor='peachpuff', linewidth=2)
ax1.add_patch(box2)
ax1.text(2, 6, 'Expert Trajectories', ha='center', fontsize=10, fontweight='bold')
ax1.text(2, 5.5, '(h=1,2,4,8)', ha='center', fontsize=8, color='red')

# Arrow to AIRL
arrow2 = FancyArrowPatch((2, 5), (2, 4), arrowstyle='->', mutation_scale=20, 
                         linewidth=2, color='blue')
ax1.add_patch(arrow2)

# AIRL Training
box3 = FancyBboxPatch((0.5, 2.5), 3, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='blue', facecolor='lightblue', linewidth=2)
ax1.add_patch(box3)
ax1.text(2, 3.5, 'AIRL Training', ha='center', fontsize=10, fontweight='bold')
ax1.text(2, 3, 'h별 독립 학습', ha='center', fontsize=8)

# Reward Network (depth-agnostic)
box4 = FancyBboxPatch((5.5, 5), 3.5, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='green', facecolor='lightgreen', linewidth=3)
ax1.add_patch(box4)
ax1.text(7.25, 6, 'Reward Network', ha='center', fontsize=10, fontweight='bold')
ax1.text(7.25, 5.5, 'r(s,a,s\') - NO h!', ha='center', fontsize=9, 
         color='green', fontweight='bold')

# Generator (h-implicit)
box5 = FancyBboxPatch((5.5, 2.5), 3.5, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='purple', facecolor='plum', linewidth=2)
ax1.add_patch(box5)
ax1.text(7.25, 3.5, 'Generator (PPO)', ha='center', fontsize=10, fontweight='bold')
ax1.text(7.25, 3, 'h-implicit behavior', ha='center', fontsize=8, style='italic')

# Connection arrows
arrow3 = FancyArrowPatch((3.5, 3.1), (5.5, 5.5), arrowstyle='<->', mutation_scale=15, 
                         linewidth=1.5, color='gray', linestyle='--')
ax1.add_patch(arrow3)
arrow4 = FancyArrowPatch((3.5, 3.1), (5.5, 3.1), arrowstyle='<->', mutation_scale=15, 
                         linewidth=1.5, color='gray', linestyle='--')
ax1.add_patch(arrow4)

# Key principle box
key_box = FancyBboxPatch((0.5, 0.2), 8.5, 1.5, boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor='yellow', alpha=0.3, linewidth=2)
ax1.add_patch(key_box)
ax1.text(4.75, 1.3, '핵심 원칙:', ha='center', fontsize=10, fontweight='bold')
ax1.text(4.75, 0.9, 'h는 Expert 생성 시에만 사용 (BFS depth limit)', ha='center', fontsize=9)
ax1.text(4.75, 0.5, 'Reward network은 h를 모름 (depth-agnostic)', ha='center', fontsize=9)

# ═══════════════════════════════════════════════════════
# Part 2: Option A vs Option B
# ═══════════════════════════════════════════════════════
ax2 = plt.subplot(2, 2, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Part 2: Option A vs Option B', fontsize=14, fontweight='bold', pad=20)

# Option A
ax2.text(2.5, 9, 'Option A: Pure NN', ha='center', fontsize=12, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.5))

box_a1 = FancyBboxPatch((0.5, 7), 4, 1, boxstyle="round,pad=0.05", 
                        edgecolor='blue', facecolor='lightblue', linewidth=1.5)
ax2.add_patch(box_a1)
ax2.text(2.5, 7.5, '1. BFS(h) → Expert Data', ha='center', fontsize=9)

box_a2 = FancyBboxPatch((0.5, 5.5), 4, 1, boxstyle="round,pad=0.05", 
                        edgecolor='blue', facecolor='lightblue', linewidth=1.5)
ax2.add_patch(box_a2)
ax2.text(2.5, 6, '2. Random Init PPO', ha='center', fontsize=9)

box_a3 = FancyBboxPatch((0.5, 4), 4, 1, boxstyle="round,pad=0.05", 
                        edgecolor='blue', facecolor='lightblue', linewidth=1.5)
ax2.add_patch(box_a3)
ax2.text(2.5, 4.5, '3. AIRL (50K steps)', ha='center', fontsize=9)

ax2.text(2.5, 3, '결과: h=4만 성공', ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Option B
ax2.text(7.5, 9, 'Option B: BC-init', ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

box_b1 = FancyBboxPatch((5.5, 7), 4, 1, boxstyle="round,pad=0.05", 
                        edgecolor='red', facecolor='mistyrose', linewidth=1.5)
ax2.add_patch(box_b1)
ax2.text(7.5, 7.5, '1. BFS(h) → Expert Data', ha='center', fontsize=9)

box_b2 = FancyBboxPatch((5.5, 5.5), 4, 1, boxstyle="round,pad=0.05", 
                        edgecolor='red', facecolor='mistyrose', linewidth=1.5)
ax2.add_patch(box_b2)
ax2.text(7.5, 6, '2. BC from BFS → PPO', ha='center', fontsize=9)

box_b3 = FancyBboxPatch((5.5, 4), 4, 1, boxstyle="round,pad=0.05", 
                        edgecolor='red', facecolor='mistyrose', linewidth=1.5)
ax2.add_patch(box_b3)
ax2.text(7.5, 4.5, '3. AIRL fine-tune', ha='center', fontsize=9)

ax2.text(7.5, 3, '기대: 더 안정적 학습', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Comparison
comp_box = FancyBboxPatch((1, 0.5), 8, 1.5, boxstyle="round,pad=0.1", 
                          edgecolor='purple', facecolor='lavender', linewidth=2)
ax2.add_patch(comp_box)
ax2.text(5, 1.6, '비교 기준:', ha='center', fontsize=10, fontweight='bold')
ax2.text(5, 1.2, '• KL divergence (expert와의 유사도)', ha='center', fontsize=8)
ax2.text(5, 0.8, '• Trajectory length (게임 길이)', ha='center', fontsize=8)

# ═══════════════════════════════════════════════════════
# Part 3: h 분석 방법 (AIRL)
# ═══════════════════════════════════════════════════════
ax3 = plt.subplot(2, 2, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Part 3: AIRL에서 h 분석', fontsize=14, fontweight='bold', pad=20)

# Different h values
h_values = [1, 2, 4, 8]
x_positions = [1.5, 3.5, 5.5, 7.5]
colors = ['lightcoral', 'salmon', 'lightgreen', 'coral']
results = ['실패\n(KL=9.36)', '실패\n(KL=15.82)', '성공\n(KL=0.24)', '실패\n(KL=14.01)']

for i, (h, x, color, result) in enumerate(zip(h_values, x_positions, colors, results)):
    # h box
    if h == 4:
        box = FancyBboxPatch((x-0.7, 6.5), 1.4, 1.2, boxstyle="round,pad=0.1", 
                            edgecolor='green', facecolor=color, linewidth=3)
    else:
        box = FancyBboxPatch((x-0.7, 6.5), 1.4, 1.2, boxstyle="round,pad=0.1", 
                            edgecolor='gray', facecolor=color, linewidth=1.5)
    ax3.add_patch(box)
    ax3.text(x, 7.4, f'h={h}', ha='center', fontsize=11, fontweight='bold')
    ax3.text(x, 6.9, 'BFS expert', ha='center', fontsize=7)
    
    # Arrow
    arrow = FancyArrowPatch((x, 6.5), (x, 5.5), arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color='blue')
    ax3.add_patch(arrow)
    
    # AIRL
    result_box = FancyBboxPatch((x-0.7, 4), 1.4, 1.2, boxstyle="round,pad=0.1", 
                               edgecolor='blue', facecolor='lightblue', linewidth=1.5)
    ax3.add_patch(result_box)
    ax3.text(x, 4.6, result.split('\n')[0], ha='center', fontsize=9, fontweight='bold')
    ax3.text(x, 4.2, result.split('\n')[1], ha='center', fontsize=7)

# Metrics explanation
metrics_box = FancyBboxPatch((0.5, 1.5), 9, 2, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='lightyellow', linewidth=2)
ax3.add_patch(metrics_box)
ax3.text(5, 3.1, '평가 지표:', ha='center', fontsize=11, fontweight='bold')
ax3.text(5, 2.7, '1. KL Divergence: KL(expert || generated) - 낮을수록 좋음', 
         ha='center', fontsize=8)
ax3.text(5, 2.4, '2. Trajectory Length: Expert(8.3) vs Generated', 
         ha='center', fontsize=8)
ax3.text(5, 2.1, '3. Win Rate: 승률 (4-in-a-row 게임)', ha='center', fontsize=8)
ax3.text(5, 1.8, '→ h=4: KL=0.24, Length=6.7 (expert 8.3 대비 81%)', 
         ha='center', fontsize=8, color='green', fontweight='bold')

# Conclusion
ax3.text(5, 0.8, '결론: h=4에서만 expert data가 "학습 가능한" 품질', 
         ha='center', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# ═══════════════════════════════════════════════════════
# Part 4: h 분석 방법 (Human Data)
# ═══════════════════════════════════════════════════════
ax4 = plt.subplot(2, 2, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Part 4: Human Data에서 h 추정', fontsize=14, fontweight='bold', pad=20)

# Raw data
data_box = FancyBboxPatch((0.5, 8), 3, 1.2, boxstyle="round,pad=0.1", 
                         edgecolor='brown', facecolor='wheat', linewidth=2)
ax4.add_patch(data_box)
ax4.text(2, 8.8, 'Raw Data', ha='center', fontsize=10, fontweight='bold')
ax4.text(2, 8.4, '67K trials, 40명', ha='center', fontsize=8)

# Arrow to processing
arrow_p1 = FancyArrowPatch((2, 8), (2, 7), arrowstyle='->', mutation_scale=15, 
                          linewidth=2, color='blue')
ax4.add_patch(arrow_p1)

# Processing
proc_box = FancyBboxPatch((0.5, 5.5), 3, 1.2, boxstyle="round,pad=0.1", 
                         edgecolor='blue', facecolor='lightblue', linewidth=2)
ax4.add_patch(proc_box)
ax4.text(2, 6.5, 'RT 분석', ha='center', fontsize=10, fontweight='bold')
ax4.text(2, 6.1, 'response_time', ha='center', fontsize=8)

# Arrow to h estimation
arrow_p2 = FancyArrowPatch((2, 5.5), (2, 4.5), arrowstyle='->', mutation_scale=15, 
                          linewidth=2, color='red')
ax4.add_patch(arrow_p2)

# h estimation
h_est_box = FancyBboxPatch((0.5, 3), 3, 1.2, boxstyle="round,pad=0.1", 
                          edgecolor='red', facecolor='lightcoral', linewidth=2)
ax4.add_patch(h_est_box)
ax4.text(2, 4, 'h 추정', ha='center', fontsize=10, fontweight='bold')
ax4.text(2, 3.6, 'RT z-score → h', ha='center', fontsize=8)
ax4.text(2, 3.3, 'h ∈ [1.5, 4]', ha='center', fontsize=7, style='italic')

# Expertise calculation (parallel path)
arrow_e1 = FancyArrowPatch((3.5, 8.6), (5.5, 8.6), arrowstyle='->', mutation_scale=15, 
                          linewidth=2, color='green')
ax4.add_patch(arrow_e1)

exp_box = FancyBboxPatch((5.5, 8), 3, 1.2, boxstyle="round,pad=0.1", 
                        edgecolor='green', facecolor='lightgreen', linewidth=2)
ax4.add_patch(exp_box)
ax4.text(7, 8.8, 'Expertise Score', ha='center', fontsize=10, fontweight='bold')
ax4.text(7, 8.4, '경험 + RT 패턴', ha='center', fontsize=8)

# Arrow to correlation
arrow_corr1 = FancyArrowPatch((2, 3), (4.5, 1.5), arrowstyle='->', mutation_scale=15, 
                             linewidth=2, color='purple')
ax4.add_patch(arrow_corr1)
arrow_corr2 = FancyArrowPatch((7, 8), (5.5, 1.5), arrowstyle='->', mutation_scale=15, 
                             linewidth=2, color='purple')
ax4.add_patch(arrow_corr2)

# Correlation analysis
corr_box = FancyBboxPatch((3, 0.5), 4, 0.8, boxstyle="round,pad=0.1", 
                         edgecolor='purple', facecolor='plum', linewidth=3)
ax4.add_patch(corr_box)
ax4.text(5, 0.9, 'Correlation: h ↔ Expertise', ha='center', fontsize=10, fontweight='bold')

# Results (surprise!)
result_box = FancyBboxPatch((0.5, 0.5), 2, 2, boxstyle="round,pad=0.1", 
                           edgecolor='red', facecolor='mistyrose', linewidth=3)
ax4.add_patch(result_box)
ax4.text(1.5, 2.2, '결과 (충격!)', ha='center', fontsize=10, fontweight='bold', color='red')
ax4.text(1.5, 1.9, 'r = -0.71', ha='center', fontsize=9, fontweight='bold')
ax4.text(1.5, 1.6, '(p < 0.001)', ha='center', fontsize=8)
ax4.text(1.5, 1.2, 'Expert: h=2.27', ha='center', fontsize=8)
ax4.text(1.5, 0.9, 'Novice: h=2.88', ha='center', fontsize=8)

# Interpretation box
interp_box = FancyBboxPatch((7.5, 0.5), 2, 2, boxstyle="round,pad=0.1", 
                           edgecolor='orange', facecolor='lightyellow', linewidth=2)
ax4.add_patch(interp_box)
ax4.text(8.5, 2.2, '해석', ha='center', fontsize=10, fontweight='bold')
ax4.text(8.5, 1.8, 'RT는 h의', ha='center', fontsize=8)
ax4.text(8.5, 1.5, '나쁜 proxy?', ha='center', fontsize=8)
ax4.text(8.5, 1.1, 'Or 진짜로', ha='center', fontsize=8)
ax4.text(8.5, 0.8, 'Expert=shallow?', ha='center', fontsize=8)

# Method box
method_box = FancyBboxPatch((5, 3.5), 4.5, 3.5, boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='white', linewidth=1, alpha=0.8)
ax4.add_patch(method_box)
ax4.text(7.25, 6.7, 'RT-based h 추정 공식:', ha='center', fontsize=9, fontweight='bold')
ax4.text(5.3, 6.4, 'Step 1: RT z-score 계산', ha='left', fontsize=7)
ax4.text(5.5, 6.1, '  z = (RT - mean) / std', ha='left', fontsize=7, family='monospace')
ax4.text(5.3, 5.7, 'Step 2: z → h 매핑', ha='left', fontsize=7)
ax4.text(5.5, 5.4, '  z < -1: h=1.5 (매우 빠름)', ha='left', fontsize=7)
ax4.text(5.5, 5.1, '  -1 < z < 0: h=2-2.5', ha='left', fontsize=7)
ax4.text(5.5, 4.8, '  0 < z < 1: h=3-3.5', ha='left', fontsize=7)
ax4.text(5.5, 4.5, '  z > 1: h=4 (매우 느림)', ha='left', fontsize=7)
ax4.text(5.3, 4.1, 'Step 3: Expertise score', ha='left', fontsize=7)
ax4.text(5.5, 3.8, '  0.6×경험 + 0.4×RT패턴', ha='left', fontsize=7)

plt.tight_layout()
plt.savefig('figures/h_structure_analysis_explanation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: figures/h_structure_analysis_explanation.png")
plt.close()

# ═══════════════════════════════════════════════════════
# Create detailed table summary
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Title
fig.suptitle('h 적용 구조 및 분석 방법 종합', fontsize=16, fontweight='bold', y=0.98)

# Table data
table_data = [
    ['구분', '세부 항목', '설명', 'h의 역할'],
    ['', '', '', ''],
    ['1. Expert\nGeneration', 'BFS Policy', 'van Opheusden heuristic 사용\n6×6 board, depth-limited search', 
     'h = depth_limit\n직접 사용 (명시적)'],
    ['', 'Expert Data', 'BFS로 생성된 게임 trajectories\n(state, action, next_state) 시퀀스', 
     'h별로 다른 데이터\nh=1,2,4,8 각각 생성'],
    ['', '', '', ''],
    ['2. AIRL\nTraining', 'Reward Network', 'r(s, a, s\'): 89-dim → scalar\nNO h parameter!', 
     'h 사용 안 함\n(depth-agnostic)'],
    ['', 'Generator (PPO)', 'Policy: π(a|s), 89-dim obs\nh-implicit (behavior만 닮음)', 
     'h 직접 사용 안 함\nexpert behavior 모방'],
    ['', 'Training', 'Option A: random init, 50K steps\nOption B: BC init, fine-tune', 
     'h별로 독립 학습\n각 h마다 별도 모델'],
    ['', '', '', ''],
    ['3. AIRL\n평가', 'KL Divergence', 'KL(expert || generated)\naction distribution 유사도', 
     'h별 성능 비교\nh=4: 0.24 (최고)'],
    ['', 'Trajectory Length', 'Expert: 8.3 moves\nGenerated: h별로 다름', 
     'h=1,2,8: ~2 moves (실패)\nh=4: 6.7 moves (성공)'],
    ['', '', '', ''],
    ['4. Human\nData 분석', 'Response Time', '각 수를 두는 데 걸린 시간\n40명, 67K trials', 
     'RT → h 추정 (proxy)\nRT 길면 h↑ 가정'],
    ['', 'h Estimation', 'RT z-score 기반 매핑\nz < -1: h=1.5, z>1: h=4', 
     '개인별 h 추정값\nh ∈ [1.5, 4]'],
    ['', 'Expertise Score', '경험(trial 수) + RT 패턴\n0.6×exp + 0.4×RT', 
     'Expert vs Novice 분류\nmedian split'],
    ['', 'Correlation', 'Pearson: r = -0.71 (p<0.001)\nSpearman: ρ = -0.39 (p=0.012)', 
     'Expert: h=2.27\nNovice: h=2.88 (역전!)'],
    ['', '', '', ''],
    ['5. 핵심\n원칙', 'Depth-Agnostic', 'Reward는 h를 모름\n환경의 본질적 속성만 학습', 
     '이론적 정당성\n(Yao et al. 2024)'],
    ['', 'h-Implicit', 'Generator는 h를 parameter로 안 받음\nbehavior만 expert 닮음', 
     'BC로 h 전이\n명시적 사용 X'],
]

# Create table
table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.12, 0.15, 0.45, 0.28])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', fontsize=10)

# Style category rows
category_rows = [2, 5, 9, 12, 17]
for row in category_rows:
    cell = table[(row, 0)]
    cell.set_facecolor('#D6DCE4')
    cell.set_text_props(weight='bold', fontsize=10)

# Style separator rows
for row in [1, 4, 8, 11, 16]:
    for col in range(4):
        table[(row, col)].set_facecolor('#F2F2F2')
        table[(row, col)].set_height(0.3)

# Highlight key findings
for row in [9, 15]:  # AIRL result and correlation result
    for col in range(4):
        table[(row, col)].set_facecolor('#FFF2CC')

# Add borders
for key, cell in table.get_celld().items():
    cell.set_edgecolor('gray')
    cell.set_linewidth(0.5)

plt.savefig('figures/h_structure_table.png', dpi=150, bbox_inches='tight')
print("✓ Saved: figures/h_structure_table.png")
plt.close()

print("\n" + "="*80)
print("시각화 완료!")
print("="*80)
print("\n생성된 파일:")
print("  1. figures/h_structure_analysis_explanation.png - 4개 파트 다이어그램")
print("  2. figures/h_structure_table.png - 종합 테이블")
