#!/usr/bin/env python3
"""
교차 검증 결과 분석 스크립트
K-Fold CV vs. 원본 결과 비교 및 로버스트성 검증
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    print("="*80)
    print("교차 검증 결과 분석")
    print("="*80)

    # 1. CV 결과 로드
    cv_results_path = Path("results/kfold_cv/combined_estimates.csv")

    if not cv_results_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {cv_results_path}")
        print("먼저 교차 검증을 실행하세요: ./run_kfold_cv_parallel.sh")
        return

    cv_df = pd.read_csv(cv_results_path)
    print(f"\n✅ CV 결과 로드 완료: {len(cv_df)}명")

    # 2. 원본 결과 로드 (있는 경우)
    original_path = Path("results/human_h_rollout_free_estimates.csv")

    if original_path.exists():
        orig_df = pd.read_csv(original_path)
        print(f"✅ 원본 결과 로드 완료: {len(orig_df)}명")

        # 매칭
        merged = orig_df.merge(
            cv_df[['participant', 'E[h]', 'fold']],
            on='participant',
            suffixes=('_orig', '_cv')
        )
        print(f"✅ 매칭 완료: {len(merged)}명")
    else:
        print(f"⚠️  원본 결과 없음: {original_path}")
        merged = None

    print("\n" + "="*80)
    print("1. CV 결과 기본 통계")
    print("="*80)

    # 전체 통계
    print(f"\n전체 참가자: {len(cv_df)}명")
    print(f"전체 이동 수: {cv_df['n_moves'].sum():,}개")
    print(f"평균 E[h]: {cv_df['E[h]'].mean():.3f} ± {cv_df['E[h]'].std():.3f}")
    print(f"중앙값 E[h]: {cv_df['E[h]'].median():.3f}")
    print(f"범위: [{cv_df['E[h]'].min():.3f}, {cv_df['E[h]'].max():.3f}]")

    # Planning depth 분포
    print(f"\nPlanning depth 사후 확률 분포:")
    for h in [1, 2, 3, 4]:
        prob = cv_df[f'P(h={h})'].mean()
        std = cv_df[f'P(h={h})'].std()
        print(f"  h={h}: {prob:.1%} ± {std:.1%}")

    # Fold별 통계
    print(f"\nFold별 결과:")
    for fold in sorted(cv_df['fold'].unique()):
        fold_df = cv_df[cv_df['fold'] == fold]
        print(f"  Fold {fold}: E[h] = {fold_df['E[h]'].mean():.3f} "
              f"± {fold_df['E[h]'].std():.3f}, n = {len(fold_df)}명")

    # Fold 간 분산 분석 (ANOVA)
    fold_groups = [cv_df[cv_df['fold'] == f]['E[h]'].values
                   for f in sorted(cv_df['fold'].unique())]
    f_stat, p_val = stats.f_oneway(*fold_groups)
    print(f"\nFold 간 ANOVA: F = {f_stat:.3f}, p = {p_val:.3e}")
    if p_val > 0.05:
        print("  ✅ Fold 간 유의미한 차이 없음 (안정적인 추정)")
    else:
        print("  ⚠️  Fold 간 유의미한 차이 있음 (불안정 가능성)")

    if merged is not None:
        print("\n" + "="*80)
        print("2. CV vs. 원본 비교 (로버스트성 검증)")
        print("="*80)

        # 평균 비교
        print(f"\n평균 E[h]:")
        print(f"  원본:  {merged['E[h]_orig'].mean():.3f} ± {merged['E[h]_orig'].std():.3f}")
        print(f"  CV:    {merged['E[h]_cv'].mean():.3f} ± {merged['E[h]_cv'].std():.3f}")

        # 차이
        diff = merged['E[h]_cv'] - merged['E[h]_orig']
        print(f"\n차이 (CV - 원본):")
        print(f"  평균: {diff.mean():.3f} ± {diff.std():.3f}")
        print(f"  중앙값: {diff.median():.3f}")

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(merged['E[h]_cv'], merged['E[h]_orig'])
        print(f"\nPaired t-test: t = {t_stat:.3f}, p = {p_val:.3e}")

        # Correlation
        r, p = stats.pearsonr(merged['E[h]_orig'], merged['E[h]_cv'])
        print(f"\nPearson 상관계수:")
        print(f"  r = {r:.3f}, p = {p:.3e}")

        if r > 0.8:
            print("  ✅ 높은 상관관계 (로버스트한 추정)")
        elif r > 0.6:
            print("  ⚠️  중간 상관관계 (일부 불안정성)")
        else:
            print("  ❌ 낮은 상관관계 (불안정한 추정)")

        # Spearman 상관계수 (순위 기반)
        rho, p = stats.spearmanr(merged['E[h]_orig'], merged['E[h]_cv'])
        print(f"\nSpearman 상관계수:")
        print(f"  ρ = {rho:.3f}, p = {p:.3e}")

        # 시각화
        print("\n" + "="*80)
        print("3. 시각화 생성")
        print("="*80)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # (1) Scatter plot: CV vs. Original
        ax = axes[0, 0]
        ax.scatter(merged['E[h]_orig'], merged['E[h]_cv'], alpha=0.6, s=50)

        # 대각선
        lim_min = min(merged['E[h]_orig'].min(), merged['E[h]_cv'].min())
        lim_max = max(merged['E[h]_orig'].max(), merged['E[h]_cv'].max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                'r--', label='Perfect agreement', linewidth=2)

        # 회귀선
        z = np.polyfit(merged['E[h]_orig'], merged['E[h]_cv'], 1)
        p = np.poly1d(z)
        ax.plot(merged['E[h]_orig'].sort_values(),
                p(merged['E[h]_orig'].sort_values()),
                'b-', label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}', linewidth=2)

        ax.set_xlabel('E[h] - 원본 (학습/테스트 중복)', fontsize=12)
        ax.set_ylabel('E[h] - CV (학습/테스트 분리)', fontsize=12)
        ax.set_title(f'CV vs. 원본 비교 (r={r:.3f}, p={p:.2e})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2) Bland-Altman plot
        ax = axes[0, 1]
        mean_h = (merged['E[h]_orig'] + merged['E[h]_cv']) / 2
        diff_h = merged['E[h]_cv'] - merged['E[h]_orig']

        ax.scatter(mean_h, diff_h, alpha=0.6, s=50)
        ax.axhline(diff_h.mean(), color='r', linestyle='--',
                   label=f'Mean diff: {diff_h.mean():.3f}')
        ax.axhline(diff_h.mean() + 1.96*diff_h.std(), color='gray',
                   linestyle=':', label=f'±1.96 SD')
        ax.axhline(diff_h.mean() - 1.96*diff_h.std(), color='gray', linestyle=':')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xlabel('평균 E[h]', fontsize=12)
        ax.set_ylabel('차이 (CV - 원본)', fontsize=12)
        ax.set_title('Bland-Altman Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (3) Distribution comparison
        ax = axes[1, 0]
        ax.hist(merged['E[h]_orig'], bins=20, alpha=0.5, label='원본', color='blue')
        ax.hist(merged['E[h]_cv'], bins=20, alpha=0.5, label='CV', color='red')
        ax.set_xlabel('E[h]', fontsize=12)
        ax.set_ylabel('빈도', fontsize=12)
        ax.set_title('E[h] 분포 비교', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # (4) Fold별 E[h] 분포
        ax = axes[1, 1]
        fold_data = [cv_df[cv_df['fold'] == f]['E[h]'].values
                     for f in sorted(cv_df['fold'].unique())]
        bp = ax.boxplot(fold_data, labels=[f'Fold {i}' for i in range(1, 6)],
                        patch_artist=True)

        # 색상
        colors = plt.cm.Set3(range(5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # 전체 평균선
        ax.axhline(cv_df['E[h]'].mean(), color='r', linestyle='--',
                   linewidth=2, label=f'전체 평균: {cv_df["E[h]"].mean():.3f}')

        ax.set_ylabel('E[h]', fontsize=12)
        ax.set_title('Fold별 E[h] 분포', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # 저장
        output_path = Path("results/kfold_cv/cv_validation_plots.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 시각화 저장: {output_path}")

        # 추가 플롯: Planning depth 분포
        fig2, ax = plt.subplots(figsize=(10, 6))

        h_probs = cv_df[[f'P(h={h})' for h in [1, 2, 3, 4]]].mean()
        h_stds = cv_df[[f'P(h={h})' for h in [1, 2, 3, 4]]].std()

        x_pos = np.arange(4)
        ax.bar(x_pos, h_probs, yerr=h_stds, alpha=0.7,
               color=plt.cm.viridis(np.linspace(0, 1, 4)),
               capsize=5, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'h={h}' for h in [1, 2, 3, 4]])
        ax.set_ylabel('사후 확률 (평균)', fontsize=12)
        ax.set_xlabel('Planning depth', fontsize=12)
        ax.set_title('Planning Depth 사후 확률 분포 (CV)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 값 표시
        for i, (prob, std) in enumerate(zip(h_probs, h_stds)):
            ax.text(i, prob + std + 0.02, f'{prob:.1%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        output_path2 = Path("results/kfold_cv/planning_depth_distribution.png")
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"✅ Planning depth 분포 저장: {output_path2}")

    # 요약 리포트 저장
    print("\n" + "="*80)
    print("4. 요약 리포트 생성")
    print("="*80)

    report_lines = []
    report_lines.append("# 교차 검증 결과 요약 리포트\n")
    report_lines.append(f"생성 일시: {pd.Timestamp.now()}\n\n")

    report_lines.append("## 1. CV 결과\n\n")
    report_lines.append(f"- 전체 참가자: {len(cv_df)}명\n")
    report_lines.append(f"- 전체 이동 수: {cv_df['n_moves'].sum():,}개\n")
    report_lines.append(f"- 평균 E[h]: {cv_df['E[h]'].mean():.3f} ± {cv_df['E[h]'].std():.3f}\n")
    report_lines.append(f"- 중앙값 E[h]: {cv_df['E[h]'].median():.3f}\n")
    report_lines.append(f"- 범위: [{cv_df['E[h]'].min():.3f}, {cv_df['E[h]'].max():.3f}]\n\n")

    report_lines.append("## 2. Planning Depth 분포\n\n")
    for h in [1, 2, 3, 4]:
        prob = cv_df[f'P(h={h})'].mean()
        std = cv_df[f'P(h={h})'].std()
        report_lines.append(f"- h={h}: {prob:.1%} ± {std:.1%}\n")

    report_lines.append("\n## 3. Fold별 안정성\n\n")
    report_lines.append(f"| Fold | E[h] 평균 | E[h] 표준편차 | 참가자 수 |\n")
    report_lines.append(f"|------|-----------|---------------|----------|\n")
    for fold in sorted(cv_df['fold'].unique()):
        fold_df = cv_df[cv_df['fold'] == fold]
        report_lines.append(f"| {fold} | {fold_df['E[h]'].mean():.3f} | "
                          f"{fold_df['E[h]'].std():.3f} | {len(fold_df)} |\n")

    report_lines.append(f"\nFold 간 ANOVA: F = {f_stat:.3f}, p = {p_val:.3e}\n")

    if merged is not None:
        report_lines.append("\n## 4. 로버스트성 검증 (CV vs. 원본)\n\n")
        report_lines.append(f"- 원본 평균: {merged['E[h]_orig'].mean():.3f} ± "
                          f"{merged['E[h]_orig'].std():.3f}\n")
        report_lines.append(f"- CV 평균: {merged['E[h]_cv'].mean():.3f} ± "
                          f"{merged['E[h]_cv'].std():.3f}\n")
        report_lines.append(f"- 차이: {diff.mean():.3f} ± {diff.std():.3f}\n")

        t_stat, p_val_t = stats.ttest_rel(merged['E[h]_cv'], merged['E[h]_orig'])
        report_lines.append(f"- Paired t-test: t = {t_stat:.3f}, p = {p_val_t:.3e}\n")

        r, p_r = stats.pearsonr(merged['E[h]_orig'], merged['E[h]_cv'])
        report_lines.append(f"- Pearson 상관: r = {r:.3f}, p = {p_r:.3e}\n")

        rho, p_rho = stats.spearmanr(merged['E[h]_orig'], merged['E[h]_cv'])
        report_lines.append(f"- Spearman 상관: ρ = {rho:.3f}, p = {p_rho:.3e}\n")

        report_lines.append("\n## 5. 해석\n\n")
        if r > 0.8 and abs(diff.mean()) < 0.1:
            report_lines.append("✅ **결과 로버스트함**: CV와 원본 결과가 높은 상관관계를 보이며, "
                              "평균 차이가 작음. 학습/테스트 중복이 결과에 큰 영향을 미치지 않았음.\n\n")
        elif r > 0.6:
            report_lines.append("⚠️ **중간 수준 로버스트성**: CV와 원본 결과가 중간 정도의 상관관계를 보임. "
                              "일부 참가자에서 추정치가 불안정할 수 있음.\n\n")
        else:
            report_lines.append("❌ **로버스트성 낮음**: CV와 원본 결과의 상관관계가 낮음. "
                              "학습/테스트 중복이 결과에 영향을 미쳤을 가능성이 있음.\n\n")

    report_lines.append("## 6. 다음 단계\n\n")
    report_lines.append("1. CV 결과를 논문에 보고 (메인 결과로 사용)\n")
    report_lines.append("2. Expertise와의 상관관계 재분석 (CV 기반)\n")
    report_lines.append("3. Pedestrian 데이터에 동일한 CV 프레임워크 적용\n")

    report_path = Path("results/kfold_cv/CV_VALIDATION_REPORT.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"\n✅ 요약 리포트 저장: {report_path}")

    print("\n" + "="*80)
    print("✅ 분석 완료!")
    print("="*80)

if __name__ == "__main__":
    main()
