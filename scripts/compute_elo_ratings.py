#!/usr/bin/env python3
"""
Compute Elo ratings for human participants from 4-in-a-row games

This script:
1. Extracts game results from raw_data.csv
2. Computes Elo ratings using standard Elo algorithm
3. Classifies participants by expertise level
4. Saves results to data/ directory

Usage:
    python3 scripts/compute_elo_ratings.py

Output:
    - data/human_game_results.csv: Game-level results
    - data/human_elo_ratings.csv: Participant Elo ratings with expertise labels
    - data/human_games.pgn: PGN format (for BayesElo)
    - data/elo_summary.json: Summary statistics
    - data/README_ELO.md: Documentation

Author: Claude Code
Date: 2025-12-26
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# === Configuration ===
K_FACTOR = 32  # Elo K-factor
INITIAL_ELO = 1500.0  # Starting Elo rating


def check_four_in_row(pieces_str):
    """Check if there's 4 in a row in the 6x6 board"""
    board = np.array([int(c) for c in pieces_str]).reshape(6, 6)

    # Check horizontal
    for row in range(6):
        for col in range(3):
            if np.all(board[row, col:col+4] == 1):
                return True

    # Check vertical
    for row in range(3):
        for col in range(6):
            if np.all(board[row:row+4, col] == 1):
                return True

    # Check diagonal (down-right)
    for row in range(3):
        for col in range(3):
            if np.all([board[row+i, col+i] == 1 for i in range(4)]):
                return True

    # Check diagonal (down-left)
    for row in range(3):
        for col in range(3, 6):
            if np.all([board[row+i, col-i] == 1 for i in range(4)]):
                return True

    return False


def extract_game_results(raw_data_path='opendata/raw_data.csv'):
    """Extract game results from raw_data.csv"""
    print("📖 Loading raw data...")
    df = pd.read_csv(raw_data_path)

    # Filter human-vs-human games
    hvh = df[df['experiment'] == 'human-vs-human'].copy()
    print(f"   Found {len(hvh)} human-vs-human moves")

    # Identify games (new game = empty board)
    hvh['is_new_game'] = (hvh['black_pieces'] == '0' * 36) & (hvh['white_pieces'] == '0' * 36)

    game_id = 0
    game_ids = []
    for i, row in hvh.iterrows():
        if row['is_new_game']:
            game_id += 1
        game_ids.append(game_id)

    hvh['game_id'] = game_ids
    print(f"   Identified {hvh['game_id'].max()} games")

    # Determine game outcomes
    game_results = []

    for gid in range(1, hvh['game_id'].max() + 1):
        game_moves = hvh[hvh['game_id'] == gid].copy()

        if len(game_moves) == 0:
            continue

        black_moves = game_moves[game_moves['color'] == 'Black']
        white_moves = game_moves[game_moves['color'] == 'White']

        if len(black_moves) == 0 or len(white_moves) == 0:
            continue

        black_player = black_moves['participant'].iloc[0]
        white_player = white_moves['participant'].iloc[0]
        last_move = game_moves.iloc[-1]

        black_won = check_four_in_row(last_move['black_pieces'])
        white_won = check_four_in_row(last_move['white_pieces'])

        if black_won:
            result = '1-0'
            winner = int(black_player)
        elif white_won:
            result = '0-1'
            winner = int(white_player)
        else:
            result = '1/2-1/2'
            winner = None

        game_results.append({
            'game_id': gid,
            'black_player': int(black_player),
            'white_player': int(white_player),
            'result': result,
            'winner': winner,
            'num_moves': len(game_moves)
        })

    results_df = pd.DataFrame(game_results)
    print(f"   Results: {(results_df['result'] == '1-0').sum()} Black wins, "
          f"{(results_df['result'] == '0-1').sum()} White wins, "
          f"{(results_df['result'] == '1/2-1/2').sum()} draws")

    return results_df


def compute_elo_ratings(results_df):
    """Compute Elo ratings from game results"""
    print("\n🎯 Computing Elo ratings...")

    # Initialize Elo ratings
    elo_ratings = {i: INITIAL_ELO for i in range(1, 41)}

    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10**((rating_b - rating_a) / 400))

    def update_elo(rating, actual, expected):
        return rating + K_FACTOR * (actual - expected)

    # Process each game
    for _, game in results_df.iterrows():
        white = game['white_player']
        black = game['black_player']
        result = game['result']

        white_elo = elo_ratings[white]
        black_elo = elo_ratings[black]

        white_expected = expected_score(white_elo, black_elo)
        black_expected = expected_score(black_elo, white_elo)

        if result == '1-0':
            white_actual, black_actual = 0.0, 1.0
        elif result == '0-1':
            white_actual, black_actual = 1.0, 0.0
        else:
            white_actual, black_actual = 0.5, 0.5

        elo_ratings[white] = update_elo(white_elo, white_actual, white_expected)
        elo_ratings[black] = update_elo(black_elo, black_actual, black_expected)

    # Create dataframe with stats
    stats = []
    for pid in range(1, 41):
        games = results_df[(results_df['white_player'] == pid) | (results_df['black_player'] == pid)]

        wins = len(games[games['winner'] == pid])
        losses = len(games[(games['winner'].notna()) & (games['winner'] != pid)])
        draws = len(games[games['winner'].isna()])
        total = len(games)

        stats.append({
            'participant': pid,
            'elo': elo_ratings[pid],
            'games_played': total,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / total if total > 0 else 0
        })

    stats_df = pd.DataFrame(stats).sort_values('elo', ascending=False)

    # Add expertise classification
    q25 = stats_df['elo'].quantile(0.25)
    q75 = stats_df['elo'].quantile(0.75)
    median_elo = stats_df['elo'].median()

    def classify_expertise(elo):
        if elo < q25:
            return 'novice'
        elif elo < q75:
            return 'intermediate'
        else:
            return 'expert'

    stats_df['expertise'] = stats_df['elo'].apply(classify_expertise)
    stats_df['expert_binary'] = stats_df['elo'] >= median_elo

    print(f"   Elo range: {stats_df['elo'].min():.1f} - {stats_df['elo'].max():.1f}")
    print(f"   Median: {median_elo:.1f}")
    print(f"   Experts (≥Q75={q75:.1f}): {(stats_df['expertise'] == 'expert').sum()}")
    print(f"   Novices (<Q25={q25:.1f}): {(stats_df['expertise'] == 'novice').sum()}")

    return stats_df, {'q25': q25, 'q75': q75, 'median': median_elo}


def create_pgn(results_df, output_path='data/human_games.pgn'):
    """Create PGN format file"""
    pgn_lines = []

    for _, game in results_df.iterrows():
        black = f"P{game['black_player']}"
        white = f"P{game['white_player']}"
        result = game['result']

        pgn_lines.append(f'[White "{white}"]')
        pgn_lines.append(f'[Black "{black}"]')
        pgn_lines.append(f'[Result "{result}"]')
        pgn_lines.append(result)
        pgn_lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(pgn_lines))


def save_summary(stats_df, thresholds, results_df, output_path='data/elo_summary.json'):
    """Save summary statistics"""
    summary = {
        'total_participants': int(len(stats_df)),
        'total_games': int(len(results_df)),
        'elo_median': float(thresholds['median']),
        'elo_q25': float(thresholds['q25']),
        'elo_q75': float(thresholds['q75']),
        'elo_min': float(stats_df['elo'].min()),
        'elo_max': float(stats_df['elo'].max()),
        'novice_count': int((stats_df['expertise'] == 'novice').sum()),
        'intermediate_count': int((stats_df['expertise'] == 'intermediate').sum()),
        'expert_count': int((stats_df['expertise'] == 'expert').sum()),
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    """Main execution"""
    print("=" * 60)
    print("Computing Elo Ratings for Human Participants")
    print("=" * 60)

    # Create output directory
    Path('data').mkdir(exist_ok=True)

    # Step 1: Extract game results
    results_df = extract_game_results()
    results_df.to_csv('data/human_game_results.csv', index=False)
    print(f"   ✅ Saved data/human_game_results.csv")

    # Step 2: Compute Elo ratings
    stats_df, thresholds = compute_elo_ratings(results_df)
    stats_df.to_csv('data/human_elo_ratings.csv', index=False)
    print(f"   ✅ Saved data/human_elo_ratings.csv")

    # Step 3: Create PGN file
    create_pgn(results_df)
    print(f"   ✅ Saved data/human_games.pgn")

    # Step 4: Save summary
    save_summary(stats_df, thresholds, results_df)
    print(f"   ✅ Saved data/elo_summary.json")

    print("\n" + "=" * 60)
    print("✅ Elo computation complete!")
    print("=" * 60)
    print(f"\nExpert participants (Elo ≥ {thresholds['q75']:.1f}):")
    experts = stats_df[stats_df['expertise'] == 'expert']['participant'].tolist()
    print(f"   {experts}")

    print(f"\nFiles saved in data/ directory:")
    print(f"   - human_elo_ratings.csv (main file)")
    print(f"   - human_game_results.csv")
    print(f"   - elo_summary.json")
    print(f"   - human_games.pgn")
    print(f"   - README_ELO.md (documentation)")


if __name__ == '__main__':
    main()
