"""
analyze_results.py
------------------
Reads all experiment logs from results/exp*/log*.txt, tests the best
evolved agent from each run, and produces plots + a summary table.

Usage:
    python analyze_results.py                  # test 100 episodes per agent
    python analyze_results.py 200              # test 200 episodes per agent
"""

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')           # no display needed
import matplotlib.pyplot as plt
import gymnasium as gym

# ── import simulation helpers from the main script ──────────────────────────
# The filename contains dashes, so we can't use a plain import statement.
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    'ne_lander',
    pathlib.Path(__file__).parent / 'NE-LunarLander-alunos.py'
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

simulate   = _mod.simulate
load_bests = _mod.load_bests
GRAVITY           = _mod.GRAVITY
ENABLE_WIND       = _mod.ENABLE_WIND
WIND_POWER        = _mod.WIND_POWER
TURBULENCE_POWER  = _mod.TURBULENCE_POWER

# ── config ───────────────────────────────────────────────────────────────────
RESULTS_DIR  = 'results'
PLOTS_DIR    = 'plots'
TEST_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# Human-readable label for each experiment (from Table 2 in enunciado)
EXP_LABELS = {
    1: 'E1  mut=0.008 cross=0.5 elite=0',
    2: 'E2  mut=0.050 cross=0.5 elite=0',
    3: 'E3  mut=0.008 cross=0.9 elite=0',
    4: 'E4  mut=0.050 cross=0.9 elite=0',
    5: 'E5  mut=0.008 cross=0.5 elite=1',
    6: 'E6  mut=0.050 cross=0.5 elite=1',
    7: 'E7  mut=0.008 cross=0.9 elite=1',
    8: 'E8  mut=0.050 cross=0.9 elite=1',
}

# ── helpers ──────────────────────────────────────────────────────────────────

def find_experiments():
    """Return sorted list of experiment IDs that have at least one log file."""
    ids = []
    if not os.path.isdir(RESULTS_DIR):
        return ids
    for name in sorted(os.listdir(RESULTS_DIR)):
        if name.startswith('exp'):
            try:
                ids.append(int(name[3:]))
            except ValueError:
                pass
    return sorted(ids)


def load_experiment_logs(exp_id):
    """
    Load all run logs for a given experiment.
    Returns a list of lists: fitness_curves[run][generation]
    and a list of best individuals: best_inds[run] = (shape, genotype).
    """
    exp_dir = os.path.join(RESULTS_DIR, f'exp{exp_id}')
    fitness_curves = []
    best_inds      = []

    run = 0
    while True:
        log_path = os.path.join(exp_dir, f'log{run}.txt')
        if not os.path.isfile(log_path):
            break
        bests = load_bests(log_path)
        # bests[gen] = (fitness, shape, genotype)
        curve = [b[0] for b in bests]
        fitness_curves.append(curve)
        # last entry = best of final generation
        best_inds.append((bests[-1][1], bests[-1][2]))
        run += 1

    return fitness_curves, best_inds


def test_agent(shape, genotype, n_episodes):
    """Run n_episodes and return (mean_fitness, success_rate)."""
    total_fit = 0.0
    total_suc = 0
    for _ in range(n_episodes):
        f, s = simulate(genotype, render_mode=None, seed=None)
        total_fit += f
        total_suc += int(s)
    return total_fit / n_episodes, total_suc / n_episodes


# ── main analysis ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    exp_ids = find_experiments()
    if not exp_ids:
        print(f"No experiment folders found in '{RESULTS_DIR}/'. "
              "Run the experiments first.")
        return

    print(f"Found experiments: {exp_ids}")
    print(f"Testing each best agent for {TEST_EPISODES} episodes...\n")

    # ── collect data ─────────────────────────────────────────────────────────
    all_curves   = {}   # exp_id -> list of curves (one per run)
    all_mean_fit = {}   # exp_id -> mean fitness across runs
    all_std_fit  = {}
    all_mean_suc = {}   # exp_id -> mean success rate
    all_std_suc  = {}

    summary_rows = []

    for exp_id in exp_ids:
        curves, best_inds = load_experiment_logs(exp_id)
        if not curves:
            print(f"  Exp {exp_id}: no log files found, skipping.")
            continue

        all_curves[exp_id] = curves

        run_fits = []
        run_sucs = []
        for run_idx, (shape, genotype) in enumerate(best_inds):
            print(f"  Exp {exp_id} | Run {run_idx+1}/{len(best_inds)} "
                  f"– testing {TEST_EPISODES} episodes …", end=' ', flush=True)
            mf, ms = test_agent(shape, genotype, TEST_EPISODES)
            run_fits.append(mf)
            run_sucs.append(ms)
            print(f"fit={mf:.2f}  success={ms:.2%}")

        all_mean_fit[exp_id] = np.mean(run_fits)
        all_std_fit[exp_id]  = np.std(run_fits)
        all_mean_suc[exp_id] = np.mean(run_sucs)
        all_std_suc[exp_id]  = np.std(run_sucs)

        summary_rows.append((exp_id, run_fits, run_sucs))

    # ── print summary table ──────────────────────────────────────────────────
    print("\n" + "="*75)
    print(f"{'Exp':<5} {'Label':<35} {'Avg Fitness':>12} {'Std Fit':>9} "
          f"{'Success%':>10} {'Std Suc':>9}")
    print("-"*75)
    for exp_id in exp_ids:
        if exp_id not in all_mean_fit:
            continue
        label = EXP_LABELS.get(exp_id, f'Exp {exp_id}')
        print(f"{exp_id:<5} {label:<35} "
              f"{all_mean_fit[exp_id]:>12.3f} "
              f"{all_std_fit[exp_id]:>9.3f} "
              f"{all_mean_suc[exp_id]*100:>9.1f}% "
              f"{all_std_suc[exp_id]*100:>8.1f}%")
    print("="*75)

    # ── Plot 1: fitness evolution curves per experiment ──────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    axes = axes.flatten()

    for ax_idx, exp_id in enumerate(exp_ids):
        if exp_id not in all_curves or ax_idx >= len(axes):
            continue
        ax     = axes[ax_idx]
        curves = all_curves[exp_id]

        # pad curves to same length (in case runs ended early)
        max_len = max(len(c) for c in curves)
        padded  = [c + [c[-1]] * (max_len - len(c)) for c in curves]
        arr     = np.array(padded)          # shape: (n_runs, n_gens)

        mean_c = arr.mean(axis=0)
        std_c  = arr.std(axis=0)
        gens   = np.arange(max_len)

        ax.plot(gens, mean_c, lw=2, label='mean')
        ax.fill_between(gens, mean_c - std_c, mean_c + std_c,
                        alpha=0.25, label='±1 std')
        for c in padded:
            ax.plot(gens, c, alpha=0.15, lw=0.8, color='grey')

        ax.set_title(EXP_LABELS.get(exp_id, f'Exp {exp_id}'), fontsize=8)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best fitness')
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Fitness Evolution per Experiment (mean ± std over 5 runs)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(PLOTS_DIR, 'fitness_curves.png')
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"\nSaved {path1}")

    # ── Plot 2: combined fitness curves (all experiments on one axes) ─────────
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_ids)))

    for color, exp_id in zip(colors, exp_ids):
        if exp_id not in all_curves:
            continue
        curves  = all_curves[exp_id]
        max_len = max(len(c) for c in curves)
        padded  = [c + [c[-1]] * (max_len - len(c)) for c in curves]
        arr     = np.array(padded)
        mean_c  = arr.mean(axis=0)
        std_c   = arr.std(axis=0)
        gens    = np.arange(max_len)
        label   = EXP_LABELS.get(exp_id, f'Exp {exp_id}')
        ax.plot(gens, mean_c, lw=2, color=color, label=label)
        ax.fill_between(gens, mean_c - std_c, mean_c + std_c,
                        alpha=0.12, color=color)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best fitness (mean over 5 runs)', fontsize=12)
    ax.set_title('All Experiments – Fitness Evolution', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    path2 = os.path.join(PLOTS_DIR, 'fitness_curves_combined.png')
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved {path2}")

    # ── Plot 3: bar chart – average fitness ───────────────────────────────────
    valid_ids = [eid for eid in exp_ids if eid in all_mean_fit]
    labels    = [f'E{eid}' for eid in valid_ids]
    means_f   = [all_mean_fit[eid] for eid in valid_ids]
    stds_f    = [all_std_fit[eid]  for eid in valid_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, means_f, yerr=stds_f, capsize=5,
                  color=plt.cm.tab10(np.linspace(0, 1, len(valid_ids))),
                  edgecolor='black', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Average Fitness', fontsize=12)
    ax.set_title('Average Fitness of Best Agent per Experiment\n'
                 f'(mean ± std over 5 runs, {TEST_EPISODES} test episodes each)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path3 = os.path.join(PLOTS_DIR, 'avg_fitness_bar.png')
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"Saved {path3}")

    # ── Plot 4: bar chart – success rate ──────────────────────────────────────
    means_s = [all_mean_suc[eid] * 100 for eid in valid_ids]
    stds_s  = [all_std_suc[eid]  * 100 for eid in valid_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, means_s, yerr=stds_s, capsize=5,
           color=plt.cm.tab10(np.linspace(0, 1, len(valid_ids))),
           edgecolor='black', linewidth=0.7)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Landing Success Rate per Experiment\n'
                 f'(mean ± std over 5 runs, {TEST_EPISODES} test episodes each)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path4 = os.path.join(PLOTS_DIR, 'success_rate_bar.png')
    plt.savefig(path4, dpi=150)
    plt.close()
    print(f"Saved {path4}")

    # ── Plot 5: grouped bars – mutation effect ────────────────────────────────
    # Group by (crossover, elitism) pair and show low vs high mutation side-by-side
    groups = [
        ('cross=0.5, elite=0', 1, 2),
        ('cross=0.9, elite=0', 3, 4),
        ('cross=0.5, elite=1', 5, 6),
        ('cross=0.9, elite=1', 7, 8),
    ]
    group_labels = [g[0] for g in groups]
    low_mut_suc  = []
    high_mut_suc = []
    for _, low_id, high_id in groups:
        low_mut_suc.append(all_mean_suc.get(low_id, 0) * 100)
        high_mut_suc.append(all_mean_suc.get(high_id, 0) * 100)

    x   = np.arange(len(groups))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, low_mut_suc,  w, label='mut=0.008', color='steelblue',
           edgecolor='black', linewidth=0.7)
    ax.bar(x + w/2, high_mut_suc, w, label='mut=0.050', color='tomato',
           edgecolor='black', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Effect of Mutation Rate on Success Rate\n'
                 '(grouped by crossover prob and elitism)',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path5 = os.path.join(PLOTS_DIR, 'mutation_effect.png')
    plt.savefig(path5, dpi=150)
    plt.close()
    print(f"Saved {path5}")

    # ── Plot 6: grouped bars – elitism effect ─────────────────────────────────
    groups_e = [
        ('mut=0.008, cross=0.5', 1, 5),
        ('mut=0.050, cross=0.5', 2, 6),
        ('mut=0.008, cross=0.9', 3, 7),
        ('mut=0.050, cross=0.9', 4, 8),
    ]
    no_elite_suc  = []
    yes_elite_suc = []
    for _, no_id, yes_id in groups_e:
        no_elite_suc.append(all_mean_suc.get(no_id,  0) * 100)
        yes_elite_suc.append(all_mean_suc.get(yes_id, 0) * 100)

    gl = [g[0] for g in groups_e]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, no_elite_suc,  w, label='elite=0', color='slategrey',
           edgecolor='black', linewidth=0.7)
    ax.bar(x + w/2, yes_elite_suc, w, label='elite=1', color='goldenrod',
           edgecolor='black', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(gl, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Effect of Elitism on Success Rate\n'
                 '(grouped by mutation rate and crossover prob)',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path6 = os.path.join(PLOTS_DIR, 'elitism_effect.png')
    plt.savefig(path6, dpi=150)
    plt.close()
    print(f"Saved {path6}")

    print(f"\nAll plots saved to '{PLOTS_DIR}/'")
    print("Done.")


if __name__ == '__main__':
    main()
