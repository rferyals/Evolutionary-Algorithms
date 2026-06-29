# Neuroevolution for LunarLander

Evolves neural network controllers to soft-land a spacecraft in OpenAI Gymnasium's **LunarLander-v3** environment using a custom evolutionary algorithm. Includes a full experimental study comparing mutation rate, crossover probability, elitism, and wind conditions across 11 configurations.


---

## How it works

A population of neural networks is evolved over generations. Each network is encoded as a flat genotype (list of weights). The evolutionary loop:

1. **Evaluate** — simulate each individual in LunarLander-v3, score with a custom fitness function
2. **Select** — tournament selection (size 5)
3. **Reproduce** — uniform crossover + adaptive Gaussian mutation
4. **Survive** — optional elitism carries the best individual unchanged

### Network architecture

```
8 inputs → 12 hidden (tanh) → 2 outputs (continuous thrust)
```

### Fitness function

Penalizes distance from landing pad, unsafe velocity, and unstable orientation. Rewards soft leg contact and grants a +120 bonus for a successful landing. Penalizes hard landings and unnecessary movement near the pad.

---

## Experiments

11 experiments were run, each with 5 independent seeds:

| Exp | Mutation | Crossover | Elite | Pop | Gens | Wind |
|-----|----------|-----------|-------|-----|------|------|
| E1  | 0.008    | 0.5       | 0     | 100 | 100  | No   |
| E2  | 0.050    | 0.5       | 0     | 100 | 100  | No   |
| E3  | 0.008    | 0.9       | 0     | 100 | 100  | No   |
| E4  | 0.050    | 0.9       | 0     | 100 | 100  | No   |
| E5  | 0.008    | 0.5       | 1     | 100 | 100  | No   |
| E6  | 0.050    | 0.5       | 1     | 100 | 100  | No   |
| E7  | 0.008    | 0.9       | 1     | 100 | 100  | No   |
| E8  | 0.050    | 0.9       | 1     | 100 | 100  | No   |
| E9  | 0.050    | 0.5       | 1     | 100 | 100  | Yes  |
| E10 | 0.100    | 0.5       | 1     | 100 | 150  | Yes  |
| E11 | 0.050    | 0.9       | 0     | 150 | 150  | Yes  |

Results are saved to `results/exp<id>/log<run>.txt`. Analysis plots are in `plots/`.

---

## Requirements

```bash
pip install gymnasium numpy matplotlib
```

## Usage

**Run all experiments:**
```bash
python NE-LunarLander.py 0 experiments_config.txt
```

**Single run with current config:**
```bash
python NE-LunarLander.py 0
```

**Test an evolved controller (no visualisation):**
```bash
python NE-LunarLander.py 1 results/exp9/log0.txt
```

**Test with visualisation:**
```bash
python NE-LunarLander.py 2 results/exp9/log0.txt
```

**Analyse results and generate plots:**
```bash
python analyze_results.py
python analyze_results.py 200   # 200 test episodes per agent
```

---

## Project structure

```
NE-LunarLander.py          # Main evolutionary algorithm
analyze_results.py         # Result analysis and plot generation
experiments_config.txt     # Experiment parameter grid (Meta 1 — no wind)
wind_config.txt            # Wind experiment configs (Meta 2)
results/                   # Per-experiment log files
plots/                     # Generated analysis plots
Logs_wihtout_wind/         # Raw logs from no-wind runs
Logs_with_wind/            # Raw logs from wind runs
```
