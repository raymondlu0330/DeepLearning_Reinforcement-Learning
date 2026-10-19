# LAB5: Value-Based Reinforcement Learning

Student ID: M11415097
Name: 呂梓叡
Course: 535518 Deep Learning, Spring 2026
Due: 2026/05/05 23:59

---

## Final Results

| Task | Model | 20-seed Avg | Score |
|---|---|---|---|
| Task 1 — CartPole-v1 (Vanilla DQN) | `LAB5_M11415097_task1.pt` | **500.00** | 15 / 15 |
| Task 2 — Pong (Vanilla DQN + fine-tune) | `LAB5_M11415097_task2.pt` | **18.15** | ~19.6 / 20 |
| Task 3 — Pong (DDQN + PER + 3-step + Dueling) | `LAB5_M11415097_task3_best.pt` | **14.80** | 11 / 20 |

Task 3 first reached score ≥19 at **2,249,000 env steps** (2M–2.5M tier).

---

## Project Layout

```
LAB5_M11415097_呂梓叡/
├── LAB5_M11415097_Code/          # Source code (included in zip)
│   ├── dqn.py                    # Unified training script (all 3 tasks)
│   ├── test_model.py             # Pong evaluation (TA's script, +avg patch)
│   ├── test_model_task1.py       # CartPole evaluation
│   ├── test_model_task3.py       # Pong evaluation with --use-dueling flag
│   ├── requirements.txt
│   ├── run_task1.sh
│   ├── run_task2.sh
│   └── run_task3.sh
├── results/                      # Training checkpoints (NOT in zip)
│   ├── task1_v4/best_model.pt    # → LAB5_M11415097_task1.pt
│   ├── task2_finetune/best_model.pt  # → LAB5_M11415097_task2.pt
│   └── task3_v3/model_ep2400.pt  # → LAB5_M11415097_task3_best.pt
├── report/                       # LaTeX source + figures (NOT in zip)
├── screenshots/                  # Eval videos (NOT in zip)
├── wandb/                        # W&B local logs (NOT in zip)
├── LAB5_M11415097_task1.pt
├── LAB5_M11415097_task2.pt
├── LAB5_M11415097_task3_600000.pt
├── LAB5_M11415097_task3_1000000.pt
├── LAB5_M11415097_task3_1500000.pt
├── LAB5_M11415097_task3_2000000.pt
├── LAB5_M11415097_task3_2500000.pt
├── LAB5_M11415097_task3_best.pt  # = model_ep2400.pt (20-seed avg 14.80)
├── LAB5_M11415097.pdf            # Final report
├── LAB5_M11415097.mp4            # Demo video (5–6 min)
├── HANDOFF.md
└── README.md
```

---

## Environment Setup

```bash
conda create -n lab5 python=3.10 -y
conda activate lab5
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r LAB5_M11415097_Code/requirements.txt
wandb login
```

---

## Evaluation Commands

```powershell
# Task 1 — CartPole (expected avg: 500.00)
python LAB5_M11415097_Code\test_model_task1.py `
    --model-path .\LAB5_M11415097_task1.pt --episodes 20 --seed 0

# Task 2 — Pong vanilla + fine-tune (expected avg: ~18.15)
python LAB5_M11415097_Code\test_model.py `
    --model-path .\LAB5_M11415097_task2.pt --episodes 20 --seed 0

# Task 3 — Pong enhanced (expected avg: ~14.80)
python LAB5_M11415097_Code\test_model_task3.py `
    --model-path .\LAB5_M11415097_task3_best.pt --episodes 20 --seed 0 --use-dueling
```

---

## Training Commands

### Task 1: Vanilla DQN on CartPole-v1

```bash
bash LAB5_M11415097_Code/run_task1.sh
# or manually:
python LAB5_M11415097_Code/dqn.py \
    --env-name CartPole-v1 \
    --lr 5e-4 --memory-size 50000 --target-update-freq 1000 \
    --epsilon-decay 0.9999 --epsilon-min 0.05 \
    --wandb-project DLP-Lab5-Task1-CartPole \
    --wandb-run-name task1-vanilla-v4 \
    --save-dir ./results/task1_v4
```

### Task 2: Vanilla DQN on Pong (two-stage)

```bash
bash LAB5_M11415097_Code/run_task2.sh
# Stage 1: main training (~7M steps)
# Stage 2: fine-tune from best_ep0980 with lr=1e-5, epsilon=0.05
```

### Task 3: Enhanced DQN on Pong (DDQN + PER + 3-step + Dueling)

```bash
bash LAB5_M11415097_Code/run_task3.sh
# or manually:
python LAB5_M11415097_Code/dqn.py \
    --env-name "ALE/Pong-v5" \
    --use-double-dqn --use-per --n-step 3 --use-dueling --task3-mode \
    --wandb-project DLP-Lab5-Task3-Enhanced \
    --wandb-run-name task3-rainbow-lite-v3 \
    --save-dir ./results/task3_v3
```

---

## Key Implementation Notes

- **Single `dqn.py`** handles all three tasks; CLI flags select enhancements.
- **Truncation bug fix**: stores `terminated` (not `done`) in replay buffer — decisive for CartPole (avg jumped from ~150 → 500).
- **Fine-tuning (Task 2)**: `--resume-from best_ep0980_r20.pt`, lr=1e-5, fixed ε=0.05 → avg 17.25 → 18.15.
- **Task 3 best model**: `model_ep2400.pt` (20-seed avg 14.80), first score ≥19 at **2.249M env steps**.

---

## Submission Checklist

- [x] `LAB5_M11415097.pdf` (report, compiled from Overleaf)
- [ ] `LAB5_M11415097.mp4` (5–6 min demo video, English)
- [x] `LAB5_M11415097_task1.pt` (avg 500.00)
- [x] `LAB5_M11415097_task2.pt` (avg 18.15, fine-tuned)
- [x] `LAB5_M11415097_task3_600000.pt`
- [x] `LAB5_M11415097_task3_1000000.pt`
- [x] `LAB5_M11415097_task3_1500000.pt`
- [x] `LAB5_M11415097_task3_2000000.pt`
- [x] `LAB5_M11415097_task3_2500000.pt`
- [x] `LAB5_M11415097_task3_best.pt` (model_ep2400, avg 14.80)
- [x] `requirements.txt` in Code folder
- [x] Eval screenshots in report (task1, task2, task3)
- [x] Score 19 timestep in report (2,249,000 env steps)

---

## Hyperparameters Summary

### Task 1 (final: v4)
| Hyperparameter | Value |
|---|---|
| Network | MLP 4→128→128→2 |
| lr | 5e-4 |
| Memory size | 50,000 |
| Batch size | 64 |
| Target update freq | 1,000 steps |
| ε decay | 0.9999 / step |
| ε min | 0.05 |
| Gamma | 0.99 |

### Task 2 (fine-tune stage)
| Hyperparameter | Value |
|---|---|
| Network | CNN (DeepMind Nature) |
| Resume from | best_ep0980_r20.pt |
| lr | 1e-5 |
| Epsilon | 0.05 (fixed) |
| Extra episodes | ~1,500 |

### Task 3 (rainbow-lite-v3)
| Hyperparameter | Value |
|---|---|
| Enhancements | DDQN + PER + 3-step + Dueling |
| lr | 1e-4 |
| PER α | 0.6 |
| PER β | 0.4 → 1.0 |
| n-step | 3 |
| Target update freq | 1,000 steps |
