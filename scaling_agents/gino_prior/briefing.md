# Briefing: what earlier campaigns found (2026-10-17)

All numbers are on this exact setup. The model is the overset GINO with 8,452,582 parameters,
trained full-batch on 9 Reynolds-number cases, with Re180 held out. Every full run has the same
fixed budget: 2000 epochs, physics warm-up 200 epochs, physics ramp 800 epochs. `test_rel_l2` is
the harness score: the mean relative L2 error on Re180 over u, v and p, computed with the
original `evaluate.py`. `val_data` is train.py's normalized MSE on Re180.

## Sources (read these for details)

- `board_20261017_004602.jsonl`: the 1-hour, 4-agent campaign (Claude Opus). It covers round 1 in
  full: proposals, critiques, claims, insights, run results, and one report (agent3).
- `board_20261017_002510.jsonl`: an aborted run that only got as far as three round-1 proposals
  and the baseline result.
- `kestrel_adam_10000_epochs.log`: a separate 10,000-epoch Adam run by the user (details below).
- If the Polaris campaign directory `runs/gino_optimizer_20261017_004602/` is in the prior list,
  it contains the earlier agents' code (`agent*/gino/`) and each run's exact code
  (`agent*/runs/<run_id>/code/`). Reuse it instead of rewriting:
  - agent2: `soap.py` and `--optimizer soap`, `--soap-betas`, `--soap-precond-freq`,
    `--soap-shampoo-beta`. The size-2 real/imaginary axis is merged, so `[32,32,16,16,2]` is
    preconditioned as `[32,32,16,32]`.
  - agent3: `--lr-warmup-epochs N` (linear warm-up, then cosine).
  - agent0: `--lbfgs-time-budget MIN`, `--lbfgs-keep-history`, per-block s/eval logging, and a
    limited-memory self-scaled BFGS (`ssbroyden.py`) that was never finished.
  - agent1: `--max-wall-minutes`.

## Results so far (lower test_rel_l2 is better; one seed each)

| run | what | test_rel_l2 (u / v / p) | best val_data | wall |
|---|---|---|---|---|
| agent2-run003 | SOAP, lr 3e-3, betas 0.95/0.95, precondition frequency 10, cosine to 1e-5, no tail | **0.00127** (0.0011/0.0014/0.0013) | 1.59e-6 | 4.7 min |
| agent0-run003 | Adam lr 1e-3, then fp64 L-BFGS tail (250 iterations per block, history 100, value balance, 54-min time budget) | 0.00260 | 6.94e-6 | 49.5 min |
| agent3-run002 | Adam lr 6e-3, 100-epoch LR warm-up | 0.00360 (0.0030/0.0044/0.0034) | 1.35e-5 | 5.1 min |
| agent3-run001 | Adam lr 3e-3, 50-epoch LR warm-up | 0.00467 (0.0039/0.0060/0.0041) | 2.29e-5 | 4.4 min |
| agent1-run001 | Adam lr 1e-3, then fp32 L-BFGS tail (grad balance) | 0.00540 | 3.04e-5 | 8.2 min |
| agent0-run000 | baseline: unmodified Adam lr 1e-3 | 0.00603 (0.0054/0.0079/0.0048) | 3.89e-5 | 4.7 min |
| agent1-run003 | SOAP lr 6e-3, then fp64 L-BFGS tail | not finished | SOAP phase 1.21e-6; the tail improved it by only 0.4% in 30 min | — |

## What was learned

1. **SOAP is the best so far, and its gain is not a learning-rate effect.** It beat Adam at the
   same LR (3e-3) by 3.7×, and the best Adam (6e-3) by 2.8×. Its optimizer step is about 21 ms
   against 0.7 ms for Adam, but per-epoch time is dominated by the forward/backward pass, so the
   2000-epoch run time was the same as Adam's. SOAP at 6e-3 reached a lower val_data than at 3e-3
   (1.21e-6 against 1.59e-6), but that run has no test score yet.
2. **The baseline Adam LR of 1e-3 is too small.** 3e-3, then 6e-3, improved every channel each
   time, with no divergence. There was only a transient val bump during the physics ramp (epochs
   ~480–640). The optimum for Adam may be above 6e-3.
3. **Every run was still improving at epoch 2000**, and the best checkpoint was the last one.
   The budget is epoch-limited, not converged.
4. **L-BFGS tails help Adam but not SOAP.**
   - After Adam, the fp64 tail gave 2.3×, but the whole gain came in about the first 17 minutes
     (blocks 0–26). After that it was flat.
   - After SOAP it gave essentially nothing.
   - In fp32, the tail stalled after about 3 minutes.
   - A fp64 function+gradient evaluation costs 0.10–0.19 s. The line search used about 1.05
     evaluations per iteration, so it almost always accepted the first step.
5. **Why the L-BFGS tails stall (verified in the torch source).** `torch.optim.LBFGS` updates
   its curvature history only if `ys > 1e-10`, an absolute threshold (line ~400 of
   `torch/optim/lbfgs.py`). With the loss around 1e-5 and small steps, `ys` falls below it. The
   history then freezes, the method degrades to near steepest descent, and blocks burn max_eval
   evaluations. Fixes proposed but not yet tested: rescale the tail objective by 1/loss0, or use a
   relative curvature test (ys > eps·|y||s|) in a custom implementation.
6. **The fixed budget is only partly used.** Adam and SOAP take about 5 minutes per 2000-epoch run.
7. **Feasibility arithmetic the team agreed on.** Any dense N×N curvature matrix needs about 286 TB
   in fp32 (N² ≈ 7.1e13), so dense BFGS, SS-Broyden and Gauss-Newton are out. SS-Broyden is only
   possible in limited-memory form.

## Kestrel run (user, `kestrel_adam_10000_epochs.log`)

- **Setup:** Adam at a constant lr 1e-3 for 10,000 epochs. Physics is off until epoch 1010, then
  ramps to its full weight of 0.1 by epoch 2000. This is a different schedule from the shared budget.
- **Best result:** val_data 1.87e-6 at epoch 9520.
- **Noise:** the late phase is very noisy. After epoch 8000 the median val_data is 1.0e-5, with
  spikes up to 3.7e-4, and 15% of logged points are more than 5× the median.
- **Comparison:** SOAP's 2000-epoch run reached val_data 1.59e-6, lower than this 10,000-epoch
  constant-LR Adam run's best checkpoint.

## The DSGNAR paper (`2607.02194v1`: Doubly-Sketched Gauss–Newton with Adaptive Ratio)

**What it is.** A Levenberg–Marquardt / Gauss–Newton method for least-squares PINN objectives.
- **Sketching.** The residual Jacobian J (N × d) is sketched on both sides: CountSketch on the
  residual rows, and a subsampled randomised cosine transform on the parameter columns. The result
  is a small square s×s matrix J̃.
- **Step selection.** One SVD of J̃ gives LM steps for any regularisation λ. Each step is chosen by
  probing for a target decrease ratio ϱ⋆, and ϱ⋆ adapts in two stages: conservative first (to lower
  the effective regularisation), then aggressive.
- **Weights.** There is also a residual-based condition-weight update.
- **Reference code:** https://www.github.com/wephy/physics-informed-neural-networks (JAX).

**What the paper reports.** Relative L2 errors as low as 3e-16 in fp64, and very fast fp32 solves,
on PINNs (SIREN/MLP/SPINN). Its networks have d ≈ 9k–17k parameters, with sketch sizes s =
1,000–5,000. Its Jacobian-building timings go up to 1,048,576 × 252,503 with s = 10,000. It
recommends s between d/3 and d/2.

**Why it can't be used here as published.** The model has d = 8.45M parameters. At s = d/3, the
s×s matrix alone would need about 31 TB. Building J̃ also costs roughly one reverse-mode pass per
sketched row. On this model, a full-batch backward pass takes about the epoch time of 0.15–0.3 s,
so s = 20,000 would take about 1–2 hours per iteration unless it is batched or vectorised.

**Open questions for the team.**
- Can it be adapted:
  - with a much smaller s;
  - per layer or per block;
  - restricted to a subset of parameters;
  - as a short tail after SOAP;
  - with batched VJPs/JVPs (`torch.func`)?
- Does our objective have the least-squares residual form it needs? The data term is an MSE over
  about 124k residuals (9 cases × 4583 nodes × 3 fields). The physics terms are means of squared
  dimensionless residuals (`RANSPhysicsLossQuad2D.physics_loss` in `pino_physics_loss.py`), so a
  residual vector exists, but it has to be exposed.
- Is it worth it at all, given that SOAP already reaches 0.00127?

## Suggestions from the review of the 1-hour run (not yet tested)

1. **Measure noise.** Repeat the top configurations (SOAP lr 3e-3, Adam lr 6e-3) with 2–3 seeds
   (`--seed`). Every result so far is one seed, so small differences may not be real.
2. **Tune SOAP.** Score SOAP at 6e-3 without a tail, and push higher: LR, betas, precondition
   frequency, warm-up, `--min-lr`.
3. **Find the Adam peak.** Push Adam to 1e-2 and above. Tuned Adam, not the baseline, is the bar
   that SOAP and anything newer must beat.
4. **Fix L-BFGS before using it as a tail.** Fix the stall (point 5 above) first. Cap tails at
   about 15–20 min, since the gain came early.
5. **SS-Broyden in limited-memory form only.** agent0 started `ssbroyden.py`.
6. **Settle DSGNAR.** Decide by arithmetic and a cheap prototype (e.g. on a parameter subset or a
   toy problem with `run_script`) whether an adapted DSGNAR is worth a full run.
7. **Compare at equal compute too.** The 2000-epoch budget gives equal epochs, but the L-BFGS
   tails used about 10× more wall time. Report wall time with every claim.
8. **Remember the selection bias.** Re180 both selects `checkpoint_best.pt` and scores it, so all
   scores are slightly optimistic, equally for every run.
