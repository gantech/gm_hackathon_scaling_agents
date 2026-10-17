# Scaling Agents

In this module, you will use Academy to deploy a set of agents and experiment with different communication patterns. 

The examples are configured to run locally and use an in-memory exchange; however, this can be extended to use remote computing resources and a cloud messaging exchange for federated deployments. 

You will need to set environment variables to use Claude agents

```bash
export ANTHROPIC_AUTH_TOKEN=$AMSC_TOKEN
export ANTHROPIC_BASE_URL=https://mag.genesis.americansciencecloud.org
```

Each script runs Wordle agents. The agents use the Claude Agent SDK, and Academy runs them. Wordle is a small model of an expensive simulation: each guess is one evaluation, and the feedback tells the agent how near it is to the answer.

| Script | Pattern | Agents |
|---|---|---|
| `1_independent.py` | **Independent.** Each agent plays alone. | 2 `WordleAgent` |
| `2_message_board.py` | **Peers.** The agents share what they learn on a blackboard. | 2 `WordleAgent` |
| `3_multi_strategy.py` | **Coordinator and sub-agents.** The sub-agents test strategies on a cheap simulator. Then the coordinator plays the real game one time with the best strategy. | 1 `CoordinatorAgent` and 3 `StrategyAgent` |
| `4_gino_optimizer_board.py` | **Debate, then work, in rounds.** For the GINO model in `aifiv/2DLaminar`, the agents debate which optimizers are feasible and worth trying, and each claims one. Each agent implements its claim in its own copy of the code, trains it on its own GPU, and reports back on a message board. Then the next round starts. | 1 `OptimizerAgent` per GPU |

At the end, each script shows the answer from each agent and the true target word.

These scripts use `LocalExchangeFactory`, so all agents run in one process on your computer. To run agents on an HPC system, use `HttpExchangeFactory` and a Globus Compute executor, as in intermediate Module 3.

## GINO optimizer agents (`4_gino_optimizer_board.py`)

This is the message-board pattern of `2_message_board.py` applied to training the 2DLaminar GINO
surrogate in `aifiv/2DLaminar/gino`. The paper in `aifiv/` reports that SOAP, SS-Broyden, and
Adam followed by L-BFGS beat plain Adam on physics-informed problems. The agents decide
for themselves which of these, or other optimizers, are feasible and worth implementing for this
model. Then they test them.

Before round 1, the harness measures the original model on GPU 0 with `gino_profile.py`. It records
the parameter count and tensor shapes, the seconds per epoch with and without physics, the GPU
memory, and the size of a dense N x N matrix. The harness then starts the unmodified Adam baseline
(`agent0-run000`). Each round has these phases:

| Phase | Agents | What happens |
|---|---|---|
| propose | in parallel | Each agent proposes 1-3 optimizers with feasibility arithmetic: optimizer state against GPU memory, extra cost per step against the epoch time, full-batch or line-search needs, risk |
| critique | in parallel | Each agent checks its peers' arithmetic, points out infeasible items or better variants (for example a limited-memory version), and ranks the directions |
| claim | one at a time | Each agent claims one direction that no one else has claimed this round |
| work | in parallel | Each agent implements its claim in its own workspace, smoke-tests it, trains it on its GPU, posts insights, and ends with a round report |

The final message of each phase is posted to the board. Each agent's conversation continues across
phases and rounds, so in later rounds the agents build on the results and reports of earlier rounds.

| Tool | Phases | What it does |
|---|---|---|
| `read_board` / `write_board` | all | Leaderboard, this round's debate, round reports, insights |
| `Read`, `Glob`, `Grep` | all | Read the whole campaign (including peers' code) and the paper |
| `Write`, `Edit` | work | Edit the agent's own workspace only. The agents have no shell. |
| `smoke_test` | work | 3 epochs of the agent's `train.py`. It is free. |
| `run_script` | work | Runs a script from the agent's workspace on its GPU, for example a toy-problem test of a new optimizer |
| `launch_training` + `wait_for_job` | work | One full run with the shared fixed budget (`--fixed-args`, appended last). The agent gets `--runs-per-round` runs per round. |

The harness, not the agent, scores every full run. It runs a pristine copy of the original
`evaluate.py` on `checkpoint_best.pt` for the held-out case (`--eval-re`, default Re180) and ranks
the runs by the mean relative L2 error of u, v and p.

Outputs go in `runs/gino_optimizer_<timestamp>/`:
- `reference/model_facts.json`: the profile that the debate uses.
- `agent<i>/gino/`: the agent's edited code.
- `agent<i>/runs/<run_id>/`: the run's `train.log`, `eval.log`, `result.json`, checkpoints, and a
  `code/` snapshot of the exact code the run used.
- `board.jsonl`: every proposal, critique, claim, insight, result and report.
- `leaderboard_round<r>.json` and `leaderboard.json`: the ranked full runs.

To use a winning optimizer, copy it from its run's `code/` snapshot into `aifiv/2DLaminar/gino`.

### Run on one Polaris node (4 A100s, 4 agents)

`aifiv` is a git submodule. Clone with `git clone --recurse-submodules`, or run
`git submodule update --init` in an existing clone. The paper PDF is not tracked in `aifiv`, so
copy it into `aifiv/` or pass `--paper`.

The agents and the training use two separate Python environments:

| Env | Runs | Set up with |
|---|---|---|
| `.venv` (uv, Python 3.13) | the agents: Academy, the Claude Agent SDK, the message board | `uv sync` in this repo |
| `aifiv/aifiv_env` on ALCF conda base | `train.py`, `evaluate.py`, `gino_profile.py` | see below |

The training env is a venv on top of ALCF's conda base, so it keeps ALCF's optimized torch and
adds only PyG. The code uses PyG's `scatter` with sum and mean, which run on torch's native
`scatter_add_`. It also uses `radius`. So `torch_scatter`, `torch_sparse` and `pyg_lib` are not
needed. Run these from the repo root on a login node:

```bash
module use /soft/modulefiles; module load conda; conda activate base
```

```bash
python -m venv --system-site-packages aifiv/aifiv_env
```

```bash
source aifiv/aifiv_env/bin/activate && pip install torch_geometric
```

Keep the venv out of the aifiv repo:

```bash
cd aifiv && echo aifiv_env/ >> "$(git rev-parse --git-path info/exclude)" && cd ..
```

The agents never load conda. Each training run goes through
`scaling_agents/polaris_train_python.sh` (the default `TRAIN_PYTHON`), which loads the conda
module and activates `aifiv_env` for that subprocess only. Set `AIFIV_ENV` to use a venv
somewhere else. Before starting the agents, the job checks the training env: torch, PyG
`radius` on a GPU, netCDF4, and 4 visible GPUs. A broken env fails in seconds, not in round 1.

The case data `Re*/results/cylinder_2d_min.e` must be in `aifiv/2DLaminar` (or pass `--source`).
Put your AMSC token in `aurora_env_setup.sh`. The agents use `claude-opus-5-5`, and fall back to
`claude-sonnet-5-5` when Opus is overloaded. MAG serves both under the `google` provider header that
`aurora_env_setup.sh` already sets. Override them with `GINO_LLM_MODEL` and
`GINO_LLM_FALLBACK_MODEL`. These are separate from `LLM_MODEL`, which the Wordle examples use.
Submit from the repo root:

```bash
qsub -A <project> scaling_agents/polaris_gino_optimizer.pbs
```

The script uses environment variables for the main settings: `FIXED_ARGS` (default
`--epochs 2000 --warmup-epochs 200 --ramp-epochs 800`), `ROUNDS` (default 3), `RUNS_PER_ROUND`
(default 2), `RUN_TIMEOUT_MIN` (default 60) and `CAMPAIGN` (pass an existing campaign directory to
reuse its workspaces). The walltime needs to cover about ROUNDS x RUNS_PER_ROUND x RUN_TIMEOUT_MIN,
plus the debates. The defaults need about 7 hours, and the script requests 8. Each
run is pinned to its GPU with `CUDA_VISIBLE_DEVICES` and to the 8 cores next to that GPU
(`--cpu-bind polaris`). Check the core mapping with `nvidia-smi topo -m`.
