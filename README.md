# Scaling Multi-Agent Systems

## Before you start

1. Install this repo.
2. Setup uv and dependencies
   ```bash
   $ RUN curl -LsSf https://astral.sh/uv/install.sh | sh
   $ source ~/.venv/bin/activate
   $ uv sync
   ```

4. Start an interactive compute node
   ```bash
   $ qsub -I -q R8914313 -l select=1,walltime=01:00:00 -A GenesisHackathonOct26 -l filesystems=home:flare
   ```

5. Configure access to a model. E.g., via the Model Access Gateway (MAG), local inference service, or commercial provider. For MAG, refer to [MAG get started](https://amsc-docs-d762d2.gitlab.io/mag-get-started). 

5. Update `aurora_env_setup.sh` with your AMSC token
6. Activate uv environment on the compute node
   ```bash
   $ source ~/.venv/bin/activate
   $ source aurora_env_setup.sh
   ```

### Scaling Agents

Each script runs Wordle agents. The agents use the Claude Agent SDK, and Academy runs the agents and enables communication between agents. Wordle is a small model of an expensive simulation: each guess is one evaluation, and the feedback tells the agent how near it is to the answer.

| Script | Pattern | Agents |
|---|---|---|
| `1_independent.py` | **Independent.** Each agent plays alone. | 2 `WordleAgent` |
| `2_message_board.py` | **Peers.** The agents share what they learn on a blackboard. | 2 `WordleAgent` |
| `3_multi_strategy.py` | **Coordinator and sub-agents.** The sub-agents test strategies on a cheap simulator. Then the coordinator plays the real game one time with the best strategy. | 1 `CoordinatorAgent` and 3 `StrategyAgent` |
| `4_gino_optimizer_board.py` | **Debate, then work, in rounds.** For the GINO model in `aifiv/2DLaminar`, the agents debate which optimizers are feasible and worth trying, and each claims one. Each agent implements and trains its claim on its own node, up to 4 runs at once, then reports back on a message board. See `scaling_agents/README.md` and `polaris_gino_optimizer.pbs`. | 1 `OptimizerAgent` per node (5 on Polaris `prod`) |
