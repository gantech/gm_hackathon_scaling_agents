"""Peer agents that debate, implement and test optimizers for the 2DLaminar GINO model.

This is `2_message_board.py` with the Wordle game replaced by an expensive simulation:
training the GINO surrogate in `aifiv/2DLaminar/gino`. One agent per GPU (4 on a Polaris
node). Before the first round the harness profiles the model (parameter count and shapes,
seconds per epoch, GPU memory) and starts the unmodified Adam baseline. Then, every round:

  1. propose   each agent proposes optimizers worth trying (from the paper in aifiv/ or
               elsewhere) with feasibility arithmetic for THIS model: optimizer state vs
               GPU memory, extra cost per step vs the epoch time, implementation risk;
  2. critique  each agent checks its peers' proposals and arithmetic and ranks directions;
  3. claim     agents claim one direction each, in order, so no two take the same one;
  4. work      each agent implements its claim in its own copy of the code, smoke-tests it,
               trains it on its GPU, posts insights, and ends with a round report.

Every post goes on a shared message board, and every agent keeps its conversation across
phases and rounds. Each finished run is scored by a pristine copy of the original
`evaluate.py` on the held-out Reynolds number (relative L2 error of u, v, p), so all agents
are ranked on the same metric.

Example (one Polaris node, see polaris_gino_optimizer.pbs)::

    python 4_gino_optimizer_board.py --gpus 0 1 2 3 --rounds 3 --runs-per-round 2 \\
        --train-python /path/to/conda/envs/aifiv/bin/python \\
        --fixed-args "--epochs 2000 --warmup-epochs 200 --ramp-epochs 800"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from academy.agent import action
from academy.agent import Agent
from academy.context import ActionContext
from academy.exchange import LocalExchangeFactory
from academy.exchange.transport import AgentRegistrationT
from academy.handle import Handle
from academy.logging.recommended import recommended_logging
from academy.manager import Manager
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import create_sdk_mcp_server
from claude_agent_sdk import PermissionResultAllow
from claude_agent_sdk import PermissionResultDeny
from claude_agent_sdk import query
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import tool

from gino_runner import GinoSetup
from gino_runner import GpuRunner

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
FILE_TOOLS = ['Read', 'Write', 'Edit', 'Glob', 'Grep']
WRITE_TOOLS = {'Write', 'Edit'}
DEBATE_STEPS = ('propose', 'critique', 'claim')
# Own variables, so LLM_MODEL in aurora_env_setup.sh can stay on the Wordle examples' model.
# MAG serves Claude models under its "google" provider too, so the shared header still works.
MODEL = os.environ.get('GINO_LLM_MODEL', 'claude-opus-5-5')
FALLBACK_MODEL = os.environ.get('GINO_LLM_FALLBACK_MODEL', 'claude-sonnet-5-5')  # used when MODEL is overloaded
# Claude Code settings MAG recommends; no background calls (session titles) to other models.
CLI_ENV = {
    'DISABLE_NON_ESSENTIAL_MODEL_CALLS': '1',
    'DISABLE_TELEMETRY': '1',
    'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
    'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '16384',
    'MCP_TOOL_TIMEOUT': str(20 * 60 * 1000),
}

STEP_PROMPTS = {
    'propose': """ROUND {round} DEBATE, step 1 of 3: PROPOSE.
Read the board (leaderboard, previous round reports), and as needed the paper and the training code.
Propose 1-3 optimization strategies worth trying THIS round. Draw on the paper (Adam then L-BFGS, SOAP,
SS-Broyden) or anything else you know (Shampoo, Muon, natural-gradient / Gauss-Newton, limited-memory
quasi-Newton variants, LR schedules, phase switching, ...). For each one give:
  - what it is, and how it would plug into train.py;
  - feasibility for THIS model, with arithmetic from MODEL FACTS: optimizer state size against GPU
    memory, extra cost per step against the epoch time, whether it needs full-batch / line searches,
    implementation effort and risk. Rule options out explicitly when the numbers say so (for
    example a dense N x N curvature matrix);
  - expected benefit, and the concrete runs that would test it within {runs} full runs.
Your final message is posted to the board as your proposal.""",
    'critique': """ROUND {round} DEBATE, step 2 of 3: CRITIQUE.
Read every proposal on the board for this round. Check your peers' feasibility arithmetic and
claims: point out errors, infeasible or redundant items, and better variants (for instance a
limited-memory or block-diagonal version of something infeasible). Then rank the directions the
team should run this round (one per agent) and say why.
Your final message is posted to the board as your critique.""",
    'claim': """ROUND {round} DEBATE, step 3 of 3: CLAIM.
Read the critiques and the claims your peers have already posted this round (agents claim in
order agent0, agent1, ...). Claim ONE direction that nobody has claimed this round, preferring the
team's ranking. State exactly what you will implement, the files and flags you will add, and the
{runs} full runs you plan. Keep it under 150 words. Your final message is posted as your claim.""",
    'work': """ROUND {round} WORK.
Carry out your claim: implement it in your workspace, check it with `smoke_test` (and `run_script`
for standalone tests), then use your {runs} full runs (launch_training, then wait_for_job until it
finishes). Post insights with write_board as you learn things. If the claim turns out infeasible
or broken, post an insight saying why and switch to the next-best unclaimed direction.
Finish with your ROUND REPORT, which is posted automatically: what you implemented (files, flags),
each run_id with test_rel_l2 and wall time against the baseline, what you learned, and what the
team should try next round.""",
}


def text_result(payload: Any) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=1, default=str)
    return {'content': [{'type': 'text', 'text': text}]}


def extract_paper(pdf: Path, dest: Path) -> Path:
    """Plain text of the paper for models that cannot read PDFs; falls back to the PDF."""
    if pdf.suffix.lower() != '.pdf':
        return pdf
    try:
        import pymupdf
    except ImportError:
        return pdf
    if not dest.exists():
        with pymupdf.open(pdf) as doc:
            dest.write_text(''.join(page.get_text() for page in doc))
    return dest


class OptimizerAgent(Agent):
    """An LLM agent that debates and tests optimizers for the GINO model on one GPU.

    Args:
        peers: Handles of all agents (itself included; removed on startup).
        index: This agent's index; also picks its GPU from ``gpus``.
        gpus: GPU ids, one per agent.
        setup: Shared paths, training budget and scoring case.
        paper: Path to the optimizer paper (text or PDF).
        runs_per_round: Full training runs this agent may launch per round.
    """

    def __init__(
        self,
        peers: list[Handle[OptimizerAgent]],
        index: int,
        gpus: list[int],
        setup: GinoSetup,
        paper: Path,
        runs_per_round: int,
    ) -> None:
        self.model = MODEL
        self.peers = peers
        self.index = index
        self.name = f'agent{index}'
        self.gpu = gpus[index]
        self.num_agents = len(gpus)
        self.setup = setup
        self.paper = paper
        self.runs_per_round = runs_per_round
        self.workspace = setup.campaign / self.name
        self.runner = GpuRunner(setup, self.name, self.gpu, self.workspace)
        self.posts: list[dict[str, Any]] = []
        self.facts: dict[str, Any] = {}
        self.round = 0
        self.phase = 'idle'
        self.runs_left = 0
        self.session_id: str | None = None
        self._background: set[asyncio.Task[Any]] = set()

    async def agent_on_startup(self) -> None:
        self.peers = [p for p in self.peers if p.agent_id != self.agent_id]

    async def agent_on_shutdown(self) -> None:
        await self.runner.stop()

    # -- message board -----------------------------------------------------------------

    def _record_post(self, post: dict[str, Any]) -> None:
        self.posts.append(post)

    async def _publish(self, post: dict[str, Any]) -> None:
        post = {'time': time.strftime('%H:%M:%S'), 'round': self.round, 'author': self.name, **post}
        self._record_post(post)
        with (self.setup.campaign / 'board.jsonl').open('a') as f:
            f.write(json.dumps(post, default=str) + '\n')
        await asyncio.gather(
            *(peer.receive_post(post) for peer in self.peers),
            return_exceptions=True,
        )

    @action(context=True)
    async def receive_post(self, post: dict[str, Any], *, context: ActionContext) -> None:
        """Receive a post (proposal, critique, claim, insight, report or result) from a peer."""
        self._record_post(post)
        logger.info(f'{self.name} <- {post["author"]}: round {post.get("round")} {post.get("kind")}')

    async def _post_finished(self) -> None:
        """Post every finished training run that is not on the board yet."""
        for record in self.runner.records:
            if record.status != 'running' and not record.posted and record.kind != 'script':
                record.posted = True
                await self._publish({'kind': 'result', 'result': record.brief()})

    def read_board(self) -> str:
        def fmt(p: dict[str, Any], limit: int | None = None) -> str:
            text = p['text'] if limit is None or len(p['text']) <= limit else p['text'][:limit] + ' ...'
            return f'--- [round {p["round"]} {p["author"]} {p["kind"]} {p["time"]}]\n{text}'

        results = [p['result'] for p in self.posts if p['kind'] == 'result']
        scored = [r for r in results if r['kind'] == 'full' and r.get('test_rel_l2', {}).get('mean') is not None]
        scored.sort(key=lambda r: r['test_rel_l2']['mean'])
        others = [r for r in results if r not in scored]
        texts = [p for p in self.posts if 'text' in p]
        current = [p for p in texts if p['round'] == self.round and p['kind'] in DEBATE_STEPS]
        reports = [p for p in texts if p['kind'] == 'report' and p['round'] < self.round]
        insights = [p for p in texts if p['kind'] == 'insight'][-20:]

        lines = ['LEADERBOARD (full runs, lower test_rel_l2.mean is better):']
        lines += [json.dumps(r, default=str) for r in scored] or ['(no scored runs yet)']
        lines += ['', 'OTHER RUNS (smoke tests, failures, unscored; latest 10):']
        lines += [json.dumps(r, default=str) for r in others[-10:]] or ['(none)']
        lines += ['', f'ROUND {self.round} DEBATE SO FAR:'] + ([fmt(p) for p in current] or ['(nothing yet)'])
        if reports:
            last = max(p['round'] for p in reports)
            lines += ['', 'ROUND REPORTS (earlier rounds truncated):']
            lines += [fmt(p, None if p['round'] == last else 800) for p in reports]
        lines += ['', 'INSIGHTS (latest 20):'] + ([fmt(p, 1500) for p in insights] or ['(none yet)'])
        return '\n'.join(lines)

    # -- file sandbox ------------------------------------------------------------------

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        return path == root or root in path.parents

    async def can_use_tool(self, tool_name: str, tool_input: dict[str, Any], context: Any):
        """Reads anywhere in the campaign + the paper; writes only in this agent's workspace, only while working."""
        if tool_name not in FILE_TOOLS:
            return PermissionResultDeny(message=f'{tool_name} is not available; use the gino tools')
        pattern = str(tool_input.get('pattern', ''))
        if tool_name == 'Glob' and (pattern.startswith(('/', '~')) or '..' in pattern):
            return PermissionResultDeny(message='Glob patterns must be relative; set path instead')
        raw = tool_input.get('file_path') or tool_input.get('path') or str(self.workspace / 'gino')
        path = Path(raw)
        if not path.is_absolute():
            path = self.workspace / 'gino' / path
        path = path.resolve()
        if tool_name in WRITE_TOOLS:
            if self.phase != 'work':
                return PermissionResultDeny(message='No edits during the debate; implement in the work phase')
            root = self.workspace.resolve()
            ok = self._inside(path, root) and 'runs' not in path.relative_to(root).parts
            where = f'your workspace {self.workspace}'
        else:
            ok = self._inside(path, self.setup.campaign.resolve()) or path == self.paper.resolve()
            where = f'the campaign directory {self.setup.campaign} or the paper'
        if ok:
            return PermissionResultAllow()
        return PermissionResultDeny(message=f'{tool_name} on {path} denied: only {where} is allowed')

    # -- prompts and tools -------------------------------------------------------------

    def _system_prompt(self) -> str:
        setup = self.setup
        return f"""You are {self.name}, one of {self.num_agents} peer research agents on one GPU node, each with
its own GPU (yours is GPU {self.gpu}). Together you are improving the OPTIMIZER used to train a GINO
(graph-informed neural operator: GNO encoder, FNO on overset latent grids, GNO decoder) surrogate for
2D laminar flow past a cylinder over a Reynolds-number sweep, trained full-batch on CFD data plus a
physics-residual loss.

Current training: Adam with cosine LR decay, a physics-loss warm-up and ramp with gradient balancing,
and an optional full-batch L-BFGS phase (--lbfgs-blocks). The paper at {self.paper} reports that
quasi-second-order (SOAP) and second-order (SS-Broyden) optimizers reach far lower errors than Adam on
physics-informed objectives, on the paper's own networks (compare their sizes with MODEL FACTS).
Whether any of it is feasible or worth it here is for the team to work out. A well-explained negative result is also valuable.

You work in rounds: debate (propose, critique, claim), then work (implement, train, report).

MODEL FACTS (measured by the harness on the original code, short profiling run):
{json.dumps(self.facts, indent=1)}

Your workspace (edit freely during the work phase): {self.workspace / 'gino'} (train.py, model.py,
...); the data stays in {setup.source}. You may read, not edit, peers' workspaces under
{setup.campaign} to reuse code that works.
Rules:
- Change only the optimization: optimizer, its hyperparameters, LR schedule, phase switching
  (e.g. Adam then L-BFGS), and the physics/data loss balancing if you argue it is part of the
  optimizer. Do NOT change the model architecture, inputs, data or checkpoint format: the harness
  scores checkpoint_best.pt with the ORIGINAL evaluate.py on Re{setup.eval_re}.
- Every full run uses the same fixed budget, appended last to your args:
  {shlex.join(setup.fixed_args)}
  and is killed after {setup.run_timeout_min:g} minutes. Wall-clock time matters too.
- You get {self.runs_per_round} full runs per round. `smoke_test` (3 epochs) is free; use it after every
  code change, and `run_script` for standalone checks (e.g. an optimizer on a toy problem).
- Only one job runs on your GPU at a time: launch, then call `wait_for_job` until it finishes.
- The unmodified Adam baseline is run by the harness as agent0-run000 (on agent0's GPU); compare
  against it.
Results of every run are posted to the board automatically."""

    def _tools(self) -> list[Any]:
        runner = self.runner

        @tool('read_board', 'Read the shared message board: leaderboard, this round\'s debate, reports, insights.', {})
        async def read_board(args: dict[str, Any]) -> dict[str, Any]:
            return text_result(self.read_board())

        @tool('write_board', 'Post an insight (something you learned) to the shared board.', {'text': str})
        async def write_board(args: dict[str, Any]) -> dict[str, Any]:
            await self._publish({'kind': 'insight', 'text': args['text']})
            return text_result('posted')

        if self.phase != 'work':
            return [read_board, write_board]

        async def launch(kind: str, args: dict[str, Any], script: str | None = None) -> dict[str, Any]:
            await self._post_finished()
            if kind == 'full' and self.runs_left <= 0:
                return text_result('No full runs left this round. Write your round report.')
            try:
                record = runner.launch(kind, shlex.split(args.get('extra_args') or args.get('args') or ''),
                                       args.get('notes', ''), script=script)
            except (RuntimeError, ValueError) as exc:
                return text_result(f'{exc}. Call wait_for_job first.')
            if kind == 'full':
                self.runs_left -= 1
            return text_result({'run_id': record.run_id, 'run_dir': record.run_dir,
                                'full_runs_left_this_round': self.runs_left, 'next': 'call wait_for_job'})

        @tool(
            'launch_training',
            'Launch a FULL training run of your workspace train.py on your GPU with the fixed budget. '
            'extra_args: train.py flags as one shell string (e.g. "--lr 3e-3 --lbfgs-blocks 50", or flags '
            'you added). notes: what this run tests and why.',
            {'extra_args': str, 'notes': str},
        )
        async def launch_training(args: dict[str, Any]) -> dict[str, Any]:
            return await launch('full', args)

        @tool(
            'smoke_test',
            'Run your workspace train.py for 3 epochs (free; does not use a run) to check that your code '
            'runs. Same arguments as launch_training.',
            {'extra_args': str, 'notes': str},
        )
        async def smoke_test(args: dict[str, Any]) -> dict[str, Any]:
            return await launch('smoke', args)

        @tool(
            'run_script',
            'Run a Python script from your workspace on your GPU (training env, cwd = your gino dir, '
            '10 min limit), e.g. a unit test of a new optimizer. path: script path inside your workspace. '
            'args: optional command-line arguments as one shell string.',
            {'path': str, 'args': str},
        )
        async def run_script(args: dict[str, Any]) -> dict[str, Any]:
            path = Path(args['path'])
            if not path.is_absolute():
                path = self.workspace / 'gino' / path
            path = path.resolve()
            if not self._inside(path, self.workspace.resolve()) or not path.is_file():
                return text_result(f'{path} is not a file in your workspace {self.workspace}')
            return await launch('script', {'args': args.get('args', '')}, script=str(path))

        @tool(
            'wait_for_job',
            'Wait up to max_minutes (<= 15) for the job on your GPU. Returns the result when it finishes '
            '(scores, learning curve, error tail), else its progress.',
            {'max_minutes': float},
        )
        async def wait_for_job(args: dict[str, Any]) -> dict[str, Any]:
            if not runner.records:
                return text_result('No job has been launched.')
            minutes = min(max(float(args.get('max_minutes', 10)), 0.1), 15.0)
            record = await runner.wait(minutes)
            if record is None:
                return text_result(runner.progress() if runner.busy else 'No job is running.')
            if record.kind == 'script':
                return text_result({'run_id': record.run_id, 'status': record.status,
                                    'returncode': record.returncode, 'output_tail': record.error_tail})
            await self._post_finished()
            return text_result({**record.brief(), 'curve': record.train.get('curve'),
                                'error_tail': record.error_tail})

        return [read_board, write_board, launch_training, smoke_test, run_script, wait_for_job]

    async def _converse(self, prompt: str, max_turns: int) -> str:
        """One phase of this agent's conversation, resumed from the previous phase."""
        tools = self._tools()
        options = ClaudeAgentOptions(
            model=self.model,
            fallback_model=FALLBACK_MODEL if FALLBACK_MODEL and FALLBACK_MODEL != self.model else None,
            system_prompt=self._system_prompt(),
            cwd=str(self.workspace / 'gino'),
            mcp_servers={'gino': create_sdk_mcp_server('gino', tools=tools)},
            # File tools go through can_use_tool (sandboxed); no shell, no web.
            tools=FILE_TOOLS,
            strict_mcp_config=True,
            allowed_tools=[f'mcp__gino__{t.name}' for t in tools],
            can_use_tool=self.can_use_tool,
            setting_sources=[],
            env=CLI_ENV,
            max_turns=max_turns,
            resume=self.session_id,
        )
        result = ''
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result = message.result or f'(no final message: {message.subtype})'
                self.session_id = message.session_id or self.session_id
        return result

    # -- actions driven by main() ------------------------------------------------------

    @action
    async def profile(self) -> dict[str, Any]:
        """Measure the original model on this agent's GPU (run once, by agent0)."""
        return await self.runner.profile()

    @action
    async def set_facts(self, facts: dict[str, Any]) -> None:
        """Receive the model facts measured by the profiling agent."""
        self.facts = facts

    @action
    async def start_baseline(self) -> str:
        """Launch the unmodified Adam baseline on this GPU and post its result when it finishes."""
        record = self.runner.launch('full', [], 'harness: unmodified Adam baseline (original train.py defaults)')

        async def post_when_done() -> None:
            await self.runner.wait(self.setup.run_timeout_min + 30)
            await self._post_finished()

        task = asyncio.create_task(post_when_done())
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return record.run_id

    @action
    async def debate(self, round_: int, step: str) -> str:
        """One debate step; the agent's final message is posted as a proposal, critique or claim."""
        self.round, self.phase = round_, step
        text = await self._converse(
            STEP_PROMPTS[step].format(round=round_, runs=self.runs_per_round), max_turns=40,
        )
        await self._publish({'kind': step, 'text': text})
        self.phase = 'idle'
        return text

    @action
    async def work(self, round_: int) -> str:
        """Implement and train the claimed direction; the final message is posted as the round report."""
        self.round, self.phase = round_, 'work'
        self.runs_left = self.runs_per_round
        text = await self._converse(
            STEP_PROMPTS['work'].format(round=round_, runs=self.runs_per_round), max_turns=300,
        )
        if self.runner.busy and self.runner.records[-1].kind == 'full':
            # The conversation ended with a run still going: let it finish so the next round sees it.
            await self.runner.wait(self.setup.run_timeout_min + 30)
        await self._post_finished()
        await self._publish({'kind': 'report', 'text': text})
        self.phase = 'idle'
        return text


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3], help='One agent per GPU')
    parser.add_argument('--source', type=Path, default=REPO / 'aifiv' / '2DLaminar',
                        help='Directory with gino/, pino_physics_loss.py and the Re* case data')
    parser.add_argument('--paper', type=Path, default=REPO / 'aifiv' / '1-s2.0-S0045782526004299-main.pdf')
    parser.add_argument('--campaign', type=Path, default=None,
                        help='Output directory (default: runs/gino_optimizer_<timestamp>); reuse to keep workspaces')
    parser.add_argument('--train-python', default=sys.executable,
                        help='Python of the env with torch/torch_geometric (e.g. the aifiv conda env)')
    parser.add_argument('--fixed-args', default='--epochs 2000 --warmup-epochs 200 --ramp-epochs 800',
                        help='train.py budget every full run gets, appended last')
    parser.add_argument('--eval-re', type=int, default=180, help='Held-out case scored by evaluate.py')
    parser.add_argument('--rounds', type=int, default=3, help='Debate + work rounds')
    parser.add_argument('--runs-per-round', type=int, default=2, help='Full training runs per agent per round')
    parser.add_argument('--run-timeout-min', type=float, default=60.0, help='Wall-clock limit per full run')
    parser.add_argument('--no-baseline', action='store_true', help='Skip the harness Adam baseline run')
    parser.add_argument('--cpu-bind', choices=['polaris', 'none'], default='polaris',
                        help='Pin each run to the 8 cores next to its GPU (Polaris layout)')
    return parser.parse_args()


def write_leaderboard(campaign: Path, name: str) -> list[dict[str, Any]]:
    results = [json.loads(p.read_text()) for p in sorted(campaign.glob('agent*/runs/*/result.json'))]
    results = [r for r in results if r['kind'] == 'full']
    results.sort(key=lambda r: r['score'].get('mean', float('inf')))
    (campaign / name).write_text(json.dumps(results, indent=2))
    return results


async def run_phase(label: str, calls: list[Any]) -> list[Any]:
    results = await asyncio.gather(*calls, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.error(f'{label}: agent{i} failed: {result!r}')
    return results


async def main() -> None:
    cli = parse_cli()
    campaign = (cli.campaign or REPO / 'runs' / time.strftime('gino_optimizer_%Y%m%d_%H%M%S')).resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    setup = GinoSetup(
        source=cli.source.resolve(),
        campaign=campaign,
        train_python=cli.train_python,
        fixed_args=shlex.split(cli.fixed_args),
        eval_re=cli.eval_re,
        run_timeout_min=cli.run_timeout_min,
        cpu_bind=cli.cpu_bind,
    )
    setup.prepare(len(cli.gpus))
    paper = extract_paper(cli.paper.resolve(), campaign / 'reference' / 'paper.txt')
    logger.info(f'Campaign {campaign}: {len(cli.gpus)} agents on GPUs {cli.gpus}, {cli.rounds} rounds, '
                f'model {MODEL} (fallback {FALLBACK_MODEL or "none"})')

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        log_config=recommended_logging(),
    ) as manager:
        registrations: list[AgentRegistrationT] = [
            await manager.register_agent(OptimizerAgent) for _ in cli.gpus
        ]
        peers = [Handle(reg.agent_id) for reg in registrations]
        handles = [
            await manager.launch(
                OptimizerAgent,
                args=(peers, i, cli.gpus, setup, paper, cli.runs_per_round),
                registration=reg,
            )
            for i, reg in enumerate(registrations)
        ]

        logger.info('Profiling the original model on the first GPU.')
        facts = await handles[0].profile()
        logger.info(f'Model facts: {json.dumps(facts)}')
        await asyncio.gather(*(h.set_facts(facts) for h in handles))
        if not cli.no_baseline:
            logger.info(f'Started baseline {await handles[0].start_baseline()}')

        for round_ in range(1, cli.rounds + 1):
            for step in DEBATE_STEPS:
                logger.info(f'Round {round_}: {step}')
                if step == 'claim':  # in order, so each agent sees the claims before it
                    for h in handles:
                        await run_phase(f'round {round_} claim', [h.debate(round_, step)])
                else:
                    await run_phase(f'round {round_} {step}', [h.debate(round_, step) for h in handles])
            logger.info(f'Round {round_}: work')
            await run_phase(f'round {round_} work', [h.work(round_) for h in handles])
            for r in write_leaderboard(campaign, f'leaderboard_round{round_}.json')[:5]:
                logger.info(f'after round {round_}: {r["run_id"]} test rel L2 {r["score"]} | '
                            f'{r["wall_minutes"]:.1f} min | {shlex.join(r["extra_args"])}')

    results = write_leaderboard(campaign, 'leaderboard.json')
    logger.info(f'Board: {campaign / "board.jsonl"} | leaderboard: {campaign / "leaderboard.json"}')
    for r in results[:5]:
        logger.info(f'{r["run_id"]}: test rel L2 {r["score"]} | {r["wall_minutes"]:.1f} min | '
                    f'{shlex.join(r["extra_args"])} | code {r["run_dir"]}/code')


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
