"""Workspaces, GPU-pinned training runs and independent scoring for the GINO optimizer agents.

Each agent gets its own copy of the 2DLaminar GINO code (``<campaign>/agent<i>/gino``) that
it may edit freely, e.g. to add a new optimizer to ``train.py``. Training runs launch that
copy's ``train.py`` on the agent's GPU. Every finished run is then scored by a pristine copy
of the original ``evaluate.py`` (``<campaign>/reference``) on the held-out case, so an agent
cannot change the metric it is judged by.

Run layout::

    <campaign>/
      reference/            pristine gino/ + pino_physics_loss.py (scoring only)
      agent0/gino/          agent 0's editable code
      agent0/runs/run003/   train.log, eval.log, checkpoint_best.pt, code/ (snapshot), result.json
      board.jsonl           every message-board post, in order
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shlex
import shutil
import signal
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

# One "epoch ..." log line of train.py (Adam and L-BFGS phases share the columns).
EPOCH_RE = re.compile(
    r'^epoch\s+(?P<epoch>\d+) \|.*?\| data (?P<data>\S+) \|.*?'
    r'\| physics (?P<physics>\S+) \| val_data (?P<val>\S+)',
)
# evaluate.py: "<field> blended RMSE x | L2 error / L2 of (ref - mean) y" (label empty
# for single-grid models; per-branch lines are skipped).
EVAL_RE = re.compile(
    r'^(?P<name>velocity_x|velocity_y|pressure)\s+(?:blended\s+)?RMSE (?P<rmse>\S+) \| '
    r'L2 error / L2 of \(ref - mean\) (?P<rel>\S+)\s*$',
)
SMOKE_ARGS = [
    '--epochs', '3', '--warmup-epochs', '1', '--ramp-epochs', '1',
    '--log-every', '1', '--checkpoint-every', '1000',
]
# Two data-only and four physics epochs: step times for both kinds of epoch.
PROFILE_ARGS = [
    '--epochs', '6', '--warmup-epochs', '2', '--ramp-epochs', '2',
    '--log-every', '1', '--checkpoint-every', '1000',
]


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def copy_gino_tree(source: Path, dest: Path) -> None:
    """Copy ``<source>/gino`` and ``pino_physics_loss.py`` (train.py imports it from gino/..)."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source / 'gino',
        dest / 'gino',
        ignore=shutil.ignore_patterns('runs', '__pycache__', 'log.*', '*.pt'),
        dirs_exist_ok=True,
    )
    shutil.copy2(source / 'pino_physics_loss.py', dest / 'pino_physics_loss.py')


def snapshot_code(workspace: Path, dest: Path) -> None:
    """Keep the exact code a run used, so the best run can be reproduced or merged back."""
    dest.mkdir(parents=True, exist_ok=True)
    for path in workspace.rglob('*.py'):
        if 'runs' in path.relative_to(workspace).parts:
            continue
        target = dest / path.relative_to(workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def parse_train_log(text: str) -> dict[str, Any]:
    rows = []
    for line in text.splitlines():
        m = EPOCH_RE.match(line)
        if m:
            rows.append({
                'epoch': int(m['epoch']),
                'data': _float(m['data']),
                'physics': _float(m['physics']),
                'val_data': _float(m['val']),
            })
    summary: dict[str, Any] = {'logged_epochs': len(rows)}
    if not rows:
        return summary
    finite = [r for r in rows if math.isfinite(r['val_data'])]
    best = min(finite, key=lambda r: r['val_data']) if finite else None
    # A short, evenly spaced learning curve the LLM can reason about.
    stride = max(1, len(rows) // 12)
    curve = rows[::stride]
    if curve[-1] is not rows[-1]:
        curve.append(rows[-1])
    summary.update({
        'last': rows[-1],
        'best_val_data': best['val_data'] if best else math.nan,
        'best_epoch': best['epoch'] if best else None,
        'diverged': any(not math.isfinite(r['data']) for r in rows),
        'curve': curve,
    })
    return summary


def parse_eval_log(text: str) -> dict[str, float]:
    rel = {}
    for line in text.splitlines():
        m = EVAL_RE.match(line.strip())
        if m and m['name'] not in rel:
            rel[m['name']] = _float(m['rel'])
    if len(rel) == 3:
        rel['mean'] = sum(rel.values()) / 3.0
    return rel


@dataclass
class GinoSetup:
    """Everything an agent's runs share; built once by the launcher."""

    source: Path            # aifiv/2DLaminar (holds gino/, pino_physics_loss.py, Re*/)
    campaign: Path
    train_python: str
    fixed_args: list[str]   # appended last so they win over agent args (the shared budget)
    eval_re: int
    run_timeout_min: float
    smoke_timeout_min: float = 10.0
    cpu_bind: str = 'polaris'
    deadline: float | None = None   # time.time() by which every job must have ended (job walltime)

    @property
    def reference(self) -> Path:
        return self.campaign / 'reference'

    def minutes_left(self) -> float:
        return math.inf if self.deadline is None else (self.deadline - time.time()) / 60.0

    def cap(self, timeout_min: float) -> float:
        """A job timeout that also ends before the deadline (at least 1 minute)."""
        return max(1.0, min(timeout_min, self.minutes_left()))

    def prepare(self, num_agents: int) -> list[Path]:
        if not (self.reference / 'gino').exists():
            copy_gino_tree(self.source, self.reference)
        workspaces = []
        for i in range(num_agents):
            root = self.campaign / f'agent{i}'
            if not (root / 'gino').exists():
                copy_gino_tree(self.source, root)
            (root / 'runs').mkdir(parents=True, exist_ok=True)
            workspaces.append(root)
        return workspaces


@dataclass
class RunRecord:
    run_id: str
    agent: str
    kind: str               # "full" | "smoke" | "script"
    gpu: int
    extra_args: list[str]
    notes: str
    run_dir: str
    status: str = 'running'  # running | finished | failed | timeout
    started: float = field(default_factory=time.time)
    wall_minutes: float = 0.0
    returncode: int | None = None
    train: dict[str, Any] = field(default_factory=dict)
    score: dict[str, float] = field(default_factory=dict)
    error_tail: str = ''
    posted: bool = False     # result already on the message board

    def brief(self) -> dict[str, Any]:
        """What gets posted on the board (no long curves)."""
        out = {
            'run_id': self.run_id, 'agent': self.agent, 'kind': self.kind,
            'status': self.status, 'extra_args': self.extra_args, 'notes': self.notes,
            'wall_minutes': round(self.wall_minutes, 1),
            'test_rel_l2': self.score,
            'best_val_data': self.train.get('best_val_data'),
            'best_epoch': self.train.get('best_epoch'),
            'diverged': self.train.get('diverged'),
            'code_snapshot': str(Path(self.run_dir) / 'code'),
        }
        return out


class GpuRunner:
    """Runs one subprocess at a time on one GPU for one agent."""

    def __init__(self, setup: GinoSetup, agent: str, gpu: int, workspace: Path) -> None:
        self.setup = setup
        self.agent = agent
        self.gpu = gpu
        self.workspace = workspace
        self.records: list[RunRecord] = []
        self._task: asyncio.Task[RunRecord] | None = None
        self._proc: asyncio.subprocess.Process | None = None

    # -- process helpers ---------------------------------------------------------------

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        env.setdefault('OMP_NUM_THREADS', '8')
        env['PYTHONUNBUFFERED'] = '1'
        return env

    def _prefix(self) -> list[str]:
        # Polaris: GPU 0 sits next to cores 24-31, GPU 3 next to 0-7 (check `nvidia-smi topo -m`).
        if self.setup.cpu_bind == 'polaris' and shutil.which('taskset'):
            first = (3 - self.gpu % 4) * 8
            return ['taskset', '-c', f'{first}-{first + 7}']
        return []

    async def _run(self, cmd: list[str], log_path: Path, cwd: Path, timeout_min: float) -> tuple[int | None, bool]:
        with log_path.open('w') as log:
            log.write('$ ' + shlex.join(cmd) + '\n')
            log.flush()
            self._proc = await asyncio.create_subprocess_exec(
                *self._prefix(), *cmd, cwd=cwd, env=self._env(),
                stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
            )
            try:
                return await asyncio.wait_for(self._proc.wait(), timeout_min * 60.0), False
            except TimeoutError:
                os.killpg(self._proc.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._proc.wait(), 30)
                except TimeoutError:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                    await self._proc.wait()
                return self._proc.returncode, True
            finally:
                self._proc = None

    @staticmethod
    def _tail(path: Path, lines: int = 40) -> str:
        if not path.exists():
            return ''
        return '\n'.join(path.read_text(errors='replace').splitlines()[-lines:])

    # -- public API --------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def launch(self, kind: str, extra_args: list[str], notes: str, script: str | None = None) -> RunRecord:
        if self.busy:
            raise RuntimeError(f'GPU {self.gpu} is busy with {self.records[-1].run_id}; wait for it first')
        run_id = f'{self.agent}-run{len(self.records):03d}'
        run_dir = self.workspace / 'runs' / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record = RunRecord(run_id, self.agent, kind, self.gpu, list(extra_args), notes, str(run_dir))
        self.records.append(record)
        self._task = asyncio.create_task(self._execute(record, script))
        return record

    async def _execute(self, record: RunRecord, script: str | None) -> RunRecord:
        run_dir = Path(record.run_dir)
        gino = self.workspace / 'gino'
        setup = self.setup
        try:
            snapshot_code(self.workspace, run_dir / 'code')
            if record.kind == 'script':
                cmd = [setup.train_python, script, *record.extra_args]
                rc, timed_out = await self._run(cmd, run_dir / 'script.log', gino, setup.cap(setup.smoke_timeout_min))
                record.returncode = rc
                record.status = 'timeout' if timed_out else ('finished' if rc == 0 else 'failed')
                record.error_tail = self._tail(run_dir / 'script.log', 60)
                return record

            budget = SMOKE_ARGS if record.kind == 'smoke' else setup.fixed_args
            if record.kind == 'smoke' and '--lbfgs-blocks' in record.extra_args:
                budget = [*budget, '--lbfgs-blocks', '1', '--lbfgs-iters', '2']
            cmd = [
                setup.train_python, 'train.py', *record.extra_args,
                '--data-root', str(setup.source), '--device', 'cuda',
                '--output-dir', str(run_dir), *budget,
            ]
            timeout = setup.cap(setup.smoke_timeout_min if record.kind == 'smoke' else setup.run_timeout_min)
            rc, timed_out = await self._run(cmd, run_dir / 'train.log', gino, timeout)
            record.returncode = rc
            log_text = (run_dir / 'train.log').read_text(errors='replace')
            record.train = parse_train_log(log_text)
            record.status = 'timeout' if timed_out else ('finished' if rc == 0 else 'failed')
            if record.status != 'finished':
                record.error_tail = self._tail(run_dir / 'train.log')

            checkpoint = run_dir / 'checkpoint_best.pt'
            if checkpoint.exists():
                record.score = await self._score(checkpoint, run_dir)
        except Exception as exc:  # noqa: BLE001 - report harness errors to the agent instead of dying
            record.status = 'failed'
            record.error_tail = f'harness error: {exc!r}\n{record.error_tail}'
        finally:
            record.wall_minutes = (time.time() - record.started) / 60.0
            (run_dir / 'result.json').write_text(json.dumps(asdict(record), indent=2, default=str))
        return record

    async def _score(self, checkpoint: Path, run_dir: Path) -> dict[str, float]:
        setup = self.setup
        cmd = [
            setup.train_python, 'evaluate.py', '--checkpoint', str(checkpoint),
            '--case-dir', str(setup.source / f'Re{setup.eval_re}'), '--device', 'cuda',
            '--output', str(run_dir / 'prediction.npz'),
        ]
        # Scoring gets its own few minutes even when the run itself was cut at the deadline.
        await self._run(cmd, run_dir / 'eval.log', setup.reference / 'gino', 5.0)
        return parse_eval_log((run_dir / 'eval.log').read_text(errors='replace'))

    async def profile(self) -> dict[str, Any]:
        """Parameter shapes, step time and GPU memory of the original code (cached per campaign)."""
        setup = self.setup
        out = setup.reference / 'model_facts.json'
        if out.exists():
            return json.loads(out.read_text())
        run_dir = setup.reference / 'profile_run'
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            setup.train_python, str(Path(__file__).with_name('gino_profile.py')), str(out),
            '--data-root', str(setup.source), '--device', 'cuda', '--output-dir', str(run_dir),
            *PROFILE_ARGS,
        ]
        await self._run(cmd, run_dir / 'profile.log', setup.reference / 'gino', setup.smoke_timeout_min)
        if not out.exists():
            return {'error': 'profiling failed', 'log_tail': self._tail(run_dir / 'profile.log')}
        return json.loads(out.read_text())

    async def wait(self, max_minutes: float) -> RunRecord | None:
        """Wait for the current run; returns it when done, None if still running."""
        if self._task is None:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(self._task), max_minutes * 60.0)
        except TimeoutError:
            return None

    def progress(self) -> dict[str, Any]:
        record = self.records[-1]
        log = Path(record.run_dir) / ('script.log' if record.kind == 'script' else 'train.log')
        text = log.read_text(errors='replace') if log.exists() else ''
        parsed = parse_train_log(text)
        return {
            'run_id': record.run_id, 'status': 'running',
            'elapsed_minutes': round((time.time() - record.started) / 60.0, 1),
            'last_logged': parsed.get('last'), 'best_val_data_so_far': parsed.get('best_val_data'),
        }

    async def stop(self) -> None:
        if self._proc is not None:
            os.killpg(self._proc.pid, signal.SIGTERM)
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
