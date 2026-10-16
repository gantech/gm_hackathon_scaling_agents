"""Profile a short GINO training run so the agents can debate optimizer feasibility with numbers.

Runs ``train.py`` (from the current directory) in-process with the given arguments and records
the parameters handed to the first optimizer (count, shapes, largest dimensions), the time per
epoch and per optimizer step (data-only and physics epochs), and the GPU memory used and available. Writes
JSON to the first argument::

    python gino_profile.py facts.json --epochs 6 --warmup-epochs 2 --ramp-epochs 2 ...
"""

from __future__ import annotations

import itertools
import json
import runpy
import sys
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook
from torch.optim.optimizer import register_optimizer_step_pre_hook

out_path = Path(sys.argv[1])
train_args = sys.argv[2:]
params: list[torch.Tensor] = []
step_times: list[float] = []
_start: list[float] = []
_orig_init = torch.optim.Optimizer.__init__


def _capture_init(self, param_list, defaults):
    param_list = list(param_list)
    if not params:
        for item in param_list:
            params.extend(item['params'] if isinstance(item, dict) else [item])
    _orig_init(self, param_list, defaults)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pre(optimizer, args, kwargs) -> None:
    _sync()
    _start.append(time.perf_counter())


def _post(optimizer, args, kwargs) -> None:
    _sync()
    step_times.append(time.perf_counter() - _start[-1])


torch.optim.Optimizer.__init__ = _capture_init
register_optimizer_step_pre_hook(_pre)
register_optimizer_step_post_hook(_post)

wall = time.perf_counter()
sys.path.insert(0, str(Path.cwd()))
sys.argv = ['train.py', *train_args]
runpy.run_path('train.py', run_name='__main__')
wall = time.perf_counter() - wall

shapes = [tuple(p.shape) for p in params]
numel = [p.numel() for p in params]
dims = sorted({d for s in shapes for d in s}, reverse=True)
largest = sorted(zip(numel, shapes), reverse=True)[:8]
facts = {
    'parameters': sum(numel),
    'tensors': len(params),
    'dtype': str(params[0].dtype) if params else None,
    'largest_tensors': [{'numel': n, 'shape': list(s)} for n, s in largest],
    'largest_dims': dims[:8],
    'tensors_by_ndim': {str(k): sum(1 for s in shapes if len(s) == k) for k in sorted({len(s) for s in shapes})},
    'optimizer_step_seconds': [round(t, 5) for t in step_times],
    # Start-to-start of consecutive steps: one full epoch (forward, backward, physics, logging).
    'epoch_seconds': [round(b - a, 4) for a, b in itertools.pairwise(_start)],
    'run_wall_seconds': round(wall, 1),
    'run_args': train_args,
}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    facts.update({
        'gpu': props.name,
        'gpu_memory_total_gb': round(props.total_memory / 1e9, 1),
        'gpu_memory_peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
    })
n = facts['parameters']
facts['dense_n_by_n_matrix_gb'] = {'float32': round(4 * n * n / 1e9, 1), 'float64': round(8 * n * n / 1e9, 1)}
out_path.write_text(json.dumps(facts, indent=1))
print(json.dumps(facts, indent=1))
