"""Compare distributed accumulation and resume against a global-batch reference.

Run with torchrun --standalone --nproc_per_node=8 tests/test_distributed.py --output DIR.
"""
import argparse
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch import nn

from datasets.metadata import schedule
from evaluation.metrics import METRICS
from util import checkpoint, engine
from util.distributed import initialize, on_main, gather_objects, check_resources
from util.io import seed


class Dataset:
    epoch = 0

    def __len__(self):
        return 133

    def __getitem__(self, key):
        index, real = key
        return dict(index=index, real=real, sample=dict(n=index % 5 + 1, x=(index + 1) / 200))


def collate(group, device):
    valid = [item for item in group if item['real']]
    items = valid or group[:1]
    return dict(x=torch.tensor([item['sample']['x'] for item in items], device=device, dtype=torch.float64),
                n=torch.tensor([item['sample']['n'] for item in items], device=device, dtype=torch.float64),
                real=bool(valid)), None, None


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.3]))

    def forward(self, value):
        return value * self.weight


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioner = Scale()
        self.decoder = Scale()
        self.other = nn.Parameter(torch.tensor(0.1))
        self.unused = nn.Parameter(torch.tensor(0.0))
        self.loss_weight = dict(reconstruction=0.3, eos=0.2)

    def set_recon_epoch(self, epoch):
        pass

    def recon_weight_factor(self):
        return 1.0


class Model(nn.Module):
    assignment_weight = 0.4

    def __init__(self):
        super().__init__()
        self.model = Body()
        self.selector = Scale()
        self.double()

    def counts(self, sample):
        n = sample['n']
        return n, n + 1, n - 1

    def clear_cache(self):
        pass

    def forward(self, data, coords, targets, denominators, **kwargs):
        x, n = data['x'], data['n']
        y = self.model.decoder(self.model.conditioner(x[:, None])).flatten() + self.model.other
        errors = (y - x).square()
        select = (self.selector(x[:, None]).flatten() - x).square()
        real = float(data['real'])
        values = [(errors * n).sum(), (errors * (n + 1)).sum(),
                  (errors * (n - 1)).sum(), (select * n).sum(),
                  n.sum(), (n + 1).sum(), (n - 1).sum(), n.new_tensor(len(n))]
        values = [v * real for v in values]
        d = denominators.clamp_min(1)
        loss = values[0] / d[0] + 0.2 * values[1] / d[1] + 0.3 * values[2] / d[2] + 0.4 * values[3] / d[0]
        return dict(objective=loss, metrics=dict(zip(METRICS, [v.detach() for v in values])))


class Guard:
    stopped = False

    def check(self):
        pass

    def boundary(self):
        return 'test_stop' if self.stopped else None


def optimizer_for(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    return optimizer, scheduler


def cpu_rng():
    with patch('torch.cuda.is_available', return_value=False):
        return checkpoint.rng_state()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    acc = Accelerator(cpu=True, mixed_precision='no', kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True), InitProcessGroupKwargs(backend='gloo')])
    initialize()
    world, rank = acc.num_processes, acc.process_index
    assert world == 8, 'This regression covers the eight-rank configuration.'
    output = Path(args.output)
    on_main(acc, lambda: output.mkdir(parents=True, exist_ok=False))
    cfg = dict(train=dict(batch_size=4, grad_accum_every=4, seed=100,
                          checkpoint_every_updates=999),
               distributed=dict(workers_per_rank=0), initialization=dict(source_epoch=0),
               experiment=dict(data_seed=123), scene_mix=dict(enabled=False),
               soft_guidance=dict(beta_max=2.0, beta_warmup_updates=100))
    dataset = Dataset()
    parts = gather_objects(schedule(len(dataset), 0, 123, world, rank))
    real_ids = [i for part in parts for i, real in part if real]
    assert sorted(real_ids) == list(range(len(dataset)))
    assert len({len(part) for part in parts}) == 1
    engine.batch = collate
    seed(12)
    raw = Model()
    initial = copy.deepcopy(raw.state_dict())
    optimizer, scheduler = optimizer_for(raw)
    wrapped, optimizer = acc.prepare(raw, optimizer)
    state = engine.TrainingState()
    guard = Guard() if rank == 0 else None
    identity = dict(config='distributed-test', assets={})

    def save_latest():
        rngs = gather_objects(cpu_rng())
        on_main(acc, lambda: checkpoint.write(output / 'latest.pt', raw, optimizer, scheduler,
            state.step, state.epoch, state.offset, state.totals, identity, rng_by_rank=rngs))

    assert not engine.train_one_epoch(wrapped, raw, dataset, optimizer, scheduler, acc,
                                     cfg, state, output, guard, save_latest)
    assert state.step == 2 and state.offset == 5
    assert state.totals[-1] == 133
    reference = Model()
    reference.load_state_dict(initial)
    ref_optimizer, ref_scheduler = optimizer_for(reference)
    order = schedule(len(dataset), 0, 123, 1, 0)
    reference_totals = [0.0] * len(METRICS)
    for start in (0, 128):
        group = [dataset[item] for item in order[start:start + 128]]
        inputs, coords, targets = collate(group, acc.device)
        den = torch.tensor([sum(reference.counts(d['sample'])[k] for d in group) for k in range(3)], dtype=torch.float32)
        ref_optimizer.zero_grad(set_to_none=True)
        result = reference(inputs, coords, targets, den)
        result['objective'].backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        ref_optimizer.step()
        ref_scheduler.step()
        reference_totals = [a + float(result['metrics'][k]) for a, k in zip(reference_totals, METRICS)]
    for name, value in raw.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.tensor(state.totals), torch.tensor(reference_totals))
    assert len(set(gather_objects(checkpoint.object_digest(raw.state_dict())))) == 1

    restored = Model()
    opt2, sch2 = optimizer_for(restored)
    resumed, rng = checkpoint.restore_training(restored, opt2, sch2, output / 'latest.pt', identity,
                                               rank=rank, world_size=world)
    wrapped2, opt2 = acc.prepare(restored, opt2)
    checkpoint.restore_rng(rng)
    assert checkpoint.object_digest(cpu_rng()) == checkpoint.object_digest(rng)
    assert resumed.step == 1 and resumed.offset == 4
    resumed_output = output / 'resumed'
    on_main(acc, lambda: resumed_output.mkdir())
    assert engine.train_one_epoch(wrapped2, restored, dataset, opt2, sch2, acc, cfg, resumed,
                                 resumed_output, guard, lambda: None, max_updates=1, initial_step=1)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, raw.state_dict()[name], atol=0, rtol=0)
    assert checkpoint.object_digest(opt2.state_dict()) == checkpoint.object_digest(optimizer.state_dict())
    assert sch2.state_dict() == scheduler.state_dict()
    assert resumed.totals == state.totals
    marker = on_main(acc, lambda: {'validation_owner': rank}, broadcast_result=True)
    assert marker == {'validation_owner': 0}
    if guard:
        guard.stopped = True
    assert check_resources(acc, guard, boundary=True) == 'test_stop'
    failure_seen = False
    try:
        on_main(acc, lambda: (_ for _ in ()).throw(ValueError('test error propagation')))
    except (ValueError, RuntimeError) as exc:
        failure_seen = 'test error propagation' in str(exc)
    assert failure_seen

    def finish():
        rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        assert len(rows) == 2 and [r['scenes'] for r in rows] == [128, 5]
        assert [r['scene_mix']['source_scenes'] for r in rows] == [128, 5]
        report = dict(passed=True, world_size=world, batch_size=4, accumulation=4,
                      effective_batch=128, unique_real_samples=133, dummy_samples=3,
                      global_batch_gradient_equivalence=True, global_metrics=True,
                      resume_model_optimizer_scheduler_exact=True, per_rank_rng_restored=True,
                      main_process_writes_only=True, coordinated_stop_and_error=True)
        (output / 'VERIFIED.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    on_main(acc, finish)
    acc.end_training()


if __name__ == '__main__':
    main()
