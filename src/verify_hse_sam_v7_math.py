"""Small independent CPU checks for the V7 SAM update and resume mechanics."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from torch import nn
import train_hse_sam_v7 as v
import train_hse_specialist as h

ROOT = Path(__file__).resolve().parents[1]

def net(dropout=0.):
    return nn.Sequential(nn.Linear(7, 11), nn.Tanh(), nn.Dropout(dropout), nn.Linear(11, 6)).train()

def opt(model):
    return torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)

def snapshot(model):
    return {k: x.detach().clone() for k, x in model.state_dict().items()}

def assert_same(a, b, atol=0.):
    assert set(a) == set(b)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0., atol=atol, msg=key)

def scaler(enabled=False):
    return v.ManualGradScaler(enabled=enabled, init_scale=1024., growth_interval=2000)

def reference_step(model, optimizer, x, y, rho):
    optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(x), y).backward()
    original = snapshot(model)
    if rho:
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()]))
        with torch.no_grad():
            for p in model.parameters():
                p.add_(p.grad * (rho / (norm + 1e-12)))
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), y).backward()
        model.load_state_dict(original)
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261016)
    x = torch.randn(30, 7)
    y = torch.arange(30) % 6
    checks = {}
    for rho in (0., .05):
        a = net(); b = copy.deepcopy(a); d = copy.deepcopy(a)
        oa, ob, od = opt(a), opt(b), opt(d)
        result = v.group_step(a, oa, scaler(), None, x, y, device='cpu', batch_size=16, rho=rho)
        result_full = v.group_step(b, ob, scaler(), None, x, y, device='cpu', batch_size=30, rho=rho)
        reference_step(d, od, x, y, rho)
        assert result['success'] and result_full['success'] and result['seen'] == 30
        assert_same(snapshot(a), snapshot(b), 2e-7)
        assert_same(snapshot(a), snapshot(d), 2e-7)
        checks[f'rho_{rho}_partial_group_vs_full_and_reference'] = True

    a = net(.25); b = copy.deepcopy(a)
    before = torch.get_rng_state().clone()
    v.group_step(a, opt(a), scaler(), None, x, y, device='cpu', batch_size=16, rho=0.)
    first_end = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    v.group_step(b, opt(b), scaler(), None, x, y, device='cpu', batch_size=16, rho=.05)
    assert torch.equal(first_end, torch.get_rng_state())
    checks['sam_rng_matches_one_pass_with_dropout'] = True

    class FailSecondPass(nn.Module):
        def __init__(self):
            super().__init__(); self.base = net(); self.calls = 0
        def forward(self, z):
            self.calls += 1
            value = self.base(z)
            return value * float('nan') if self.calls > 2 else value
    a = FailSecondPass(); oa = opt(a); sa = scaler(True); ea = v.FullModelEMA(a, decay=.995)
    initial = snapshot(a)
    skipped = v.group_step(a, oa, sa, ea, x, y, device='cpu', batch_size=16, rho=.05)
    assert not skipped['success']
    assert_same(snapshot(a), initial)
    assert len(oa.state) == 0 and ea.updates == 0 and sa.get_scale() == 512.
    checks['nonfinite_second_pass_restores_parameters_skips_optimizer_ema_and_halves_scale'] = True

    a = net(.2); oa = opt(a); sa = scaler(); ea = v.FullModelEMA(a, decay=.995)
    assert v.group_step(a, oa, sa, ea, x, y, device='cpu', batch_size=16, rho=.05)['success']
    state = copy.deepcopy({'model': a.state_dict(), 'optimizer': oa.state_dict(), 'scaler': sa.state_dict(),
                           'ema': ea.state_dict(), 'rng': torch.get_rng_state()})
    assert v.group_step(a, oa, sa, ea, x, y, device='cpu', batch_size=16, rho=.05)['success']
    end = snapshot(a)
    b = net(.2); ob = opt(b); sb = scaler(); eb = v.FullModelEMA(b, decay=.995)
    b.load_state_dict(state['model']); ob.load_state_dict(state['optimizer'])
    sb.load_state_dict(state['scaler']); eb.load_state_dict(state['ema']); torch.set_rng_state(state['rng'])
    assert v.group_step(b, ob, sb, eb, x, y, device='cpu', batch_size=16, rho=.05)['success']
    assert_same(end, snapshot(b)); assert ea.updates == eb.updates == 2
    assert_same(ea.shadow, eb.shadow)
    checks['resumed_update_and_ema_bitwise_equal'] = True
    report = {'state': 'passed', 'device': 'cpu', 'checks': checks,
              'trainer_sha256': h.sha256(Path(v.__file__)), 'verifier_sha256': h.sha256(__file__)}
    h.write_json(ROOT / 'logs/hse_sam_v7_math_verification.json', report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
