"""--lr must take effect when resuming.

torch schedulers capture base_lrs from the optimizer at construction time, so
loading a checkpoint's optimizer state before building the scheduler silently
overrides the requested learning rate with whatever the previous run ended on.
That happened in the v9 run: --lr 1e-4 --lr-min 1e-4 was meant to hold the LR
constant, but the restored 1e-5 became base_lr and the schedule ran upward
from 1e-5 to 1e-4 instead.
"""
import pytest

torch = pytest.importorskip("torch")

from train_conditional import reset_optimizer_lr  # noqa: E402


def make_optimizer(lr):
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(2))], lr=lr)


def test_requested_lr_overrides_restored_state():
    finished_run = make_optimizer(1e-5)                 # a run that annealed down
    resumed = make_optimizer(1e-3)
    resumed.load_state_dict(finished_run.state_dict())  # as --resume does
    assert resumed.param_groups[0]["lr"] == pytest.approx(1e-5)

    reset_optimizer_lr(resumed, 1e-4)
    assert resumed.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_scheduler_built_afterwards_uses_the_requested_lr():
    opt = make_optimizer(1e-3)
    opt.load_state_dict(make_optimizer(1e-5).state_dict())
    reset_optimizer_lr(opt, 1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-4)
    assert sched.base_lrs == [pytest.approx(1e-4)]
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert all(lr == pytest.approx(1e-4) for lr in lrs), f"LR should stay constant, got {lrs}"


def test_annealing_still_works_when_min_is_lower():
    opt = make_optimizer(1e-4)
    reset_optimizer_lr(opt, 1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
    first = opt.param_groups[0]["lr"]
    for _ in range(10):
        sched.step()
    assert opt.param_groups[0]["lr"] < first
