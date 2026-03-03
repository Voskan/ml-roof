import pytest
import torch


def test_weight_reduce_loss_ignores_nonfinite_avg_factor():
    pytest.importorskip('mmseg')
    from deeproof.models.losses import _weight_reduce_loss

    loss = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
    avg_factor = torch.tensor([float('inf')], dtype=torch.float16)

    out = _weight_reduce_loss(loss, reduction='mean', avg_factor=avg_factor)
    assert torch.isfinite(out).item()
    assert torch.allclose(out, loss.mean())


def test_hybrid_mask_loss_stays_finite_with_fp16_inf_avg_factor():
    pytest.importorskip('mmseg')
    from deeproof.models.losses import HybridMaskLoss

    n = 70000  # > float16 max finite scalar (65504)
    pred = torch.randn(n, dtype=torch.float32, requires_grad=True)
    target = (torch.rand(n) > 0.5).float()
    avg_factor = torch.tensor([float('inf')], dtype=torch.float16)

    loss_fn = HybridMaskLoss(
        bce_weight=1.0,
        lovasz_weight=1.0,
        reduction='mean',
        loss_weight=1.0,
        debug_first_n_calls=0,
    )
    loss = loss_fn(pred, target, avg_factor=avg_factor)

    assert torch.isfinite(loss).item()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
