import math
from typing import List

import torch
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS


class _DeepRoofSafeGradMixin:
    """Shared non-finite gradient handling for optim wrappers."""

    def __init__(
        self,
        skip_nonfinite_grad: bool = True,
        max_nonfinite_warnings: int = 10,
        **kwargs,
    ):
        self.skip_nonfinite_grad = bool(skip_nonfinite_grad)
        self.max_nonfinite_warnings = max(int(max_nonfinite_warnings), 0)
        self._nonfinite_step_count = 0
        self._nonfinite_warn_count = 0
        super().__init__(**kwargs)

    def _collect_grad_params(self) -> List[torch.Tensor]:
        params: List[torch.Tensor] = []
        for group in self.optimizer.param_groups:
            params.extend(group.get('params', []))
        return [p for p in params if p.requires_grad and p.grad is not None]

    @staticmethod
    def _has_nonfinite_grads(params: List[torch.Tensor]) -> bool:
        for p in params:
            g = p.grad
            if g is None:
                continue
            if not torch.isfinite(g).all():
                return True
        return False

    def _record_nonfinite_event(self, reason: str, grad_value: float | None = None) -> None:
        self._nonfinite_step_count += 1
        try:
            self.message_hub.update_info('nonfinite_grad_step_count', self._nonfinite_step_count)
            self.message_hub.update_scalar('train/nonfinite_grad_steps', float(self._nonfinite_step_count))
        except Exception:
            pass

        if self._nonfinite_warn_count < self.max_nonfinite_warnings:
            iter_idx = None
            try:
                iter_idx = self.message_hub.get_info('iter', None)
            except Exception:
                iter_idx = None
            iter_msg = f'iter={iter_idx}' if iter_idx is not None else 'iter=?'
            grad_msg = '' if grad_value is None else f' grad_norm={grad_value}'
            print(
                f'[DeepRoofSafeOptim] WARNING: skipped optimizer step due to non-finite gradients '
                f'({reason}). {iter_msg}.{grad_msg}',
                flush=True,
            )
            self._nonfinite_warn_count += 1

    def _maybe_update_grad_scalar(self, grad_value: float) -> None:
        grad_name = getattr(self, 'grad_name', None)
        if grad_name is None:
            return
        try:
            self.message_hub.update_scalar(f'train/{grad_name}', float(grad_value))
        except Exception:
            pass


@OPTIM_WRAPPERS.register_module()
class DeepRoofSafeOptimWrapper(_DeepRoofSafeGradMixin, OptimWrapper):
    """OptimWrapper that skips toxic updates when gradients become non-finite."""

    def step(self, **kwargs) -> None:
        params = self._collect_grad_params()
        grad_value = None
        grad_nonfinite = False

        if self.clip_grad_kwargs and params:
            grad = self.clip_func(params, **self.clip_grad_kwargs)
            if grad is not None:
                grad_value = float(grad)
                if math.isfinite(grad_value):
                    self._maybe_update_grad_scalar(grad_value)
                else:
                    grad_nonfinite = True

        if self.skip_nonfinite_grad and params:
            if grad_nonfinite or self._has_nonfinite_grads(params):
                self._record_nonfinite_event(
                    reason='clip_grad' if grad_nonfinite else 'grad_tensor',
                    grad_value=grad_value,
                )
                for p in params:
                    p.grad = None
                return

        self.optimizer.step(**kwargs)


@OPTIM_WRAPPERS.register_module()
class DeepRoofSafeAmpOptimWrapper(_DeepRoofSafeGradMixin, AmpOptimWrapper):
    """AMP wrapper with explicit non-finite gradient tracking."""

    def step(self, **kwargs) -> None:
        # Unscale if we need clipping or explicit finite checks.
        need_unscale = self.clip_grad_kwargs is not None or self.skip_nonfinite_grad
        if need_unscale:
            self.loss_scaler.unscale_(self.optimizer)

        params = self._collect_grad_params()
        grad_value = None
        grad_nonfinite = False

        if self.clip_grad_kwargs and params:
            grad = self.clip_func(params, **self.clip_grad_kwargs)
            if grad is not None:
                grad_value = float(grad)
                if math.isfinite(grad_value):
                    self._maybe_update_grad_scalar(grad_value)
                else:
                    grad_nonfinite = True

        if self.skip_nonfinite_grad and params and (grad_nonfinite or self._has_nonfinite_grads(params)):
            self._record_nonfinite_event(
                reason='clip_grad' if grad_nonfinite else 'grad_tensor',
                grad_value=grad_value,
            )

        # GradScaler will skip the actual optimizer step if found_inf was set
        # during unscale/clipping; keep this path for correct scaler updates.
        self.loss_scaler.step(self.optimizer, **kwargs)
        self.loss_scaler.update(self._scale_update_param)
