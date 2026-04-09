import math
import torch
from torch.optim import Optimizer
from typing import Iterable, Callable, Optional
import logging

try:
    from grokadamw._triton_fused import (
        grokadamw_fused_group,
        grokadamw_fused_step,
        _TRITON_AVAILABLE,
    )
except ImportError:
    _TRITON_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _foreach_update(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    grok_emas: list[torch.Tensor],
    grok_grad_bufs: list[torch.Tensor],
    alpha: float,
    lamb: float,
    layer_beta1_list: list[float],
    beta2: float,
    step_size_list: list[float],
    lr_wd: float,
    eps: float,
) -> None:
    n = len(params)

    grad_norms = torch._foreach_norm(grads)

    torch._foreach_mul_(grok_emas, alpha)
    torch._foreach_add_(grok_emas, grads, alpha=1 - alpha)
    torch._foreach_mul_(params, 1 - lr_wd)
    torch._foreach_copy_(grok_grad_bufs, grads)
    torch._foreach_add_(grok_grad_bufs, grok_emas, alpha=lamb)

    grok_grad_norms = torch._foreach_norm(grok_grad_bufs)
    scales = [gn / ggn.clamp(min=eps) for gn, ggn in zip(grad_norms, grok_grad_norms)]
    torch._foreach_mul_(grok_grad_bufs, scales)

    for i in range(n):
        grok_grad = grok_grad_bufs[i]
        exp_avgs[i].mul_(layer_beta1_list[i]).add_(
            grok_grad, alpha=1 - layer_beta1_list[i]
        )
        exp_avg_sqs[i].mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)
        denom = exp_avg_sqs[i].sqrt().add_(eps)
        params[i].addcdiv_(exp_avgs[i], denom, value=-step_size_list[i])


class GrokAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        alpha_init: float = 0.98,
        lamb: float = 2.0,
        gamma: float = 0.1,
        grokking_signal_fns: Optional[list[Callable[[], float]]] = None,
        grokking_signal_decay_rate: float = 0.1,
        cpu_offload: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError(f"Invalid alpha_init value: {alpha_init}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha_init=alpha_init,
            lamb=lamb,
            gamma=gamma,
            grokking_signal_fns=grokking_signal_fns,
            grokking_signal_decay_rate=grokking_signal_decay_rate,
            cpu_offload=cpu_offload,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        return self._step_impl(closure)

    def _step_impl(self, closure: Optional[Callable[[], float]]) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grokking_signal = self._compute_grokking_signal(group)

            params_with_grad = [p for p in group["params"] if p.grad is not None]
            if not params_with_grad:
                continue

            grads = [p.grad for p in params_with_grad]

            self._update_group(group, params_with_grad, grads, grokking_signal)

        return loss

    @staticmethod
    def _default_grokking_signal(
        train_loss: Optional[float], eval_loss: Optional[float]
    ) -> float:
        if train_loss is None or eval_loss is None:
            return 0.0
        diff = max(0, eval_loss - train_loss)
        max_loss = max(eval_loss, train_loss)
        return diff / max_loss if max_loss > 0 else 0.0

    def _compute_grokking_signal(self, group: dict) -> Optional[float]:
        if group["grokking_signal_fns"] is None:
            train_loss = group.get("train_loss", None)
            eval_loss = group.get("eval_loss", None)
            return self._default_grokking_signal(train_loss, eval_loss)

        signals = []
        for fn in group["grokking_signal_fns"]:
            try:
                signal = fn()
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.warning(
                    f"Error in grokking_signal_fn: {e}. Ignoring this function."
                )

        return sum(signals) / len(signals) if signals else None

    def _update_group(
        self,
        group: dict,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        grokking_signal: Optional[float],
    ) -> None:
        cpu_offload = group.get("cpu_offload", False)
        beta1, beta2 = group["betas"]
        lr_wd = group["lr"] * group["weight_decay"]
        alpha = group["alpha_init"]
        if grokking_signal is not None:
            alpha = alpha * math.exp(
                -group["grokking_signal_decay_rate"] * grokking_signal
            )

        active_params = []
        active_grads = []
        active_exp_avg = []
        active_exp_avg_sq = []
        active_grok_ema = []
        active_grok_grad_buf = []
        layer_beta1_list = []
        step_size_list = []

        triton_batch = []
        cached_step_size = None

        for i, (p, grad) in enumerate(zip(params, grads)):
            if p.numel() == 0:
                continue

            state = self.state[p]
            if not state:
                state_device = "cpu" if cpu_offload else p.device
                state.update(
                    {
                        "step": 0,
                        "exp_avg": torch.zeros(
                            p.shape, dtype=torch.float32, device=state_device
                        ),
                        "exp_avg_sq": torch.zeros(
                            p.shape, dtype=torch.float32, device=state_device
                        ),
                        "grok_ema": torch.zeros(
                            p.shape, dtype=torch.float32, device=state_device
                        ),
                        "norms_buf": torch.zeros(
                            2, dtype=torch.float32, device=p.device
                        ),
                    }
                )

            if "norms_buf" not in state:
                state["norms_buf"] = torch.zeros(
                    2, dtype=torch.float32, device=p.device
                )

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            grok_ema = state["grok_ema"]

            if cpu_offload:
                exp_avg = exp_avg.to(p.device)
                exp_avg_sq = exp_avg_sq.to(p.device)
                grok_ema = grok_ema.to(p.device)

            state["step"] += 1

            if cached_step_size is None:
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                cached_step_size = (
                    group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                )

            layer_beta1 = beta1 * (1 - group["gamma"]) ** i
            step_size = cached_step_size

            if _TRITON_AVAILABLE and p.is_cuda and not cpu_offload:
                triton_batch.append(
                    (
                        p,
                        grad,
                        grok_ema,
                        exp_avg,
                        exp_avg_sq,
                        {
                            "layer_beta1": layer_beta1,
                            "step_size": step_size,
                            "lr_wd": lr_wd,
                            "norms_buf": state["norms_buf"],
                        },
                    )
                )
                continue

            if not grad.is_contiguous():
                grad = grad.contiguous()
            if p.dtype != torch.float32:
                if not cpu_offload:
                    if "grad_fp32_buf" not in state:
                        state["grad_fp32_buf"] = torch.zeros(
                            p.shape,
                            dtype=torch.float32,
                            device=p.device,
                        )
                    grad = state["grad_fp32_buf"].copy_(grad)
                else:
                    grad = grad.float()
            else:
                grad = grad.float()

            if "grok_grad_buf" not in state:
                state["grok_grad_buf"] = torch.zeros(
                    p.shape,
                    dtype=torch.float32,
                    device="cpu" if cpu_offload else p.device,
                )
            grok_grad_buf = state["grok_grad_buf"]
            if cpu_offload:
                grok_grad_buf = grok_grad_buf.to(p.device)

            active_params.append(p)
            active_grads.append(grad)
            active_exp_avg.append(exp_avg)
            active_exp_avg_sq.append(exp_avg_sq)
            active_grok_ema.append(grok_ema)
            active_grok_grad_buf.append(grok_grad_buf)
            layer_beta1_list.append(layer_beta1)
            step_size_list.append(step_size)

        if triton_batch:
            grokadamw_fused_group(
                triton_batch,
                alpha,
                group["lamb"],
                beta2,
                group["eps"],
            )

        if not active_params:
            return

        _foreach_update(
            active_params,
            active_grads,
            active_exp_avg,
            active_exp_avg_sq,
            active_grok_ema,
            active_grok_grad_buf,
            alpha,
            group["lamb"],
            layer_beta1_list,
            beta2,
            step_size_list,
            lr_wd,
            group["eps"],
        )

        if cpu_offload:
            for p, exp_avg, exp_avg_sq, grok_ema, grok_grad_buf in zip(
                active_params,
                active_exp_avg,
                active_exp_avg_sq,
                active_grok_ema,
                active_grok_grad_buf,
            ):
                state = self.state[p]
                state["exp_avg"] = exp_avg.to("cpu")
                state["exp_avg_sq"] = exp_avg_sq.to("cpu")
                state["grok_ema"] = grok_ema.to("cpu")
                state["grok_grad_buf"] = grok_grad_buf.to("cpu")

    def state_dict(self):
        state_dict = super().state_dict()
        for group in state_dict["param_groups"]:
            group["grokking_signal_fns"] = None
        for param_id, state in state_dict.get("state", {}).items():
            state.pop("grok_grad_buf", None)
            state.pop("norms_buf", None)
            state.pop("grad_fp32_buf", None)
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        cpu_offload = self.defaults.get("cpu_offload", False)
        target_device = "cpu" if cpu_offload else None
        for group in self.param_groups:
            group["grokking_signal_fns"] = self.defaults["grokking_signal_fns"]
            group["cpu_offload"] = cpu_offload
            for p in group["params"]:
                state = self.state.get(p, {})
                if state:
                    for key in ("exp_avg", "exp_avg_sq", "grok_ema", "grok_grad_buf"):
                        if key in state and isinstance(state[key], torch.Tensor):
                            state[key] = state[key].to(dtype=torch.float32)
                            if target_device is not None:
                                state[key] = state[key].to(device=target_device)
                            elif state[key].device != p.device:
                                state[key] = state[key].to(device=p.device)
                    if "norms_buf" not in state:
                        state["norms_buf"] = torch.zeros(
                            2, dtype=torch.float32, device=p.device
                        )

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("grokking_signal_fns", None)
            group.setdefault("grokking_signal_decay_rate", 0.1)
            group.setdefault("cpu_offload", False)
