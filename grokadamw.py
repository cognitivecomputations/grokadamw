import math
import torch
from torch.optim import Optimizer
from torch.cuda.amp import autocast
from typing import Iterable, Callable, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrokAdamW(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2, alpha_init: float = 0.98, lamb: float = 2.0,
                 gamma: float = 0.1, grokking_signal_fns: Optional[list[Callable[[], float]]] = None,
                 grokking_signal_decay_rate: float = 0.1, gradient_clipping: float = 1.0):
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha_init=alpha_init, lamb=lamb, gamma=gamma,
                        grokking_signal_fns=grokking_signal_fns,
                        grokking_signal_decay_rate=grokking_signal_decay_rate,
                        gradient_clipping=gradient_clipping)
        super(GrokAdamW, self).__init__(params, defaults)

        # Pre-allocate state tensors and move to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p] = {}
                state['step'] = 0
                state['exp_avg'] = torch.empty_like(p, memory_format=torch.preserve_format).to(device)
                state['exp_avg_sq'] = torch.empty_like(p, memory_format=torch.preserve_format).to(device)
                state['grok_ema'] = torch.empty_like(p, memory_format=torch.preserve_format).to(device)
                
                # Initialize tensors
                state['exp_avg'].zero_()
                state['exp_avg_sq'].zero_()
                state['grok_ema'].zero_()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        return self._step_impl(closure, use_amp=False)

    @torch.no_grad()
    def step_amp(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        return self._step_impl(closure, use_amp=True)

    def _step_impl(self, closure: Optional[Callable[[], float]], use_amp: bool) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            use_amp (bool): Whether to use automatic mixed precision (AMP).

        Returns:
            Optional[float]: The loss value returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grokking_signal = self._compute_grokking_signal(group)

            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if not params_with_grad:
                continue

            grads = [p.grad for p in params_with_grad]

            # Function to apply parameter updates
            def _apply_updates(): 
                self._update_group(group, params_with_grad, grads, grokking_signal)

            if use_amp:
                with autocast():
                    _apply_updates()
            else:
                _apply_updates()

        return loss

    def _compute_grokking_signal(self, group: dict) -> Optional[float]:
        """Computes a combined grokking signal from multiple functions."""
        if group['grokking_signal_fns'] is None:
            return None

        signals = []
        for fn in group['grokking_signal_fns']:
            try:
                signal = fn()
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Error in grokking_signal_fn: {e}. Ignoring this function.")
        
        if not signals:
            return None

        # Example: Taking the mean of all valid signals
        return sum(signals) / len(signals)

    @staticmethod
    def _update_group(group: dict, params: list[torch.Tensor], grads: list[torch.Tensor], 
                      grokking_signal: Optional[float]) -> None:
        for i, (p, grad) in enumerate(zip(params, grads)):
            state = group['state'][p]
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1

            # Apply gradient clipping if enabled
            if group['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(p, group['gradient_clipping'])

            # Layer-wise momentum decay
            layer_beta1 = beta1 * (1 - group['gamma'])**i

            # Grokfast component
            grok_grad = GrokAdamW._update_grok_ema(grad, state, group, grokking_signal)

            # AdamW update with Grokfast-amplified gradient
            exp_avg.mul_(layer_beta1).add_(grok_grad, alpha=1 - layer_beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)

            # AdamW bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * torch.sqrt(bias_correction2) / bias_correction1

            # Decoupled weight decay (from AdamW)
            p.mul_(1 - group['lr'] * group['weight_decay'])

            # Update parameters
            p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-step_size)

    @staticmethod
    def _update_grok_ema(grad: torch.Tensor, state: dict, group: dict, 
                         grokking_signal: Optional[float]) -> torch.Tensor:
        grok_ema = state['grok_ema']
        alpha = group['alpha_init']
        if grokking_signal is not None:
            alpha = alpha * torch.exp(-group['grokking_signal_decay_rate'] * grokking_signal)
        grok_ema.mul_(alpha).add_(grad, alpha=1 - alpha)
        return grad + group['lamb'] * grok_ema
