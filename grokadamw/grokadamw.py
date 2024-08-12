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

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        return self._step_impl(closure)

    def _step_impl(self, closure: Optional[Callable[[], float]]) -> Optional[float]:
        """Performs a single optimization step."""
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

            self._update_group(group, params_with_grad, grads, grokking_signal)

        return loss

    @staticmethod
    def _default_grokking_signal(train_loss: Optional[float], eval_loss: Optional[float]) -> float:
        """Default grokking signal function based on loss difference."""
        if train_loss is None or eval_loss is None:
            return 0.0
        diff = max(0, eval_loss - train_loss)
        max_loss = max(eval_loss, train_loss)
        return diff / max_loss if max_loss > 0 else 0.0

    def _compute_grokking_signal(self, group: dict) -> Optional[float]:
        """Computes a combined grokking signal from multiple functions."""
        if group['grokking_signal_fns'] is None:
            train_loss = group.get('train_loss', None)
            eval_loss = group.get('eval_loss', None)
            return self._default_grokking_signal(train_loss, eval_loss)

        signals = []
        for fn in group['grokking_signal_fns']:
            try:
                signal = fn()
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Error in grokking_signal_fn: {e}. Ignoring this function.")
        
        return sum(signals) / len(signals) if signals else None

    @staticmethod
    def _update_group(group: dict, params: list[torch.Tensor], grads: list[torch.Tensor], 
                      grokking_signal: Optional[float]) -> None:
        for i, (p, grad) in enumerate(zip(params, grads)):
            state = group.get('state', {}).get(p, {})
            if not state:
                state = {'step': 0, 'exp_avg': torch.zeros_like(p, device='cpu'), 
                         'exp_avg_sq': torch.zeros_like(p, device='cpu'), 
                         'grok_ema': torch.zeros_like(p, device='cpu')}
                if 'state' not in group:
                    group['state'] = {}
                group['state'][p] = state
            
            exp_avg, exp_avg_sq = state['exp_avg'].to(p.device), state['exp_avg_sq'].to(p.device)
            grok_ema = state['grok_ema'].to(p.device)
            beta1, beta2 = group['betas']

            state['step'] += 1

            if group['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(p, group['gradient_clipping'])

            with autocast():
                layer_beta1 = beta1 * (1 - group['gamma'])**i

                # Update grok_ema
                alpha = group['alpha_init']
                if grokking_signal is not None:
                    alpha = alpha * math.exp(-group['grokking_signal_decay_rate'] * grokking_signal)
                grok_ema.mul_(alpha).add_(grad, alpha=1 - alpha)
                grok_grad = grad + group['lamb'] * grok_ema

                # Update moments
                exp_avg.mul_(layer_beta1).add_(grok_grad, alpha=1 - layer_beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-step_size)

            # Move states back to CPU
            state['exp_avg'] = exp_avg.to('cpu')
            state['exp_avg_sq'] = exp_avg_sq.to('cpu')
            state['grok_ema'] = grok_ema.to('cpu')

    def state_dict(self):
        state_dict = super().state_dict()
        for group in state_dict['param_groups']:
            group['grokking_signal_fns'] = None  # Cannot serialize functions
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group['grokking_signal_fns'] = self.defaults['grokking_signal_fns']

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('grokking_signal_fns', [])
            group.setdefault('grokking_signal_decay_rate', 0.1)
            group.setdefault('gradient_clipping', 1.0)