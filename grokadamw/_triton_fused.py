import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),
    ]

    @triton.autotune(
        configs=_AUTOTUNE_CONFIGS,
        key=["n_elements"],
        reset_to_zero=["out_grad_sq_ptr", "out_grok_sq_ptr"],
        restore_value=["grok_ema_ptr"],
    )
    @triton.jit
    def _norms_kernel(
        grad_ptr,
        grok_ema_ptr,
        out_grad_sq_ptr,
        out_grok_sq_ptr,
        alpha,
        lamb,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        grad = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grok_ema = tl.load(grok_ema_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        new_ema = alpha * grok_ema + (1.0 - alpha) * grad
        grok_grad = grad + lamb * new_ema

        tl.store(grok_ema_ptr + offs, new_ema, mask=mask)

        g_sq = tl.where(mask, grad * grad, 0.0)
        gg_sq = tl.where(mask, grok_grad * grok_grad, 0.0)
        tl.atomic_add(out_grad_sq_ptr, tl.sum(g_sq))
        tl.atomic_add(out_grok_sq_ptr, tl.sum(gg_sq))

    @triton.autotune(
        configs=_AUTOTUNE_CONFIGS,
        key=["n_elements"],
        restore_value=["p_ptr", "exp_avg_ptr", "exp_avg_sq_ptr"],
    )
    @triton.jit
    def _update_kernel(
        p_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        grok_ema_ptr,
        grad_norm_sq_ptr,
        grok_norm_sq_ptr,
        lamb,
        layer_beta1,
        beta2,
        step_size,
        lr_wd,
        eps,
        eps_norm,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        grad_norm = tl.sqrt(tl.load(grad_norm_sq_ptr))
        grok_norm = tl.sqrt(tl.load(grok_norm_sq_ptr))
        grok_grad_scale = tl.where(grok_norm > eps_norm, grad_norm / grok_norm, 1.0)
        grok_grad_scale = tl.where(grad_norm > 0.0, grok_grad_scale, 0.0)

        p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
        grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offs, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offs, mask=mask).to(tl.float32)
        grok_ema = tl.load(grok_ema_ptr + offs, mask=mask).to(tl.float32)

        grok_grad = grad + lamb * grok_ema
        grok_grad = grok_grad * grok_grad_scale
        exp_avg = layer_beta1 * exp_avg + (1.0 - layer_beta1) * grok_grad
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grok_grad * grok_grad
        p = p * (1.0 - lr_wd) - step_size * exp_avg / (tl.sqrt(exp_avg_sq) + eps)

        tl.store(p_ptr + offs, p, mask=mask)
        tl.store(exp_avg_ptr + offs, exp_avg, mask=mask)
        tl.store(exp_avg_sq_ptr + offs, exp_avg_sq, mask=mask)

    def _launch_norms(grad, grok_ema, norms_buf, alpha, lamb):
        n = grad.numel()
        if n == 0:
            return
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        norms_buf.zero_()
        _norms_kernel[grid](
            grad,
            grok_ema,
            norms_buf[:1],
            norms_buf[1:],
            alpha,
            lamb,
            n,
        )

    def _launch_update(
        p,
        grad,
        exp_avg,
        exp_avg_sq,
        grok_ema,
        norms_buf,
        lamb,
        layer_beta1,
        beta2,
        step_size,
        lr_wd,
        eps,
    ):
        n = p.numel()
        if n == 0:
            return
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _update_kernel[grid](
            p,
            grad,
            exp_avg,
            exp_avg_sq,
            grok_ema,
            norms_buf[:1],
            norms_buf[1:],
            lamb,
            layer_beta1,
            beta2,
            step_size,
            lr_wd,
            eps,
            eps,
            n,
        )

    def grokadamw_fused_step(
        p,
        grad,
        exp_avg,
        exp_avg_sq,
        grok_ema,
        alpha,
        lamb,
        layer_beta1,
        beta2,
        step_size,
        lr_wd,
        eps,
        grad_norm=None,
        norms_buf=None,
    ):
        n_elements = p.numel()
        if n_elements == 0:
            return True

        if not (
            p.is_contiguous()
            and exp_avg.is_contiguous()
            and exp_avg_sq.is_contiguous()
            and grok_ema.is_contiguous()
        ):
            return False

        grad = grad.contiguous()

        if norms_buf is None:
            norms_buf = torch.zeros(2, device=p.device, dtype=torch.float32)

        _launch_norms(grad, grok_ema, norms_buf, alpha, lamb)
        _launch_update(
            p,
            grad,
            exp_avg,
            exp_avg_sq,
            grok_ema,
            norms_buf,
            lamb,
            layer_beta1,
            beta2,
            step_size,
            lr_wd,
            eps,
        )
        return True

    @torch.no_grad()
    def grokadamw_fused_group(
        params_grads_states,
        alpha,
        lamb,
        beta2,
        eps,
    ):
        n_params = len(params_grads_states)
        if n_params == 0:
            return

        for i, (p, grad, grok_ema, exp_avg, exp_avg_sq, meta) in enumerate(
            params_grads_states
        ):
            if not grad.is_contiguous():
                grad = grad.contiguous()
            if not p.is_contiguous():
                p = p.contiguous()
            if not exp_avg.is_contiguous():
                exp_avg = exp_avg.contiguous()
            if not exp_avg_sq.is_contiguous():
                exp_avg_sq = exp_avg_sq.contiguous()
            if not grok_ema.is_contiguous():
                grok_ema = grok_ema.contiguous()
            params_grads_states[i] = (p, grad, grok_ema, exp_avg, exp_avg_sq, meta)

        for i, (p, grad, grok_ema, exp_avg, exp_avg_sq, meta) in enumerate(
            params_grads_states
        ):
            norms_buf = meta["norms_buf"]
            _launch_norms(grad, grok_ema, norms_buf, alpha, lamb)
            _launch_update(
                p,
                grad,
                exp_avg,
                exp_avg_sq,
                grok_ema,
                norms_buf,
                lamb,
                meta["layer_beta1"],
                beta2,
                meta["step_size"],
                meta["lr_wd"],
                eps,
            )
