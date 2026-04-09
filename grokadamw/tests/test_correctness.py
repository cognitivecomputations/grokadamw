import pytest
import torch
import torch.nn as nn
import copy
from grokadamw import GrokAdamW


def make_model():
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )


def train_step(model, optimizer, dtype=torch.float32):
    x = torch.randn(16, 64, device=next(model.parameters()).device, dtype=dtype)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


class TestNumericalEquivalence:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fp32_cpu_vs_gpu_states(self):
        torch.manual_seed(42)
        model_ref = make_model().cuda()
        state = copy.deepcopy(model_ref.state_dict())

        torch.manual_seed(42)
        model_test = make_model().cuda()
        model_test.load_state_dict(state)

        opt_ref = GrokAdamW(model_ref.parameters(), cpu_offload=True)
        opt_test = GrokAdamW(model_test.parameters(), cpu_offload=False)

        for step in range(50):
            torch.manual_seed(step + 1000)
            loss_ref = train_step(model_ref, opt_ref)
            torch.manual_seed(step + 1000)
            loss_test = train_step(model_test, opt_test)

            for j, (p_ref, p_test) in enumerate(
                zip(model_ref.parameters(), model_test.parameters())
            ):
                assert torch.allclose(
                    p_ref.float(), p_test.float(), atol=1e-5, rtol=1e-4
                ), (
                    f"Mismatch at step {step}, param {j}: max_diff={((p_ref - p_test).abs().max()).item():.2e}"
                )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bf16_cpu_vs_gpu_states(self):
        torch.manual_seed(42)
        model_ref = make_model().cuda().to(torch.bfloat16)
        state = copy.deepcopy(model_ref.state_dict())

        torch.manual_seed(42)
        model_test = make_model().cuda().to(torch.bfloat16)
        model_test.load_state_dict(state)

        opt_ref = GrokAdamW(model_ref.parameters(), cpu_offload=True)
        opt_test = GrokAdamW(model_test.parameters(), cpu_offload=False)

        for step in range(10):
            torch.manual_seed(step + 1000)
            loss_ref = train_step(model_ref, opt_ref, dtype=torch.bfloat16)
            torch.manual_seed(step + 1000)
            loss_test = train_step(model_test, opt_test, dtype=torch.bfloat16)

            for j, (p_ref, p_test) in enumerate(
                zip(model_ref.parameters(), model_test.parameters())
            ):
                assert torch.allclose(
                    p_ref.float(), p_test.float(), atol=5e-3, rtol=0.02
                ), (
                    f"Mismatch at step {step}, param {j}: max_diff={((p_ref - p_test).abs().max()).item():.2e}"
                )


class TestTritonEquivalence:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_matches_pytorch(self):
        try:
            from grokadamw._triton_fused import _TRITON_AVAILABLE

            if not _TRITON_AVAILABLE:
                pytest.skip("Triton not available")
        except ImportError:
            pytest.skip("Triton not available")

        initial_state = None

        torch.manual_seed(42)
        model_ref = make_model().cuda()
        initial_state = copy.deepcopy(model_ref.state_dict())

        torch.manual_seed(42)
        model_test = make_model().cuda()
        model_test.load_state_dict(copy.deepcopy(initial_state))

        opt_ref = GrokAdamW(model_ref.parameters(), cpu_offload=False)
        opt_test = GrokAdamW(model_test.parameters(), cpu_offload=False)

        import grokadamw.grokadamw as opt_mod

        original_triton = opt_mod._TRITON_AVAILABLE
        try:
            opt_mod._TRITON_AVAILABLE = False
            for step in range(50):
                torch.manual_seed(step + 1000)
                train_step(model_ref, opt_ref)

            opt_mod._TRITON_AVAILABLE = True
            for step in range(50):
                torch.manual_seed(step + 1000)
                train_step(model_test, opt_test)

            for j, (p_ref, p_test) in enumerate(
                zip(model_ref.parameters(), model_test.parameters())
            ):
                assert torch.allclose(
                    p_ref.float(), p_test.float(), atol=1e-5, rtol=1e-4
                ), (
                    f"Mismatch at param {j}: max_diff={((p_ref - p_test).abs().max()).item():.2e}"
                )
        finally:
            opt_mod._TRITON_AVAILABLE = original_triton


class TestStateDict:
    def test_save_load_roundtrip(self):
        model = make_model()
        opt = GrokAdamW(model.parameters(), lr=1e-3)

        for _ in range(5):
            train_step(model, opt)

        sd = opt.state_dict()
        assert len(sd["state"]) > 0, "state_dict should contain optimizer states"

        saved_exp_avg = None
        for p in model.parameters():
            s = opt.state.get(p, {})
            if "exp_avg" in s:
                saved_exp_avg = s["exp_avg"].clone()
                break
        assert saved_exp_avg is not None, "Should have at least one exp_avg state"

        opt_new = GrokAdamW(model.parameters(), lr=1e-3)
        opt_new.load_state_dict(sd)

        loaded_exp_avg = None
        for p in model.parameters():
            s = opt_new.state.get(p, {})
            if "exp_avg" in s:
                loaded_exp_avg = s["exp_avg"]
                break

        assert loaded_exp_avg is not None, "Loaded optimizer should have exp_avg state"
        assert torch.allclose(saved_exp_avg, loaded_exp_avg, atol=1e-7), (
            "Loaded state should match saved state"
        )

        for _ in range(5):
            train_step(model, opt_new)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_migration_cpu_to_gpu_states(self):
        model = make_model().cuda()
        opt_old = GrokAdamW(model.parameters(), cpu_offload=True, lr=1e-3)

        for _ in range(3):
            train_step(model, opt_old)

        sd = opt_old.state_dict()

        model_new = make_model().cuda()
        opt_new = GrokAdamW(model_new.parameters(), cpu_offload=False, lr=1e-3)
        opt_new.load_state_dict(sd)

        for p in model_new.parameters():
            state = opt_new.state.get(p, {})
            if state:
                for key in ("exp_avg", "exp_avg_sq", "grok_ema"):
                    assert state[key].dtype == torch.float32
                    assert state[key].device == p.device


class TestEdgeCases:
    def test_zero_element_params(self):
        model = nn.Sequential(nn.Linear(10, 0), nn.Linear(0, 5))
        has_params = [p for p in model.parameters() if p.numel() > 0]
        if not has_params:
            pytest.skip("No trainable params in zero-element model")
        opt = GrokAdamW(model.parameters())
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_no_grokking_signal(self):
        model = make_model()
        opt = GrokAdamW(model.parameters(), grokking_signal_fns=None)
        for _ in range(3):
            train_step(model, opt)

    def test_gamma_zero(self):
        model = make_model()
        opt = GrokAdamW(model.parameters(), gamma=0.0)
        for _ in range(3):
            train_step(model, opt)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cpu_offload_true_states_on_cpu(self):
        model = make_model().cuda()
        opt = GrokAdamW(model.parameters(), cpu_offload=True)
        for _ in range(3):
            train_step(model, opt)
        for p in model.parameters():
            state = opt.state.get(p, {})
            if state:
                for key in ("exp_avg", "exp_avg_sq", "grok_ema"):
                    assert state[key].device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_states_stay_on_gpu(self):
        model = make_model().cuda()
        opt = GrokAdamW(model.parameters(), cpu_offload=False)
        for _ in range(3):
            train_step(model, opt)
        for p in model.parameters():
            state = opt.state.get(p, {})
            if state:
                for key in ("exp_avg", "exp_avg_sq", "grok_ema"):
                    assert state[key].device.type == "cuda"
                    assert state[key].dtype == torch.float32

    def test_multiple_param_groups(self):
        model = make_model()
        params = list(model.parameters())
        mid = len(params) // 2
        opt = GrokAdamW(
            [
                {"params": params[:mid], "lr": 1e-3, "gamma": 0.1},
                {"params": params[mid:], "lr": 5e-4, "gamma": 0.0},
            ]
        )
        for _ in range(5):
            train_step(model, opt)


class TestFallback:
    def test_non_cuda_pytorch_fallback(self):
        model = make_model()
        opt = GrokAdamW(model.parameters(), cpu_offload=False)
        for _ in range(3):
            train_step(model, opt)

    def test_grokking_signal_serialization(self):
        signal_fn = lambda: 0.5
        model = make_model()
        opt = GrokAdamW(model.parameters(), grokking_signal_fns=[signal_fn])

        for _ in range(3):
            train_step(model, opt)

        sd = opt.state_dict()
        for group in sd["param_groups"]:
            assert group["grokking_signal_fns"] is None

        opt_new = GrokAdamW(model.parameters(), grokking_signal_fns=[signal_fn])
        opt_new.load_state_dict(sd)
        assert opt_new.param_groups[0]["grokking_signal_fns"] is not None

    def test_setstate_migration(self):
        model = make_model()
        opt = GrokAdamW(model.parameters())
        raw_state = opt.__getstate__()
        raw_state["param_groups"][0].pop("cpu_offload", None)
        opt.__setstate__(raw_state)
        for group in opt.param_groups:
            assert "cpu_offload" in group
            assert group["cpu_offload"] is False
