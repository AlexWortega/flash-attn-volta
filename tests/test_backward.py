"""Backward-pass correctness suite for flash_attn_volta.

Three tests, each described in the task brief:

1. test_backward_correctness_finite_diff
   Per (shape, causal) combo: compare dQ/dK/dV from `flash_attn` (autograd)
   against fp64 SDPA + autograd. Threshold: max-abs error < 2e-2 (looser than
   forward — backward has more accumulation rounds in fp16).

2. test_backward_gradcheck_subset
   Tiny (B=1, S=16, H=2, D=32) shape, gradcheck-style structural check.
   Because the kernel is fp16-only, plain `torch.autograd.gradcheck` would be
   swamped by the fp16 round-trip. We instead run a "manual gradcheck": pick
   one scalar coordinate of Q/K/V, perturb by ±h in fp32, run forward in fp32
   reference (`F.scaled_dot_product_attention`), and compare the central
   finite difference to the analytical gradient from our kernel. This catches
   the structural bugs gradcheck is meant to catch (wrong dtype propagation,
   missing terms, broken chain rule) without needing fp32 in the kernel.

3. test_backward_causal_no_leak
   Under causal mask, dQ[i] must not depend on K[j] / V[j] for j > i.
   Perturb K[j] for late j, check dQ for early rows i is bit-identical.
"""
from __future__ import annotations
import os, sys, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from flash_attn_volta import flash_attn


# Same shape matrix as forward correctness; one extra tiny shape for D=32.
SHAPES = [
    (2, 1024, 8, 64),
    (1, 2048, 16, 128),
    (4, 512, 4, 32),
    (1, 64, 2, 64),     # tiny — primary debug shape
]
TOL = 2e-2  # max-abs vs fp64 ref


def _ref_grads(q, k, v, causal, do):
    """fp64 SDPA + autograd reference. Returns (out, dq, dk, dv) all in (B,N,H,D)."""
    qf = q.detach().permute(0, 2, 1, 3).double().requires_grad_()
    kf = k.detach().permute(0, 2, 1, 3).double().requires_grad_()
    vf = v.detach().permute(0, 2, 1, 3).double().requires_grad_()
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal)
    do_ref = do.detach().permute(0, 2, 1, 3).double()
    out.backward(do_ref)
    return (out.permute(0, 2, 1, 3),
            qf.grad.permute(0, 2, 1, 3),
            kf.grad.permute(0, 2, 1, 3),
            vf.grad.permute(0, 2, 1, 3))


def test_backward_correctness_finite_diff():
    """Compare dQ/dK/dV against fp64 SDPA reference."""
    torch.manual_seed(0)
    overall_pass = True
    rows = []
    print("\n[1] backward correctness vs fp64 SDPA reference")
    print(f"{'shape':<22} {'causal':<6} {'fwd':>10} {'dq':>10} {'dk':>10} {'dv':>10} verdict")
    for B, N, H, D in SHAPES:
        for causal in (False, True):
            q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
            k = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
            v = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
            do = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")

            # fp64 reference
            out_ref, dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, causal, do)

            # ours
            out = flash_attn(q, k, v, causal=causal)
            out.backward(do)

            err_fwd = (out.float() - out_ref.float()).abs().max().item()
            err_dq = (q.grad.float() - dq_ref.float()).abs().max().item()
            err_dk = (k.grad.float() - dk_ref.float()).abs().max().item()
            err_dv = (v.grad.float() - dv_ref.float()).abs().max().item()
            worst = max(err_dq, err_dk, err_dv)
            verdict = "pass" if worst < TOL else "FAIL"
            overall_pass &= (verdict == "pass")
            print(f"({B},{N},{H},{D})    {str(causal):<5} "
                  f"{err_fwd:.2e} {err_dq:.2e} {err_dk:.2e} {err_dv:.2e} {verdict}")
            rows.append({"B": B, "N": N, "H": H, "D": D, "causal": causal,
                         "err_fwd": err_fwd, "err_dq": err_dq, "err_dk": err_dk,
                         "err_dv": err_dv, "verdict": verdict})
    assert overall_pass, "backward correctness FAIL — see table above"
    return rows


def test_backward_gradcheck_subset():
    """Manual finite-difference gradcheck on a tiny shape.

    Verifies: the chain rule wiring, the cast paths (fp16 ↔ fp32 inside
    autograd), and the dQ/dK/dV terms each independently. We don't use
    `torch.autograd.gradcheck` directly because the kernel is fp16 only and
    torch's gradcheck would compare an fp16 finite-difference against an fp16
    analytical gradient — both equally noisy. Instead we compute the
    finite-difference in fp32 (using the SDPA reference), and compare to our
    kernel's fp16 analytical gradient.
    """
    torch.manual_seed(0)
    B, N, H, D = 1, 16, 2, 32
    causal = True
    sm_scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)

    # Use a fixed grad_target so gradients are deterministic.
    torch.manual_seed(1)
    grad_target = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")

    out = flash_attn(q, k, v, causal=causal, sm_scale=sm_scale)
    loss = (out * grad_target).sum()
    loss.backward()

    dq_kernel = q.grad.detach().clone().float()
    dk_kernel = k.grad.detach().clone().float()
    dv_kernel = v.grad.detach().clone().float()

    # fp32 reference forward — we'll perturb individual coordinates of Q/K/V
    # and check the central difference against the kernel gradient.
    q32 = q.detach().float()
    k32 = k.detach().float()
    v32 = v.detach().float()
    g32 = grad_target.float()

    def loss_at(qx, kx, vx):
        # SDPA in torch 2.0.1 has no `scale` kwarg; default is 1/sqrt(D),
        # which is exactly what we pass into flash_attn (sm_scale=1/sqrt(D)).
        qp = qx.permute(0, 2, 1, 3).contiguous()
        kp = kx.permute(0, 2, 1, 3).contiguous()
        vp = vx.permute(0, 2, 1, 3).contiguous()
        out = F.scaled_dot_product_attention(qp, kp, vp, is_causal=causal)
        out = out.permute(0, 2, 1, 3).contiguous()
        return (out * g32).sum().item()

    h = 1e-3  # fp32 step
    n_check = 8  # number of random coords to check per tensor
    rng = torch.Generator(device="cuda").manual_seed(42)

    def check_one(name, x32, grad_kernel):
        """Pick a few random coordinates, compare central finite-diff to kernel grad."""
        flat = x32.numel()
        idx_list = torch.randperm(flat, generator=rng, device="cuda")[:n_check].tolist()
        worst_rel = 0.0
        worst_abs = 0.0
        for idx in idx_list:
            x_plus = x32.clone(); x_plus.view(-1)[idx] += h
            x_minus = x32.clone(); x_minus.view(-1)[idx] -= h
            if name == "q":
                lp = loss_at(x_plus, k32, v32); lm = loss_at(x_minus, k32, v32)
            elif name == "k":
                lp = loss_at(q32, x_plus, v32); lm = loss_at(q32, x_minus, v32)
            else:
                lp = loss_at(q32, k32, x_plus); lm = loss_at(q32, k32, x_minus)
            fd = (lp - lm) / (2 * h)
            an = grad_kernel.view(-1)[idx].item()
            err_abs = abs(fd - an)
            err_rel = err_abs / max(abs(fd), 1e-3)
            worst_abs = max(worst_abs, err_abs)
            worst_rel = max(worst_rel, err_rel)
        return worst_abs, worst_rel

    qa, qr = check_one("q", q32, dq_kernel)
    ka, kr = check_one("k", k32, dk_kernel)
    va, vr = check_one("v", v32, dv_kernel)
    print(f"\n[2] manual fp32 gradcheck (n={n_check} coords each, h={h})")
    print(f"  dQ: max_abs={qa:.3e}  max_rel={qr:.3e}")
    print(f"  dK: max_abs={ka:.3e}  max_rel={kr:.3e}")
    print(f"  dV: max_abs={va:.3e}  max_rel={vr:.3e}")
    # Pass criterion: combined — coord passes if EITHER abs < TOL_ABS OR
    # rel < TOL_REL. The relative-only check is too tight for coords whose
    # analytical gradient is near zero (small denominator inflates rel even
    # when the absolute error is well within fp16 noise of ~few×1e-3).
    TOL_ABS = 5e-3
    TOL_REL = 5e-2
    worst_abs = max(qa, ka, va)
    worst_rel = max(qr, kr, vr)
    assert worst_abs < TOL_ABS or worst_rel < TOL_REL, \
        (f"gradcheck FAIL — neither budget met: "
         f"worst_abs={worst_abs:.3e} (>{TOL_ABS}) AND worst_rel={worst_rel:.3e} (>{TOL_REL})")
    return {"dq": (qa, qr), "dk": (ka, kr), "dv": (va, vr),
            "tol_abs": TOL_ABS, "tol_rel": TOL_REL}


def test_backward_causal_no_leak():
    """Causal: dQ[i] must not depend on K[j] / V[j] for j > i."""
    torch.manual_seed(0)
    B, N, H, D = 1, 64, 2, 64

    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")

    do = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")

    # Run 1
    k_a = k.detach().clone().requires_grad_()
    v_a = v.detach().clone().requires_grad_()
    q_a = q.detach().clone().requires_grad_()
    out_a = flash_attn(q_a, k_a, v_a, causal=True)
    out_a.backward(do)
    dq_a = q_a.grad.detach().clone()

    # Run 2: perturb K[j>=N//2] AND V[j>=N//2]
    k_p = k.detach().clone()
    v_p = v.detach().clone()
    j_pert = N // 2  # perturb everything from the midpoint onward
    k_p[:, j_pert:] += torch.randn_like(k_p[:, j_pert:]) * 0.5
    v_p[:, j_pert:] += torch.randn_like(v_p[:, j_pert:]) * 0.5
    k_p.requires_grad_(); v_p.requires_grad_()
    q_b = q.detach().clone().requires_grad_()
    out_b = flash_attn(q_b, k_p, v_p, causal=True)
    out_b.backward(do)
    dq_b = q_b.grad.detach().clone()

    # Early rows i < j_pert must have identical dQ.
    leak = (dq_a[:, :j_pert] - dq_b[:, :j_pert]).abs().max().item()
    # Late rows must differ (sanity — the perturbation actually had effect)
    late_diff = (dq_a[:, j_pert:] - dq_b[:, j_pert:]).abs().max().item()
    print(f"\n[3] causal-leak check (perturb K[{j_pert}:], V[{j_pert}:]):")
    print(f"  dQ[0:{j_pert}] max diff (should be 0) = {leak:.3e}")
    print(f"  dQ[{j_pert}:N]  max diff (should be > 0) = {late_diff:.3e}")
    assert leak == 0.0, f"causal leak: dQ for early rows changed when future K/V perturbed (max diff {leak:.3e})"
    assert late_diff > 1e-3, f"sanity FAIL: perturbation had no effect on late rows ({late_diff:.3e})"
    return {"early_diff": leak, "late_diff": late_diff}


def main():
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0))
    rows = test_backward_correctness_finite_diff()
    grad_info = test_backward_gradcheck_subset()
    leak_info = test_backward_causal_no_leak()
    print("\nALL BACKWARD TESTS: PASS")
    return {"rows": rows, "grad": grad_info, "leak": leak_info}


if __name__ == "__main__":
    main()
