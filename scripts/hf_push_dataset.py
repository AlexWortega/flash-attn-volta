"""Push flash-attn-volta as an HF Hub dataset repo (code + benchmarks).

Reads HF_TOKEN / HF_USER from $HOME/.claude/skills/ml-intern/.env if not set
in env. Creates the dataset repo (idempotent) and uploads the run dir
contents -- omitting noisy / large files (pycache, results/*.json kept,
prompt.txt, session.* removed, etc.).
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

# bootstrap env from skill .env
SKILL_ENV = Path.home() / ".claude/skills/ml-intern/.env"
if SKILL_ENV.exists():
    for line in SKILL_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)

token = os.environ.get("HF_TOKEN") or ""
if not token:
    print("ERROR: HF_TOKEN not set in env or skill .env", file=sys.stderr)
    sys.exit(3)

from huggingface_hub import HfApi, create_repo, upload_folder

api = HfApi(token=token)
user = os.environ.get("HF_USER") or api.whoami()["name"]

slug = "flash-attn-volta-triton"
from datetime import datetime
stamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
repo_id = f"{user}/ml-intern-{slug}-{stamp}"

print(f"creating dataset repo: {repo_id}")
create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True,
            private=False)

# Stage: copy interesting files
run = Path(__file__).resolve().parent.parent
ignore = {
    "__pycache__", ".git", ".pytest_cache",
    "session.jsonl", "session.err",
    "run.sh", "prompt.txt",
}
def keep(rel: Path) -> bool:
    parts = set(rel.parts)
    if parts & ignore:
        return False
    if rel.suffix in {".pyc"}:
        return False
    return True

import tempfile, shutil
with tempfile.TemporaryDirectory() as stage:
    stage = Path(stage)
    n = 0
    for p in run.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(run)
        if not keep(rel):
            continue
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        n += 1
    # Write a dataset card
    card = stage / "README.md"
    card.write_text(_README := f"""---
license: apache-2.0
language: en
tags:
- flash-attention
- volta
- triton
- attention
- cuda
- v100
- sm70
size_categories:
- n<1K
pretty_name: flash-attn-volta (Triton)
---

# flash-attn-volta — Triton FlashAttention forward kernel for NVIDIA V100 (SM 7.0)

This dataset repo bundles **code + benchmarks** (no model weights) for a
single-file Triton port of FlashAttention-1 forward that runs on Volta /
Tesla V100, working around several Triton-on-Volta compiler bugs.

- API: `from flash_attn_volta import flash_attn_forward`
- Input: `(B, N, H, D)` fp16, `D ∈ {{32, 64, 128}}`, optional causal mask
- Output: `(B, N, H, D)` fp16, fp32 accumulation internally
- Hardware: tested on Tesla V100-SXM2 32GB (compute capability 7.0)

## Why this exists

`flash-attn >= 2.0` dropped Volta. `flash-attn 1.x` is unmaintained and
won't build cleanly against modern toolchains. Triton's own bundled
`triton.ops.flash_attention._fwd_kernel` asserts `head_dim == 64` and
gates on `capability[0] >= 8`. This repo carries a kernel that just
**works** on a V100.

## Quick reproduce

```bash
pip3 install --user triton==2.3.0 torch==2.0.1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_verify.sh
```

See [`VERIFY.md`](VERIFY.md) for the full pass/fail table and
[`RESULTS.md`](RESULTS.md) for the engineering writeup.

## Key numbers (V100-SXM2 32GB, Triton 2.3.0)

| seq  | head_dim | causal | FA TFLOP/s | speedup vs torch eager |
|------|----------|--------|------------|------------------------|
| 1024 | 64       | False  | 10.7       | 0.90x                  |
| 2048 | 64       | False  | 27.9       | **2.28x**              |
| 2048 | 64       | True   | 41.4       | **5.24x**              |
| 4096 | 64       | False  | 38.3       | **2.92x**              |
| 4096 | 128      | False  | 30.0       | 1.32x                  |

Memory at seq=4096, h=16, d=64: **fa 42 MB vs eager 1082 MB** (~26× less).

## Triton-on-Volta bugs encountered (and worked around)

1. Triton 2.0 `tt.reduce` mma-layout mismatch after `tl.dot`.
2. Triton 2.1 mma → mma cast assertion failure on `p.to(fp16)`.
3. Triton 2.3 `tl.dot` with K-dim ≠ 64 either miscompiles silently or
   raises `IndexError: map::at` during PTX gen.

Workarounds:
- All `tl.dot` calls use K-dim = 64.
- `head_dim < 64` → padded to 64 in the Python wrapper.
- `head_dim == 128` → split into two BLOCK_D=64 halves inside the kernel.

## License

Apache-2.0.
""")
    print(f"staged {n} files; uploading to {repo_id} ...")
    upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="flash-attn-volta v0.1: working Triton FA fwd on V100 SM 7.0",
    )

url = f"https://huggingface.co/datasets/{repo_id}"
print("UPLOADED:", url)
Path(run / "PUBLISHED.md").write_text(
    f"# Published\n\nHF Hub: {url}\n\nUploaded at {stamp} UTC.\n"
)
