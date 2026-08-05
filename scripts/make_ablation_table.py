#!/usr/bin/env python3
"""Emit the Vision Mamba-3 ablation table and memory-vs-accuracy plot for
doc/attention/mamba3_attention.tex.

The 3x3 grid is {2-dir, 4-dir, VSSD-gamma} x {no positional encoding, 2-D RoPE,
2-D RoPE + Mamba-3 rotary}, at both sequence lengths, plus the softmax and CNN
baselines for context.

Numbers are read from the run directories, never typed in, so the table cannot
drift from what was measured. A cell with no run prints as "--" rather than being
filled in or silently dropped -- several cells were measured before the RoPE2D
pairing fix and are deliberately *not* reused, since with both encodings enabled
the mismatched rotation planes destroyed the relative-position property (see
"Measured: Positional Encoding and Scan Directions" in mamba3_attention.tex, and
`rotate_pairs` in visionmamba3/rope2d.py).

  uv run python scripts/make_ablation_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "doc" / "attention"

# (row label, variant, result dir for T=65, result dir for T=1025)
# Only post-pairing-fix directories are listed for cells that enable both encodings.
ROWS = [
    ("CNN (ResNet)", "cnn", "table1_patch4", "table1_patch1"),
    ("Softmax attention", "vit_attn", "table1_patch4", "table1_patch1"),
    ("2-dir SSD", "vit_mamba3", "vm3_ablation_patch4_norope", "vm3_t1025_norope"),
    ("2-dir SSD, +2-D RoPE", "vit_mamba3", "vm3_ropeonly_recheck", "vm3_t1025_rope"),
    ("2-dir SSD, +2-D RoPE +rotary", "vit_mamba3", "vm3_interleaved_vit_mamba3", "vm3_t1025_rope_rotary"),
    ("4-dir SSD", "vit_mamba3_4dir", "vm3_ablation_patch4_norope", "vm3_t1025_norope"),
    ("4-dir SSD, +2-D RoPE", "vit_mamba3_4dir", "vm3_ropeonly_recheck", "vm3_t1025_rope"),
    ("4-dir SSD, +2-D RoPE +rotary", "vit_mamba3_4dir", "vm3_interleaved_vit_mamba3_4dir", "vm3_t1025_rope_rotary"),
    (r"VSSD-$\gamma$", "vit_mamba3_vssd", "vm3_ablation_patch4_norope", "vm3_t1025_norope"),
    (r"VSSD-$\gamma$, +2-D RoPE", "vit_mamba3_vssd", "vm3_ropeonly_recheck", "vm3_t1025_rope"),
    (r"VSSD-$\gamma$, +2-D RoPE +rotary", "vit_mamba3_vssd", "vm3_interleaved_vit_mamba3_vssd", "vm3_t1025_rope_rotary"),
]


def cell(run_dir: str, variant: str) -> dict | None:
    p = REPO / "result" / run_dir / "results.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text())["variants"].get(variant)
    if v is None:
        return None
    return {
        "params": v["params"] / 1e6,
        "acc": v["best_test_acc"] * 100.0,
        "lat": v["efficiency"]["latency_ms"],
        "mem": v["efficiency"]["peak_mib"],
    }


def fmt(x, spec: str) -> str:
    return "--" if x is None else format(x, spec)


def write_table(rows) -> None:
    # \hline style, matching the tables already in mamba3_attention.tex, so the
    # doc needs no extra package.
    lines = [
        r"\begin{tabular}{|l|c|ccc|ccc|}",
        r"\hline",
        r" & & \multicolumn{3}{c|}{$T{=}65$} & \multicolumn{3}{c|}{$T{=}1025$} \\",
        r"Operator & Params (M) & Acc.\,(\%) & Lat.\,(ms) & Mem (MiB)"
        r" & Acc.\,(\%) & Lat.\,(ms) & Mem (MiB) \\ \hline",
    ]
    for label, a, b in rows:
        params = (a or b or {}).get("params")
        lines.append(
            f"{label} & {fmt(params, '.2f')} "
            f"& {fmt(a and a['acc'], '.2f')} & {fmt(a and a['lat'], '.1f')} & {fmt(a and a['mem'], '.0f')} "
            f"& {fmt(b and b['acc'], '.2f')} & {fmt(b and b['lat'], '.1f')} & {fmt(b and b['mem'], '.0f')} \\\\ \\hline"
        )
    lines += [r"\end{tabular}"]
    (OUT_DIR / "ablation_table.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR / 'ablation_table.tex'}")


def write_plot(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, idx, title in ((axes[0], 0, r"$T=65$"), (axes[1], 1, r"$T=1025$")):
        for label, a, b in rows:
            d = (a, b)[idx]
            if d is None:
                continue
            base = "CNN" in label or "Softmax" in label
            fam = "2-dir" if "2-dir" in label else "4-dir" if "4-dir" in label else "VSSD"
            colour = {"2-dir": "#4878CF", "4-dir": "#E06C2B", "VSSD": "#3A9E5C"}.get(fam, "0.35")
            # Marker encodes the positional encoding: none / RoPE / RoPE+rotary.
            mk = "o" if "rotary" in label else ("s" if "RoPE" in label else "^")
            ax.scatter(d["mem"], d["acc"], s=90, marker="*" if base else mk,
                       color="0.35" if base else colour,
                       edgecolors="black", linewidths=0.5, zorder=3)
            ax.annotate(label, (d["mem"], d["acc"]), textcoords="offset points",
                        xytext=(6, 3), fontsize=7)
        ax.set_xlabel("Peak memory (MiB)")
        ax.set_ylabel("CIFAR-10 accuracy (%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "ablation_mem_acc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    rows = [(label, cell(d65, v), cell(d1025, v)) for label, v, d65, d1025 in ROWS]
    missing = [label for label, a, b in rows if a is None or b is None]
    write_table(rows)
    write_plot(rows)
    if missing:
        print("\nCells still unmeasured (printed as '--'):")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
