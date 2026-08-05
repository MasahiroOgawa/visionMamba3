#!/usr/bin/env python3
"""Emit the Vision Mamba-3 ablation table and memory-vs-accuracy plot for
doc/attention/mamba3_attention.tex.

The grid is {2-dir, 4-dir, VSSD-gamma, VSSD-beta,gamma} x a 2x2 of
{2-D RoPE on/off} x {Mamba-3 rotary on/off}, at both sequence lengths, plus the
softmax and CNN baselines for context. The rotary-only cell is what lets the
rotary's own effect be separated from its interaction with 2-D RoPE: alone it helps
every operator, but on top of RoPE it is redundant for the scan operators.

Operator and encoding are separate fields rather than one concatenated label. That
is narrower in the table -- the concatenated form overflowed the text width once
VSSD-beta,gamma's rows existed, silently clipping the T=1025 memory column -- and it
lets the plot's colour/marker lookup read the encoding directly. Deriving it by
substring instead is a trap: "both" contains "rotary", and "+2-D RoPE +rotary"
contains "+2-D RoPE", so the answer depended on the order of the tests.

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
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "doc" / "attention"

# Encoding keys. These are the literal strings printed in the table's Enc. column
# and the keys for the plot's markers, so the two cannot disagree.
NONE, ROT, ROPE, BOTH = "---", "rotary", "2-D RoPE", "both"

ENC_MARKERS = {NONE: "^", ROT: "D", ROPE: "s", BOTH: "o"}
ENC_LABELS = {NONE: "neither encoding", ROT: "rotary only",
              ROPE: "2-D RoPE only", BOTH: "both"}
OP_COLOURS = {
    "2-dir SSD": "#4878CF",
    "4-dir SSD": "#E06C2B",
    r"VSSD-$\gamma$": "#3A9E5C",
    r"VSSD-$\beta,\gamma$": "#8E44AD",
}
OP_LABELS = {
    "2-dir SSD": "2-directional",
    "4-dir SSD": "4-directional",
    r"VSSD-$\gamma$": r"VSSD-$\gamma$",
    r"VSSD-$\beta,\gamma$": r"VSSD-$\beta,\gamma$",
}
BASE_COLOURS = {"CNN (ResNet)": "black", "Softmax attention": "0.6"}

# (operator, encoding, variant, result dir for T=65, result dir for T=1025)
# Only post-pairing-fix directories are listed for cells that enable both encodings.
ROWS = [
    ("CNN (ResNet)", NONE, "cnn", "table1_patch4", "table1_patch1"),
    ("Softmax attention", NONE, "vit_attn", "table1_patch4", "table1_patch1"),

    ("2-dir SSD", NONE, "vit_mamba3", "vm3_ablation_patch4_norope", "vm3_t1025_norope"),
    ("2-dir SSD", ROT, "vit_mamba3", "vm3_ablation_patch4_norope_rotary", "vm3_t1025_rotary"),
    ("2-dir SSD", ROPE, "vit_mamba3", "vm3_ropeonly_recheck", "vm3_t1025_rope"),
    ("2-dir SSD", BOTH, "vit_mamba3", "vm3_interleaved_vit_mamba3", "vm3_t1025_rope_rotary"),

    ("4-dir SSD", NONE, "vit_mamba3_4dir", "vm3_ablation_patch4_norope", "vm3_t1025_norope"),
    ("4-dir SSD", ROT, "vit_mamba3_4dir", "vm3_ablation_patch4_norope_rotary", "vm3_t1025_rotary"),
    ("4-dir SSD", ROPE, "vit_mamba3_4dir", "vm3_ropeonly_recheck", "vm3_t1025_rope"),
    ("4-dir SSD", BOTH, "vit_mamba3_4dir", "vm3_interleaved_vit_mamba3_4dir", "vm3_t1025_rope_rotary"),

    (r"VSSD-$\gamma$", NONE, "vit_mamba3_vssd", "vm3_ablation_patch4_norope", "vm3_t1025_norope"),
    (r"VSSD-$\gamma$", ROT, "vit_mamba3_vssd", "vm3_ablation_patch4_norope_rotary", "vm3_t1025_rotary"),
    (r"VSSD-$\gamma$", ROPE, "vit_mamba3_vssd", "vm3_ropeonly_recheck", "vm3_t1025_rope"),
    (r"VSSD-$\gamma$", BOTH, "vit_mamba3_vssd", "vm3_interleaved_vit_mamba3_vssd", "vm3_t1025_rope_rotary"),

    # VSSD-beta,gamma: its own dirs, because it was added after the first sweep and
    # is not parameter-matched with the rows above (2.93 M vs 2.71 M).
    (r"VSSD-$\beta,\gamma$", NONE, "vit_mamba3_vssd_bg", "vm3_vssdbg_norope", "vm3_t1025_vssdbg_norope"),
    (r"VSSD-$\beta,\gamma$", ROT, "vit_mamba3_vssd_bg", "vm3_vssdbg_norope_rotary", "vm3_t1025_vssdbg_rotary"),
    (r"VSSD-$\beta,\gamma$", ROPE, "vit_mamba3_vssd_bg", "vm3_vssdbg_rope", "vm3_t1025_vssdbg_rope"),
    (r"VSSD-$\beta,\gamma$", BOTH, "vit_mamba3_vssd_bg", "vm3_vssdbg_rope_rotary", "vm3_t1025_vssdbg_rope_rotary"),
]

# Point labels used to be drawn inline next to each marker, but with a dozen-plus
# points per panel the labels overlapped each other and the markers (see git
# history). Colour and marker already encode operator and positional encoding, so a
# legend replaces them entirely -- no in-plot text left to overlap. Sized so the
# legend/tick/axis text reads at >= \small (10pt), matching the smallest font size
# already used elsewhere in mamba3_attention.tex (the ablation table), once scaled
# by \linewidth / FIG_W.
FIG_W = 360 / 72.27  # pt -> in; matches mamba3_attention.tex's \textwidth exactly


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
        r"\begin{tabular}{|l|l|c|ccc|ccc|}",
        r"\hline",
        r" & & & \multicolumn{3}{c|}{$T{=}65$} & \multicolumn{3}{c|}{$T{=}1025$} \\",
        r"Operator & Enc. & Par.\,(M) & Acc.\,(\%) & Lat.\,(ms) & Mem"
        r" & Acc.\,(\%) & Lat.\,(ms) & Mem \\ \hline",
    ]
    for op, enc, a, b in rows:
        params = (a or b or {}).get("params")
        lines.append(
            f"{op} & {enc} & {fmt(params, '.2f')} "
            f"& {fmt(a and a['acc'], '.2f')} & {fmt(a and a['lat'], '.1f')} & {fmt(a and a['mem'], '.0f')} "
            f"& {fmt(b and b['acc'], '.2f')} & {fmt(b and b['lat'], '.1f')} & {fmt(b and b['mem'], '.0f')} \\\\ \\hline"
        )
    lines += [r"\end{tabular}"]
    (OUT_DIR / "ablation_table.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR / 'ablation_table.tex'}")


def _legend_handles() -> list[Line2D]:
    def mark(marker, face, size, label):
        return Line2D([0], [0], marker=marker, linestyle="", markerfacecolor=face,
                      markeredgecolor="black", markersize=size, label=label)

    return (
        [mark("o", OP_COLOURS[k], 8, OP_LABELS[k]) for k in OP_COLOURS]
        + [mark(ENC_MARKERS[k], "0.75", 8, ENC_LABELS[k]) for k in (NONE, ROT, ROPE, BOTH)]
        + [mark("*", BASE_COLOURS[k], 11, k) for k in BASE_COLOURS]
    )


def write_plot(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.6))
    for ax, idx, title in ((axes[0], 0, r"$T=65$"), (axes[1], 1, r"$T=1025$")):
        for op, enc, a, b in rows:
            d = (a, b)[idx]
            if d is None:
                continue
            base = op in BASE_COLOURS
            ax.scatter(d["mem"], d["acc"], s=100 if base else 80,
                       marker="*" if base else ENC_MARKERS[enc],
                       color=BASE_COLOURS[op] if base else OP_COLOURS[op],
                       edgecolors="black", linewidths=0.6, zorder=3)
        ax.set_xlabel("Peak memory (MiB)", fontsize=10)
        ax.set_ylabel("CIFAR-10 accuracy (%)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)

    # ncol=2 (not 4): a 4-column legend is wider than the two subplots combined,
    # so bbox_inches="tight" grows the saved canvas to fit it -- silently breaking
    # the FIG_W == \linewidth point-for-point match every font size above relies
    # on. 2 columns fits inside the subplots' own width, so nothing overflows and
    # bbox_inches=None (below) saves at exactly figsize, no surprise rescale.
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.46, wspace=0.45)
    out = OUT_DIR / "ablation_mem_acc.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    rows = [(op, enc, cell(d65, v), cell(d1025, v)) for op, enc, v, d65, d1025 in ROWS]
    missing = [f"{op} {enc}" for op, enc, a, b in rows if a is None or b is None]
    write_table(rows)
    write_plot(rows)
    if missing:
        print("\nCells still unmeasured (printed as '--'):")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
