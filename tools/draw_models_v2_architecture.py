from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch


OUT_DIR = Path(__file__).resolve().parents[1] / "images"
PNG_PATH = OUT_DIR / "v2.png"
SVG_PATH = OUT_DIR / "v2.svg"


def rounded(ax, xy, w, h, text="", fc="#ffffff", ec="#1b1b1b", lw=1.6,
            fontsize=12, weight="normal", radius=0.08, ls="-", z=2,
            color="#111111", ha="center", va="center"):
    box = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            zorder=z + 1,
        )
    return box


def arrow(ax, start, end, lw=1.8, color="#111111", style="-|>", mutation=18,
          connectionstyle="arc3,rad=0", z=5):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=connectionstyle,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def encoder_wedge(ax, x, y, w, h, fc, label, modules):
    main = patches.Polygon(
        [(x, y), (x + w * 0.84, y + h * 0.10), (x + w, y + h * 0.50),
         (x + w * 0.84, y + h * 0.90), (x, y + h)],
        closed=True,
        facecolor=fc,
        edgecolor="#1c1c1c",
        linewidth=1.7,
        zorder=2,
    )
    ax.add_patch(main)
    dash = patches.Polygon(
        [(x + w * 0.10, y + h * 0.08), (x + w * 1.06, y + h * 0.18),
         (x + w * 1.19, y + h * 0.54), (x + w * 1.02, y + h * 0.88),
         (x + w * 0.10, y + h * 1.02)],
        closed=True,
        fill=False,
        edgecolor="#242424",
        linewidth=1.5,
        linestyle=(0, (4, 4)),
        zorder=1,
    )
    ax.add_patch(dash)

    n = len(modules)
    gap = 0.07
    slot_w = (w * 0.70 - gap * (n - 1)) / n
    start_x = x + w * 0.10
    for i, name in enumerate(modules):
        sx = start_x + i * (slot_w + gap)
        rounded(
            ax,
            (sx, y + h * 0.18),
            slot_w,
            h * 0.64,
            "",
            fc="#f9fbff",
            ec="#353535",
            lw=1.0,
            fontsize=8.0 if n > 4 else 9.0,
            weight="bold",
            radius=0.035,
        )
        ax.text(
            sx + slot_w / 2,
            y + h * 0.50,
            name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.0 if n > 4 else 9.0,
            fontweight="bold",
            zorder=5,
        )
    ax.text(x + w * 0.44, y - 0.28, label, ha="center", va="top",
            fontsize=13, fontweight="bold")


def token_bar(ax, x, y, n=6, first="#b8d6fb", label_left="", label_right=""):
    cell_w, cell_h = 0.22, 0.24
    for i in range(n):
        fc = first if i == 0 else "#ffffff"
        ax.add_patch(
            patches.Rectangle(
                (x + i * cell_w, y),
                cell_w,
                cell_h,
                facecolor=fc,
                edgecolor="#111111",
                linewidth=0.9,
                zorder=3,
            )
        )
    if label_left:
        ax.text(x + cell_w * 0.45, y - 0.13, label_left,
                ha="center", va="top", fontsize=8.5)
    if label_right:
        ax.text(x + cell_w * (n - 0.6), y - 0.13, label_right,
                ha="center", va="top", fontsize=8.5)
    return (x, y, n * cell_w, cell_h)


def circle_label(ax, x, y, r, text, fc, fontsize=10):
    ax.add_patch(
        patches.Circle((x, y), r, facecolor=fc, edgecolor="#111111",
                       linewidth=1.0, zorder=4)
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", zorder=5)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    })
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    blue = "#ddecff"
    yellow = "#fff1b8"
    green = "#dff6e6"
    pink = "#ffe1df"
    lavender = "#eee9ff"
    gold = "#ffe38a"
    red = "#f35e63"

    # Inputs.
    ax.text(0.62, 8.46, "Inputs", fontsize=13, fontweight="bold", ha="left")
    rounded(ax, (0.36, 5.55), 1.9, 1.55, fc="#f6fbff", lw=1.0)
    ax.add_patch(patches.Polygon(
        [(0.50, 5.85), (0.88, 6.50), (1.18, 5.95), (1.52, 6.62),
         (2.05, 5.85)],
        facecolor="#bcdcff",
        edgecolor="#111111",
        linewidth=1.0,
        zorder=3,
    ))
    ax.add_patch(patches.Circle((0.78, 6.78), 0.13, fc="#ffd45a",
                                ec="#111111", lw=0.9, zorder=4))
    ax.text(1.31, 5.25, "Image I", fontsize=12, fontweight="bold",
            fontstyle="italic", ha="center")

    rounded(
        ax,
        (0.36, 1.85),
        1.9,
        1.35,
        "Text claim / caption\ninput_ids + mask",
        fc="#ffffff",
        lw=1.0,
        fontsize=9.5,
    )
    ax.text(1.31, 1.50, "Text T", fontsize=12, fontweight="bold",
            fontstyle="italic", ha="center")

    # Encoders.
    ax.text(2.48, 8.22, "Image Encoder E_v", fontsize=12,
            fontweight="bold", ha="left")
    encoder_wedge(
        ax,
        2.48,
        5.55,
        1.52,
        1.55,
        blue,
        "Image Encoder E_v",
        ["ResNet50", "1x1 Conv", "LN"],
    )
    encoder_wedge(
        ax,
        2.48,
        1.85,
        1.52,
        1.55,
        yellow,
        "Text Encoder E_t",
        ["Emb+Pos", "N-gram", "BiLSTM", "BiGRU", "TrEnc", "Pool"],
    )
    arrow(ax, (2.26, 6.32), (2.48, 6.32))
    arrow(ax, (2.26, 2.52), (2.48, 2.52))

    # Reasoning panel.
    rounded(ax, (4.05, 1.18), 5.85, 6.95, fc="#f9fbff", ec="#111111",
            lw=1.2, radius=0.07)
    rounded(ax, (5.12, 7.55), 3.45, 0.35, "RNN_HYBRID Reasoning",
            fc="#ffffff", lw=1.0, fontsize=12, weight="bold", radius=0.04)
    ax.text(4.45, 7.16, "image patch tokens", fontsize=8.5, ha="left")
    rounded(ax, (4.42, 6.48), 2.00, 0.70, fc="#ffffff", ec="#111111",
            lw=1.0, ls=(0, (3, 3)), radius=0.02)
    token_bar(ax, 4.58, 6.72, first="#b8d6fb", label_left="v_pat",
              label_right="mean -> v_pool")

    ax.text(4.45, 3.34, "text tokens", fontsize=8.5, ha="left")
    rounded(ax, (4.42, 2.66), 2.00, 0.70, fc="#ffffff", ec="#111111",
            lw=1.0, ls=(0, (3, 3)), radius=0.02)
    token_bar(ax, 4.58, 2.90, first="#ffe17c", label_left="t_tok",
              label_right="pool -> t_cls")

    rounded(ax, (5.18, 4.10), 1.65, 1.05, "Cross-Modal\nFusion x 4",
            fc="#ffffff", lw=1.2, fontsize=11, weight="bold", radius=0.06)
    rounded(ax, (5.35, 3.28), 1.28, 0.56,
            "Text -> Image MHA\nImage -> Text MHA\nPreNorm + FFN",
            fc="#ffffff", lw=0.9, fontsize=7.3, ls=(0, (3, 3)), radius=0.03)
    arrow(ax, (5.44, 6.48), (5.74, 5.15))
    arrow(ax, (5.44, 3.36), (5.74, 4.10))
    arrow(ax, (6.83, 4.62), (7.32, 4.62))

    rounded(ax, (7.32, 3.62), 1.58, 1.95, fc="#ffffff", ec="#111111",
            lw=1.0, ls=(0, (3, 3)), radius=0.025)
    ax.text(7.48, 5.74, "fused multimodal tokens", fontsize=8.4, ha="left")
    token_bar(ax, 7.52, 5.05, first="#b99aff", label_left="m_cls",
              label_right="m_tok")
    token_bar(ax, 7.52, 4.50, first="#8bebaa", label_left="f_cls",
              label_right="f_tok")

    rounded(ax, (7.47, 3.50), 0.70, 0.55, "Mean", fc=lavender, lw=1.0,
            fontsize=9.5, radius=0.05)
    rounded(ax, (8.28, 3.50), 0.75, 0.55, "Concat\n[t, v]", fc=lavender,
            lw=1.0, fontsize=8.5, radius=0.05)
    rounded(ax, (9.10, 3.50), 0.58, 0.55, "Fused", fc=pink, lw=1.0,
            fontsize=8.5, radius=0.05)
    arrow(ax, (7.86, 4.50), (7.82, 4.05))
    arrow(ax, (8.62, 4.50), (8.64, 4.05))
    arrow(ax, (7.86, 3.50), (8.28, 3.78))
    arrow(ax, (8.64, 3.50), (8.28, 3.78))
    arrow(ax, (9.03, 3.78), (9.10, 3.78))

    rounded(ax, (7.60, 6.28), 1.44, 0.62, "BBox Head", fc=pink, lw=1.1,
            fontsize=10.5, weight="bold", radius=0.06)
    ax.text(7.84, 7.06, "L_bbox + L_giou", fontsize=9.0,
            fontstyle="italic", ha="center")
    ax.text(7.96, 5.98, "(x, y, w, h)", fontsize=8.5, ha="center")
    arrow(ax, (7.64, 5.57), (8.20, 6.28))

    rounded(ax, (6.60, 1.68), 2.25, 0.55,
            "Projection MLPs + symmetric InfoNCE",
            fc="#ffffff", lw=0.9, ls=(0, (3, 3)), fontsize=8.5, radius=0.04)
    ax.text(7.73, 2.35, "L_MAC", fontsize=10.5, fontstyle="italic",
            color="#4d006b", ha="center")
    arrow(ax, (5.22, 6.48), (6.75, 2.23), lw=1.2,
          connectionstyle="arc3,rad=-0.25")
    arrow(ax, (5.20, 2.66), (6.75, 1.95), lw=1.2,
          connectionstyle="arc3,rad=0.12")

    arrow(ax, (4.00, 6.32), (4.05, 6.32))
    arrow(ax, (4.00, 2.52), (4.05, 2.52))

    # Aggregator wedge.
    x, y, w, h = 10.35, 3.38, 1.05, 2.25
    aggr = patches.Polygon(
        [(x, y), (x + w, y + 0.22), (x + w, y + h - 0.22), (x, y + h)],
        closed=True,
        facecolor=green,
        edgecolor="#111111",
        linewidth=1.3,
        zorder=2,
    )
    ax.add_patch(aggr)
    rounded(ax, (10.52, 4.86), 0.68, 0.42, "Pool", fc="#ffffff", lw=0.8,
            fontsize=8.2, radius=0.035)
    rounded(ax, (10.52, 4.25), 0.68, 0.42, "Concat", fc="#ffffff", lw=0.8,
            fontsize=8.2, radius=0.035)
    rounded(ax, (10.52, 3.64), 0.68, 0.42, "Heads", fc="#ffffff", lw=0.8,
            fontsize=8.2, radius=0.035)
    ax.text(10.18, 5.95, "Aggregator", fontsize=10, fontweight="bold")
    ax.text(10.30, 3.05, "Multi-Modal\nAggregator F", fontsize=12,
            fontweight="bold", ha="left", va="top")
    arrow(ax, (9.68, 3.78), (10.35, 4.47))

    # Detection heads.
    rounded(ax, (11.85, 2.60), 3.75, 3.75, fc="#f9fbff", ec="#111111",
            lw=1.2, radius=0.07)
    rounded(ax, (12.92, 6.13), 1.66, 0.38, "Detection Heads",
            fc="#ffffff", lw=1.0, fontsize=12, weight="bold", radius=0.04)
    arrow(ax, (11.40, 4.47), (11.85, 4.47))

    rounded(ax, (12.04, 4.12), 1.02, 0.58, "Binary\nClassifier",
            fc=pink, lw=1.0, fontsize=8.8, weight="bold", radius=0.04)
    rounded(ax, (13.17, 4.12), 1.22, 0.58, "Multi-Label\nClassifier",
            fc=pink, lw=1.0, fontsize=8.6, weight="bold", radius=0.04)
    rounded(ax, (14.55, 4.12), 0.92, 0.58, "Token\nDetector",
            fc=pink, lw=1.0, fontsize=8.7, weight="bold", radius=0.04)

    ax.text(12.55, 3.89, "C_b", fontsize=8.5, ha="center")
    ax.text(13.78, 3.89, "C_m", fontsize=8.5, ha="center")
    ax.text(15.01, 3.89, "D_t", fontsize=8.5, ha="center")

    rounded(ax, (12.03, 5.12), 1.08, 0.80, fc="#ffffff", lw=0.9,
            ls=(0, (3, 3)), radius=0.03)
    circle_label(ax, 12.32, 5.50, 0.22, "Real", "#bfe5d5", fontsize=8.3)
    circle_label(ax, 12.78, 5.50, 0.22, "Fake", red, fontsize=8.3)
    ax.text(12.22, 6.00, "L_BIC", fontsize=8.5, fontstyle="italic")

    rounded(ax, (13.25, 5.12), 1.36, 0.80, fc="#ffffff", lw=0.9,
            ls=(0, (3, 3)), radius=0.03)
    for i, (lab, fc) in enumerate([("FS", blue), ("FA", blue),
                                   ("TS", gold), ("TA", "#fff9dc")]):
        circle_label(ax, 13.48 + i * 0.30, 5.50, 0.18, lab, fc, fontsize=7.6)
    ax.text(13.46, 6.00, "L_MLC", fontsize=8.5, fontstyle="italic")

    rounded(ax, (14.56, 5.12), 0.90, 0.80, "{manipulated\n tokens}",
            fc="#ffffff", lw=0.9, ls=(0, (3, 3)), radius=0.03,
            fontsize=7.2)
    ax.text(14.78, 6.00, "L_TMG", fontsize=8.5, fontstyle="italic")

    arrow(ax, (12.55, 4.70), (12.55, 5.12))
    arrow(ax, (13.78, 4.70), (13.78, 5.12))
    arrow(ax, (15.01, 4.70), (15.01, 5.12))
    ax.plot([12.30, 15.01], [3.82, 3.82], color="#111111", lw=1.8, zorder=3)
    arrow(ax, (12.55, 3.82), (12.55, 4.12), mutation=14)
    arrow(ax, (13.78, 3.82), (13.78, 4.12), mutation=14)
    arrow(ax, (15.01, 3.82), (15.01, 4.12), mutation=14)
    ax.text(12.88, 3.58,
            "All heads consume the shared fused representation from F",
            fontsize=6.8, color="#333333", ha="left")

    fig.tight_layout(pad=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
