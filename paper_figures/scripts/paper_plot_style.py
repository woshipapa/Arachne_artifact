"""
Common paper-quality matplotlib style.
Aligned for:
- Consistent font sizes (axes / ticks / legend)
- PDF/PS font embedding (no Type-3 fonts)
- Reviewer-friendly visual hierarchy

Usage:
    from paper_plot_style import apply_paper_style
    apply_paper_style(figsize=(W, H))
"""

import matplotlib.pyplot as plt


def apply_paper_style(figsize=(8, 5)):
    """
    Apply a unified paper-style plotting configuration.

    Args:
        figsize (tuple): Figure size in inches, e.g. (8, 5) or (16, 5.5)
    """

    # Base style (grid + spacing)
    plt.style.use("seaborn-v0_8-paper")

    plt.rcParams.update({
        # =========================
        # Font system (aligned)
        # =========================
        'font.family': 'sans-serif',
        'font.sans-serif': ['Verdana'],
        'mathtext.fontset': 'dejavusans',

        # =========================
        # PDF / PS font embedding
        # (CRITICAL for Word/WPS)
        # =========================
        'pdf.fonttype': 42,
        'ps.fonttype': 42,

        # =========================
        # Global font sizes
        # =========================
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,

        # =========================
        # Figure & axes
        # =========================
        'figure.figsize': figsize,
        'axes.linewidth': 1.2,

        # =========================
        # Tick style
        # =========================
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })