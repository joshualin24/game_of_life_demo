"""
visualize.py — Animated GIF visualizations of Catagolue periodic census data.

Produces:
  results/period_distribution.gif   — animated bar chart: objects per period, building up
  results/occurrence_decay.gif      — animated log-scale occurrence count per period
  results/top_objects_per_period.gif — top-10 objects within each period (animated bar race)
  results/occurrence_rank.gif       — rank vs occurrence (Zipf) curve per period animated
  results/period_gap_highlight.gif  — period presence/absence heatmap animating in

Usage:
    python visualize.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT  = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

BG      = "#0f1117"
TEXT_C  = "white"
ACCENT  = "#3a7bd5"
PALETTE = ["#3a7bd5","#2ecc71","#e74c3c","#f39c12","#9b59b6",
           "#1abc9c","#e67e22","#e91e63","#00bcd4","#8bc34a"]


def load_summary():
    with open(os.path.join(DATA, "summary.json")) as f:
        return json.load(f)

def load_census(period):
    path = os.path.join(DATA, f"census_xp{period}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

def styled(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=TEXT_C, fontsize=11, fontweight="bold", pad=8)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)


# ── 1. Period distribution — animated bar chart building period by period ─────

def gif_period_distribution(summary, fps=4):
    periods     = [r["period"] for r in summary]
    n_objects   = [r["n_objects"] for r in summary]
    has_objects = [n > 0 for n in n_objects]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    styled(ax, "Distinct Oscillator Types per Period (Catagolue B3/S23)")
    ax.set_xlabel("Period", color=TEXT_C, fontsize=9)
    ax.set_ylabel("# distinct objects", color=TEXT_C, fontsize=9)
    ax.set_xlim(0.5, max(periods) + 0.5)
    ax.set_ylim(0, max(n_objects) * 1.15)
    ax.set_xticks(periods)
    ax.set_xticklabels([str(p) for p in periods], fontsize=7)

    bars = ax.bar(periods, [0]*len(periods),
                  color=[PALETTE[1] if h else "#444" for h in has_objects],
                  edgecolor="#222", linewidth=0.5)
    count_text = ax.text(0.98, 0.95, "", transform=ax.transAxes,
                         ha="right", va="top", color=TEXT_C, fontsize=9)

    # annotation: gap periods
    ax.text(0.02, 0.95, "Gray = no naturally-occurring objects",
            transform=ax.transAxes, color="#888", fontsize=8, va="top")

    def update(i):
        for j, bar in enumerate(bars):
            bar.set_height(n_objects[j] if j <= i else 0)
        count_text.set_text(f"Period {periods[i]}:  {n_objects[i]:,} objects")
        return list(bars) + [count_text]

    ani = animation.FuncAnimation(fig, update, frames=len(periods),
                                  interval=1000//fps, blit=True)
    path = os.path.join(OUT, "period_distribution.gif")
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")


# ── 2. Occurrence decay — log-scale bar animating in ─────────────────────────

def gif_occurrence_decay(summary, fps=4):
    rows = [r for r in summary if r["total_occurrences"] > 0]
    periods = [r["period"] for r in rows]
    occ     = [r["total_occurrences"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    styled(ax, "Total Occurrences per Period (log scale)")
    ax.set_xlabel("Period", color=TEXT_C, fontsize=9)
    ax.set_ylabel("Total occurrences (log₁₀)", color=TEXT_C, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlim(min(periods) - 1, max(periods) + 1)
    ax.set_ylim(1, max(occ) * 5)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.set_xticks(periods)
    ax.set_xticklabels([str(p) for p in periods], fontsize=8)
    ax.grid(axis="y", color="#222", linestyle="--", linewidth=0.5)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(periods))]
    bars = ax.bar(periods, [1e-1]*len(periods), color=colors,
                  edgecolor="#222", linewidth=0.5)
    label = ax.text(0.98, 0.97, "", transform=ax.transAxes,
                    ha="right", va="top", color=TEXT_C, fontsize=9)

    def update(i):
        for j, bar in enumerate(bars):
            bar.set_height(occ[j] if j <= i else 1e-1)
        label.set_text(f"xp{periods[i]}: {occ[i]:,.0f}")
        return list(bars) + [label]

    ani = animation.FuncAnimation(fig, update, frames=len(periods),
                                  interval=1000//fps, blit=True)
    path = os.path.join(OUT, "occurrence_decay.gif")
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")


# ── 3. Top objects per period — horizontal bar race ───────────────────────────

def gif_top_objects(fps=3, top_n=10):
    active_periods = []
    all_data = {}
    summary = load_summary()
    for r in summary:
        p = r["period"]
        if r["n_objects"] == 0:
            continue
        data = load_census(p)
        if not data:
            continue
        data_sorted = sorted(data, key=lambda x: -x["occurrences"])[:top_n]
        all_data[p] = data_sorted
        active_periods.append(p)

    if not active_periods:
        return

    max_occ = max(o["occurrences"] for p in active_periods for o in all_data[p])

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_C, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.set_xlabel("Occurrences (log₁₀)", color=TEXT_C, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1, max_occ * 3)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.label.set_color(TEXT_C)

    period_title = ax.set_title("", color=TEXT_C, fontsize=12, fontweight="bold")

    def update(i):
        ax.cla()
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_C, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.set_xlabel("Occurrences (log₁₀)", color=TEXT_C, fontsize=9)
        ax.xaxis.label.set_color(TEXT_C)
        ax.set_xscale("log")
        ax.set_xlim(1, max_occ * 3)
        ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

        p = active_periods[i]
        data = all_data[p]
        labels = [o["apgcode"] for o in data]
        values = [max(o["occurrences"], 1) for o in data]
        colors = [PALETTE[j % len(PALETTE)] for j in range(len(data))]

        bars = ax.barh(range(len(data)), values, color=colors,
                       edgecolor="#222", linewidth=0.5)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(labels, fontsize=7, color=TEXT_C)
        ax.invert_yaxis()
        ax.set_title(f"Top objects — Period {p}  ({len(load_census(p))} total)",
                     color=TEXT_C, fontsize=11, fontweight="bold")
        ax.grid(axis="x", color="#222", linestyle="--", linewidth=0.4)
        return bars

    ani = animation.FuncAnimation(fig, update, frames=len(active_periods),
                                  interval=1500, blit=False)
    path = os.path.join(OUT, "top_objects_per_period.gif")
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")


# ── 4. Zipf / rank-occurrence curve animated per period ──────────────────────

def gif_zipf(fps=3):
    summary = load_summary()
    active = [(r["period"], load_census(r["period"]))
              for r in summary if r["n_objects"] > 0]
    active = [(p, d) for p, d in active if d]
    if not active:
        return

    max_occ = max(o["occurrences"] for _, d in active for o in d)
    max_rank = max(len(d) for _, d in active)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    styled(ax, "Rank vs Occurrences (Zipf) — by Period")
    ax.set_xlabel("Rank", color=TEXT_C, fontsize=9)
    ax.set_ylabel("Occurrences (log₁₀)", color=TEXT_C, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.8, max_rank * 1.5)
    ax.set_ylim(0.5, max_occ * 5)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.grid(color="#222", linestyle="--", linewidth=0.4)

    line, = ax.plot([], [], lw=2, color=ACCENT)
    dots  = ax.scatter([], [], s=15, color=ACCENT, zorder=3)
    label = ax.text(0.02, 0.05, "", transform=ax.transAxes,
                    color=TEXT_C, fontsize=10, fontweight="bold")
    period_label = ax.text(0.98, 0.97, "", transform=ax.transAxes,
                           ha="right", va="top", color=TEXT_C, fontsize=12,
                           fontweight="bold")

    def update(i):
        p, data = active[i]
        occ_sorted = sorted([o["occurrences"] for o in data], reverse=True)
        ranks = np.arange(1, len(occ_sorted) + 1)
        color = PALETTE[i % len(PALETTE)]
        line.set_data(ranks, occ_sorted)
        line.set_color(color)
        dots.set_offsets(np.c_[ranks, occ_sorted])
        dots.set_color(color)
        label.set_text(f"{len(data)} objects")
        period_label.set_text(f"xp{p}")
        return line, dots, label, period_label

    ani = animation.FuncAnimation(fig, update, frames=len(active),
                                  interval=1200, blit=True)
    path = os.path.join(OUT, "occurrence_rank.gif")
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")


# ── 5. Period presence heatmap animating in ──────────────────────────────────

def gif_period_heatmap(summary, fps=6):
    periods = [r["period"] for r in summary]
    n_obj   = np.array([r["n_objects"] for r in summary], dtype=float)

    # Build 2D grid: rows = presence buckets, cols = periods
    # Just show log(n_objects+1) as a 1×30 heatmap that reveals left to right
    vals = np.log10(n_obj + 1)

    fig, (ax_heat, ax_bar) = plt.subplots(2, 1, figsize=(12, 6),
                                           gridspec_kw={"height_ratios": [1, 3]})
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(hspace=0.35)

    # Heatmap row
    ax_heat.set_facecolor(BG)
    ax_heat.set_title("Period Presence Map  (log₁₀ distinct objects)",
                       color=TEXT_C, fontsize=10, fontweight="bold")
    ax_heat.set_yticks([])
    ax_heat.set_xticks(range(len(periods)))
    ax_heat.set_xticklabels([str(p) for p in periods], fontsize=7, color=TEXT_C)
    for sp in ax_heat.spines.values(): sp.set_edgecolor("#333")

    hidden = np.full_like(vals, np.nan)
    im = ax_heat.imshow(hidden[np.newaxis, :], cmap="YlOrRd",
                         aspect="auto", vmin=0, vmax=vals.max())
    fig.colorbar(im, ax=ax_heat, orientation="horizontal", fraction=0.05,
                 pad=0.35).ax.tick_params(colors=TEXT_C, labelsize=7)

    # Bar chart below
    styled(ax_bar, "")
    ax_bar.set_xlabel("Period", color=TEXT_C, fontsize=9)
    ax_bar.set_ylabel("Distinct objects (log₁₀)", color=TEXT_C, fontsize=9)
    ax_bar.set_yscale("log"); ax_bar.set_ylim(0.5, n_obj.max() * 3)
    ax_bar.set_xlim(-0.5, len(periods) - 0.5)
    ax_bar.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_bar.grid(axis="y", color="#222", linestyle="--", linewidth=0.4)
    ax_bar.set_xticks(range(len(periods)))
    ax_bar.set_xticklabels([str(p) for p in periods], fontsize=7)

    bars = ax_bar.bar(range(len(periods)), [1e-1]*len(periods),
                       color=[PALETTE[1] if n > 0 else "#333" for n in n_obj],
                       edgecolor="#111", linewidth=0.4)
    step_label = ax_bar.text(0.98, 0.97, "", transform=ax_bar.transAxes,
                              ha="right", va="top", color=TEXT_C, fontsize=9)

    def update(i):
        revealed = hidden.copy()
        revealed[:i+1] = vals[:i+1]
        im.set_data(revealed[np.newaxis, :])
        for j, bar in enumerate(bars):
            bar.set_height(max(n_obj[j], 1e-1) if j <= i else 1e-1)
        step_label.set_text(f"xp{periods[i]}  —  {int(n_obj[i])} objects")
        return [im] + list(bars) + [step_label]

    ani = animation.FuncAnimation(fig, update, frames=len(periods),
                                  interval=1000//fps, blit=True)
    path = os.path.join(OUT, "period_gap_highlight.gif")
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    summary = load_summary()
    print("Generating GIF visualizations …")
    gif_period_distribution(summary)
    gif_occurrence_decay(summary)
    gif_top_objects()
    gif_zipf()
    gif_period_heatmap(summary)
    print("Done.")

if __name__ == "__main__":
    main()
