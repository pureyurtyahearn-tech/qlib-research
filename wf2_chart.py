"""Render the walk-forward result as a two-panel plotly figure (per the charting preference):
  top  = annual out-of-sample top-50 net excess vs EW for FCF yield & ROE (grouped bars);
  bottom = annual market breadth (% of members beating the equal-weight universe).
2024-2026 shaded as the already-observed holdout. Writes an artifact-ready HTML fragment
(inline plotly.js, self-contained for the CSP) + a PNG for the repo.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SH = Path("git_ignore_folder/sharadar")
BLUE, ORANGE, GREEN, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781", "#e1e0d9"


def main():
    wf = pd.read_csv(SH / "wf_walkforward.csv")
    yr = wf["year"].values
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.64, 0.36],
                        vertical_spacing=0.07,
                        subplot_titles=("Annual out-of-sample top-50 net excess vs equal-weight",
                                        "Market breadth — share of members beating equal-weight"))

    fig.add_trace(go.Bar(x=yr, y=wf["fcfy_ex"] * 100, name="FCF yield",
                         marker_color=BLUE, offsetgroup=0,
                         hovertemplate="%{x}<br>FCF yield: %{y:+.1f}%<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=yr, y=wf["roe_ex"] * 100, name="ROE",
                         marker_color=ORANGE, offsetgroup=1,
                         hovertemplate="%{x}<br>ROE: %{y:+.1f}%<extra></extra>"), 1, 1)
    fig.add_hline(y=0, line_color=MUTED, line_width=1.3, row=1, col=1)

    fig.add_trace(go.Scatter(x=yr, y=wf["breadth"] * 100, name="Breadth",
                             mode="lines+markers", line=dict(color=GREEN, width=2.5),
                             marker=dict(size=6), showlegend=False,
                             hovertemplate="%{x}<br>breadth: %{y:.0f}%<extra></extra>"), 2, 1)
    bmed = wf.loc[wf.year <= 2023, "breadth"].median() * 100
    fig.add_hline(y=bmed, line_color=MUTED, line_width=1, line_dash="dot", row=2, col=1,
                  annotation_text=f"2001-23 median {bmed:.0f}%", annotation_position="bottom left",
                  annotation_font_size=11, annotation_font_color=MUTED)

    # shade 2024-2026 (already-observed holdout) in both panels
    for r in (1, 2):
        fig.add_vrect(x0=2023.5, x1=2026.5, fillcolor=MUTED, opacity=0.10,
                      line_width=0, row=r, col=1)
    fig.add_annotation(x=2025, y=1.0, yref="y domain", text="holdout<br>(already seen)",
                       showarrow=False, font=dict(size=10, color=MUTED), row=1, col=1)

    fig.update_layout(
        template="simple_white", font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                                           color=INK, size=13),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        barmode="group", bargap=0.28, bargroupgap=0.08,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        margin=dict(l=60, r=24, t=64, b=40), height=560)
    fig.update_yaxes(title_text="% / yr", gridcolor=GRID, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="% of names", gridcolor=GRID, row=2, col=1)
    fig.update_xaxes(gridcolor=GRID, dtick=2, row=2, col=1)
    for a in fig.layout.annotations[:2]:
        a.font.size = 13.5; a.font.color = INK; a.xanchor = "left"; a.x = 0

    out_png = SH / "wf_walkforward.png"
    fig.write_image(str(out_png), scale=2)
    frag = fig.to_html(full_html=False, include_plotlyjs="inline",
                       config={"displayModeBar": False, "responsive": True})
    Path("wf_chart_fragment.html").write_text(frag, encoding="utf-8")
    print(f"wrote {out_png} and wf_chart_fragment.html ({len(frag)//1024} KB)")


if __name__ == "__main__":
    main()
