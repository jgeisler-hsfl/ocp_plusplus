#!/usr/bin/env python3
"""Plot an ocp++ solution CSV (ocp::Solution::to_csv) with plotly.

The CSV has a header row ("k" followed by one named column per state and
control component, taken from the problem's Dims names or the defaults
"x_i" / "u_i") and one row per stage k = 0..N. The terminal row (k = N)
leaves the control columns blank (no u_N); blanks are drawn as gaps.

Each variable is plotted in its own subplot, all sharing the stage index
k on the x-axis.

Usage:
    python3 tools/plot_solution.py solution.csv                 # -> solution.html
    python3 tools/plot_solution.py solution.csv -o out.html
    python3 tools/plot_solution.py solution.csv -o out.png      # static PNG

Dependencies (user-installed; the C++ build does not need them):
    pip install plotly          # required (HTML output)
    pip install plotly kaleido  # only for PNG output
"""

import argparse
import csv
import os
import sys


def read_solution_csv(path):
    """Read a to_csv file; return (stages, data, names).

    stages: list of stage indices (float, one per data row)
    data:   list of lists, data[i][k] = value of variable i at stage k
            (float('nan') where the CSV cell is blank)
    names:  list of variable column names (order of the CSV columns)
    """
    with open(path, newline="") as f:
        rows = [row for row in csv.reader(f) if row]
    if len(rows) < 2:
        raise ValueError("%s: expected a header and at least one data row"
                         % path)
    header = rows[0]
    if header[0] != "k":
        raise ValueError("%s: first column must be 'k', got %r"
                         % (path, header[0]))
    names = header[1:]
    if not names:
        raise ValueError("%s: no variable columns found" % path)
    stages = [float(row[0]) for row in rows[1:]]
    data = []
    for i in range(len(names)):
        col = []
        for row in rows[1:]:
            cell = row[i + 1] if i + 1 < len(row) else ""
            col.append(float(cell) if cell.strip() != "" else float("nan"))
        data.append(col)
    return stages, data, names


def build_figure(stages, data, names, title):
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    n = len(names)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=names,
    )
    for i in range(n):
        fig.add_trace(
            go.Scatter(x=stages, y=data[i], mode="lines", name=names[i]),
            row=i + 1, col=1,
        )
    fig.update_xaxes(title_text="stage k", row=n, col=1)
    fig.update_layout(
        height=180 * n + 120,
        title=title,
        showlegend=False,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot an ocp++ solution CSV (Solution::to_csv) with "
                    "plotly; one subplot per variable.")
    parser.add_argument("csv", help="solution CSV file (Solution::to_csv)")
    parser.add_argument("-o", "--output", default=None,
                        help="output file; .html (default) or .png")
    parser.add_argument("--title", default=None,
                        help="plot title (default: CSV file name)")
    args = parser.parse_args()

    try:
        import plotly  # noqa: F401
    except ImportError:
        sys.exit("plotly is not installed. Install it with: "
                 "pip install plotly")

    try:
        stages, data, names = read_solution_csv(args.csv)
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))

    fig = build_figure(stages, data, names,
                       args.title or os.path.basename(args.csv))

    out = args.output or os.path.splitext(args.csv)[0] + ".html"
    if out.lower().endswith(".png"):
        try:
            import kaleido  # noqa: F401
        except ImportError:
            sys.exit("PNG export requires kaleido. Install it with: "
                     "pip install plotly kaleido")
        fig.write_image(out, width=1200, height=int(fig.layout.height),
                        scale=2)
    else:
        if not out.lower().endswith(".html"):
            out += ".html"
        fig.write_html(out, auto_open=False)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
