import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from pathlib import Path

DB_PATH = Path("pipeline/farmcheck.duckdb")

app = Dash(__name__, title="FarmCheck AI — Adoption Dashboard")

# ── Data loaders ──────────────────────────────────────────────────────────
def get_con():
    return duckdb.connect(str(DB_PATH), read_only=True)

def load_village_summary() -> pd.DataFrame:
    return get_con().execute("""
        SELECT * FROM gold_village_summary
        ORDER BY avg_adoption_score DESC
    """).df()

def load_household_adoption() -> pd.DataFrame:
    return get_con().execute("""
        SELECT * FROM gold_household_adoption
        ORDER BY adoption_score DESC
    """).df()

def load_indicator_distribution() -> pd.DataFrame:
    return get_con().execute("""
        SELECT indicator, COUNT(*) as count,
               ROUND(AVG(confidence) * 100, 1) as avg_confidence
        FROM silver_predictions
        GROUP BY indicator
        ORDER BY count DESC
    """).df()

def load_trend_over_time() -> pd.DataFrame:
    return get_con().execute("""
        SELECT
            CAST(received_at AS DATE)   AS date,
            ROUND(AVG(binary_score) * 100, 1) AS daily_compliance_pct,
            COUNT(*)                    AS assessments
        FROM silver_predictions
        WHERE is_valid = true
        GROUP BY CAST(received_at AS DATE)
        ORDER BY date
    """).df()

# ── KPI cards ─────────────────────────────────────────────────────────────
def kpi_card(title, value, color="#2ecc71"):
    return html.Div([
        html.P(title, style={"margin": "0", "fontSize": "13px", "color": "#888"}),
        html.H3(value, style={"margin": "4px 0 0", "color": color}),
    ], style={
        "background": "#1e1e2e",
        "borderRadius": "8px",
        "padding": "16px 20px",
        "flex": "1",
        "minWidth": "150px",
        "borderLeft": f"4px solid {color}",
    })

# ── Layout ────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.Div([
        html.H1("🌱 FarmCheck AI", style={"margin": "0", "color": "#2ecc71"}),
        html.P("Household Adoption Dashboard — Agriculture Compliance",
               style={"margin": "4px 0 0", "color": "#aaa"}),
    ], style={"padding": "24px 32px 16px", "borderBottom": "1px solid #2a2a3e"}),

    # KPI row
    html.Div(id="kpi-row", style={
        "display": "flex", "gap": "16px",
        "padding": "20px 32px", "flexWrap": "wrap"
    }),

    # Village filter
    html.Div([
        html.Label("Filter by Village:", style={"color": "#aaa", "marginRight": "12px"}),
        dcc.Dropdown(
            id="village-filter",
            options=[{"label": "All Villages", "value": "all"}],
            value="all",
            clearable=False,
            style={"width": "260px", "color": "#000"}
        )
    ], style={"padding": "0 32px 16px", "display": "flex", "alignItems": "center"}),

    # Charts row 1
    html.Div([
        dcc.Graph(id="village-bar",   style={"flex": "1", "minWidth": "300px"}),
        dcc.Graph(id="indicator-pie", style={"flex": "1", "minWidth": "300px"}),
    ], style={"display": "flex", "gap": "16px", "padding": "0 32px"}),

    # Charts row 2
    html.Div([
        dcc.Graph(id="trend-line",     style={"flex": "2", "minWidth": "400px"}),
        dcc.Graph(id="household-hist", style={"flex": "1", "minWidth": "300px"}),
    ], style={"display": "flex", "gap": "16px", "padding": "16px 32px"}),

    # Household table
    html.Div([
        html.H3("Household Adoption Scores",
                style={"color": "#eee", "marginBottom": "12px"}),
        html.Div(id="household-table")
    ], style={"padding": "0 32px 32px"}),

    # Refresh interval
    dcc.Interval(id="interval", interval=30_000, n_intervals=0),

], style={"background": "#13131f", "minHeight": "100vh", "fontFamily": "Inter, sans-serif"})


# ── Callbacks ─────────────────────────────────────────────────────────────
@app.callback(
    Output("village-filter", "options"),
    Input("interval", "n_intervals")
)
def update_village_options(_):
    df = load_village_summary()
    opts = [{"label": "All Villages", "value": "all"}]
    opts += [{"label": v, "value": v} for v in df["village"]]
    return opts


@app.callback(
    Output("kpi-row", "children"),
    Input("village-filter", "value"),
    Input("interval", "n_intervals")
)
def update_kpis(village, _):
    hh  = load_household_adoption()
    sil = get_con().execute("SELECT COUNT(*) FROM silver_predictions").fetchone()[0]

    if village != "all":
        hh = hh[hh["village"] == village]

    avg_score    = round(hh["adoption_score"].mean(), 1)
    at_risk      = int((hh["adoption_score"] < 50).sum())
    improving    = int((hh["trend"] == "improving").sum())
    total_hh     = len(hh)

    return [
        kpi_card("Total Assessments",    f"{sil:,}",      "#3498db"),
        kpi_card("Households Monitored", f"{total_hh}",   "#9b59b6"),
        kpi_card("Avg Adoption Score",   f"{avg_score}%", "#2ecc71"),
        kpi_card("At Risk (<50%)",        f"{at_risk}",    "#e74c3c"),
        kpi_card("Improving Trend",       f"{improving}",  "#f39c12"),
    ]


@app.callback(
    Output("village-bar", "figure"),
    Input("village-filter", "value"),
    Input("interval", "n_intervals")
)
def update_village_bar(village, _):
    df = load_village_summary()
    fig = px.bar(
        df, x="village", y="avg_adoption_score",
        color="avg_adoption_score",
        color_continuous_scale="Greens",
        title="Average Adoption Score by Village",
        labels={"avg_adoption_score": "Adoption Score (%)", "village": "Village"},
        text="avg_adoption_score",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**dark_layout())
    return fig


@app.callback(
    Output("indicator-pie", "figure"),
    Input("village-filter", "value"),
    Input("interval", "n_intervals")
)
def update_indicator_pie(village, _):
    df = load_indicator_distribution()
    fig = px.pie(
        df, names="indicator", values="count",
        title="Compliance Issue Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4,
    )
    fig.update_layout(**dark_layout())
    return fig


@app.callback(
    Output("trend-line", "figure"),
    Input("village-filter", "value"),
    Input("interval", "n_intervals")
)
def update_trend(village, _):
    df = load_trend_over_time()
    fig = px.line(
        df, x="date", y="daily_compliance_pct",
        title="Daily Compliance Rate Over Time",
        labels={"daily_compliance_pct": "Compliance %", "date": "Date"},
        markers=True,
    )
    fig.add_hline(y=50, line_dash="dash", line_color="#e74c3c",
                  annotation_text="50% threshold")
    fig.update_traces(line_color="#2ecc71")
    fig.update_layout(**dark_layout())
    return fig


@app.callback(
    Output("household-hist", "figure"),
    Input("village-filter", "value"),
    Input("interval", "n_intervals")
)
def update_histogram(village, _):
    hh = load_household_adoption()
    if village != "all":
        hh = hh[hh["village"] == village]

    fig = px.histogram(
        hh, x="adoption_score", nbins=10,
        title="Household Adoption Score Distribution",
        labels={"adoption_score": "Adoption Score (%)"},
        color_discrete_sequence=["#2ecc71"],
    )
    fig.update_layout(**dark_layout())
    return fig


@app.callback(
    Output("household-table", "children"),
    Input("village-filter", "value"),
    Input("interval", "n_intervals")
)
def update_table(village, _):
    hh = load_household_adoption()
    if village != "all":
        hh = hh[hh["village"] == village]

    hh = hh.sort_values("adoption_score", ascending=False)

    def score_color(score):
        if score >= 70:  return "#2ecc71"
        if score >= 50:  return "#f39c12"
        return "#e74c3c"

    def trend_icon(trend):
        return {"improving": "⬆️", "declining": "⬇️", "stable": "➡️"}.get(trend, "—")

    rows = []
    for _, row in hh.iterrows():
        rows.append(html.Tr([
            html.Td(row["household_id"],  style=cell_style()),
            html.Td(row["village"],       style=cell_style()),
            html.Td(f"{row['adoption_score']}%",
                    style={**cell_style(), "color": score_color(row["adoption_score"]),
                           "fontWeight": "bold"}),
            html.Td(row["total_assessments"], style=cell_style()),
            html.Td(row["dominant_indicator"] or "—", style=cell_style()),
            html.Td(f"{trend_icon(row['trend'])} {row['trend']}", style=cell_style()),
        ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th(h, style=header_style())
            for h in ["Household", "Village", "Adoption Score",
                      "Assessments", "Main Issue", "Trend"]
        ])),
        html.Tbody(rows)
    ], style={"width": "100%", "borderCollapse": "collapse"})


# ── Style helpers ─────────────────────────────────────────────────────────
def dark_layout():
    return dict(
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font_color="#eee",
        margin=dict(t=40, b=20, l=20, r=20),
    )

def cell_style():
    return {"padding": "10px 14px", "borderBottom": "1px solid #2a2a3e",
            "color": "#ddd", "fontSize": "13px"}

def header_style():
    return {"padding": "10px 14px", "background": "#2a2a3e",
            "color": "#aaa", "fontSize": "12px",
            "textTransform": "uppercase", "letterSpacing": "0.5px"}


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)