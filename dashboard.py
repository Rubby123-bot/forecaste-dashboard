import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px

# ===================== LOAD DATA =====================
# Change filename to your file
file_path = "forecaste_data.csv"
df = pd.read_csv(file_path)

# Ensure consistent column names (remove spaces & uppercase)
df.columns = [col.strip().upper() for col in df.columns]

# Check if date column exists and convert
if 'FORECASTED DATE' in df.columns:
    df['FORECASTED DATE'] = pd.to_datetime(df['FORECASTED DATE'], dayfirst=True, errors='coerce')

# ===================== DASH APP =====================
app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": "#f4f6f8",
        "padding": "20px",
        "font-family": "Arial"
    },
    children=[
        html.H1("📊 Procurement Forecast Dashboard", style={"textAlign": "center", "color": "#2c3e50"}),

        html.Div([
            html.Label("Filter by Supplier:"),
            dcc.Dropdown(
                id='supplier-filter',
                options=[{'label': s, 'value': s} for s in df['SUPPLIER'].unique()],
                multi=True,
                placeholder="Select Supplier"
            ),

            html.Label("Filter by Material:"),
            dcc.Dropdown(
                id='material-filter',
                options=[{'label': m, 'value': m} for m in df['MATERIAL'].unique()],
                multi=True,
                placeholder="Select Material"
            ),

            html.Label("Search by Material Description:"),
            dcc.Input(
                id='material-desc-search',
                type='text',
                placeholder="Enter material description",
                style={"width": "100%", "padding": "8px"}
            )
        ], style={"backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Div([
            dcc.Graph(id='line-chart'),
            dcc.Graph(id='bar-chart'),
            dcc.Graph(id='pie-chart')
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "10px"}),

        html.H3("Filtered Data Table"),
        dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            page_size=10
        )
    ]
)

# ===================== CALLBACKS =====================
@app.callback(
    [Output('line-chart', 'figure'),
     Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('data-table', 'data')],
    [Input('supplier-filter', 'value'),
     Input('material-filter', 'value'),
     Input('material-desc-search', 'value')]
)
def update_dashboard(selected_suppliers, selected_materials, search_text):
    filtered_df = df.copy()

    if selected_suppliers:
        filtered_df = filtered_df[filtered_df['SUPPLIER'].isin(selected_suppliers)]

    if selected_materials:
        filtered_df = filtered_df[filtered_df['MATERIAL'].isin(selected_materials)]

    if search_text:
        filtered_df = filtered_df[filtered_df['MATERIAL DISCRIPTION'].str.contains(search_text, case=False, na=False)]

    # Line chart
    line_fig = px.line(filtered_df, x='FORECASTED DATE', y='FORECASTED QUANTITY',
                       color='SUPPLIER', title="Forecast Trend")

    # Bar chart
    bar_fig = px.bar(filtered_df, x='MATERIAL', y='FORECASTED QUANTITY',
                     color='SUPPLIER', title="Forecast by Material")

    # Pie chart
    pie_fig = px.pie(filtered_df, names='SUPPLIER', values='FORECASTED QUANTITY',
                     title="Supplier Share")

    return line_fig, bar_fig, pie_fig, filtered_df.to_dict('records')

# ===================== RUN APP =====================
if __name__ == '__main__':
    app.run(debug=True)


