from dash import Dash, html, Input, Output, State, dcc, no_update, callback_context
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

# Khởi tạo app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Hàm tạo figure rỗng
def create_fig():
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="x"),
        yaxis=dict(title="y"),
        title="Biểu đồ hồi quy"
    )
    return fig

# Hàm vẽ hồi quy
def draw_regularizate_regression(x, y, mode):
    fig = create_fig()
    if not mode:
        model = LinearRegression()
    elif set(mode) == {"ridge"}:
        model = Ridge()
    elif set(mode) == {"lasso"}:
        model = Lasso()
    elif set(mode) == {"ridge", "lasso"}:
        model = ElasticNet()
    model.fit(x.reshape(-1,1), y)
    y_pred = model.predict(x.reshape(-1,1))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Thực tế"))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode="markers", name="Dự đoán"))
    return fig

# Layout
app.layout = html.Div(
    id="app-container",
    children=[
        # Nút toggle theme
        html.Div(
            dbc.Button("🌞/🌜", id="theme-toggle", n_clicks=0, color="secondary"),
            style={"textAlign": "right", "marginBottom": "10px"}
        ),

        # Header
        html.H2("Vẽ đồ thị hồi quy cơ bản", style={"textAlign": "center", "marginBottom": "30px"}),

        # Main content
        dbc.Row(
            [
                # Form bên trái
                dbc.Col(
                    [
                        dbc.Form(
                            [
                                dbc.Row([dbc.Label("Nhập a:", width=5),
                                         dbc.Col(dbc.Input(id="a", type="number", value=0), width=5)], className="my-2"),
                                dbc.Row([dbc.Label("Nhập b:", width=5),
                                         dbc.Col(dbc.Input(id="b", type="number", value=0), width=5)], className="my-2"),
                                dbc.Row([dbc.Label("Nhập x1:", width=5),
                                         dbc.Col(dbc.Input(id="x1", type="number", value=0), width=5)], className="my-2"),
                                dbc.Row([dbc.Label("Nhập x2:", width=5),
                                         dbc.Col(dbc.Input(id="x2", type="number", value=1), width=5)], className="my-2"),
                                dbc.Row([dbc.Label("Số điểm:", width=5),
                                         dbc.Col(dbc.Input(id="x-num", type="number", min=2, value=10), width=5)], className="my-2"),
                                dbc.Row([dbc.Label("Ngưỡng dưới:", width=5),
                                         dbc.Col(dbc.Input(id="epsilon1", type="number", value=-0.1), width=5)], className="my-2"),
                                dbc.Row([dbc.Label("Ngưỡng trên:", width=5),
                                         dbc.Col(dbc.Input(id="epsilon2", type="number", value=0.1), width=5)], className="my-2"),
                                dbc.Row([dbc.Label("Chọn mô hình:", width=5),
                                         dbc.Col(dcc.Checklist(
                                             id="model",
                                             options=[{"label": "Hồi quy Lasso", "value": "lasso"},
                                                      {"label": "Hồi quy Ridge", "value": "ridge"}],
                                             value=[],
                                             inputStyle={"margin-right": "10px"}
                                         ), width=5)], className="my-2")
                            ]
                        ),
                        html.Div(dbc.Button("Vẽ!", id="btn", className="mt-3"), className="text-center")
                    ],
                    width=3
                ),

                # Graph bên phải
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Biểu đồ"),
                            dbc.CardBody([dcc.Graph(id="graph")])
                        ]
                    ),
                    width=9
                ),
            ]
        ),

        # Modal cảnh báo
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("⚠️ Cảnh báo")),
                dbc.ModalBody(id="modal-body"),
                dbc.ModalFooter(dbc.Button("Đóng", id="close-modal", className="ms-auto"))
            ],
            id="modal",
            is_open=False,
        )
    ],
    style={"minHeight":"100vh", "padding":"20px", "fontFamily":"Arial, sans-serif","fontSize":"16px"}
)

# Callback toggle theme
@app.callback(
    Output("app-container", "style"),
    Input("theme-toggle", "n_clicks"),
    State("app-container", "style")
)
def toggle_theme(n_clicks, style):
    if style is None:
        style = {"minHeight":"100vh", "padding":"20px","fontFamily":"Arial, sans-serif","fontSize":"16px"}
    if n_clicks % 2 == 1:
        style.update({"backgroundColor":"#222","color":"white"})
    else:
        style.update({"backgroundColor":"white","color":"black"})
    return style

@app.callback(
    Output("graph", "figure"),
    Output("modal", "is_open"),
    Output("modal-body", "children"),
    Input("btn", "n_clicks"),
    Input("close-modal", "n_clicks"),
    State("model", "value"),
    State("a", "value"),
    State("b", "value"),
    State("x1", "value"),
    State("x2", "value"),
    State("x-num", "value"),
    State("epsilon1", "value"),
    State("epsilon2", "value"),
    State("modal", "is_open")
)
def update_graph_or_modal(btn, close_btn, model_name, a, b, x1, x2, nums, e1, e2, is_open):
    ctx = callback_context

    if not ctx.triggered:
        return create_fig(), False, ""

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "close-modal":
        return no_update, False, ""
    elif trigger_id == "btn":
        if None in [a,b,x1,x2,nums,e1,e2]:
            return create_fig(), True, "Vui lòng nhập đầy đủ các giá trị!"
        if x2 - x1 <= 1e-6:
            return create_fig(), True, "Điều kiện x1 < x2 không thỏa mãn!"
        if e1 > e2:
            return create_fig(), True, "Điều kiện epsilon1 < epsilon2 không thỏa mãn!"
        if nums < 2:
            return create_fig(), True, "Số điểm phải ≥ 2!"
        # Vẽ biểu đồ
        x = np.linspace(x1, x2, nums)
        y = a*x + b + np.random.uniform(e1,e2,nums)
        return draw_regularizate_regression(x, y, model_name), False, ""

    return create_fig(), is_open, ""


server = app.server