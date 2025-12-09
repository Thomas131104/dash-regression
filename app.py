from dash import Dash, html, Input, Output, State, dcc
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div(
    [
        html.Header(
            html.H2("Vẽ đồ thị hồi quy cơ bản", id="title")
        ),

        html.Main(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Form(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Label("Nhập a:", html_for="a", width=3),
                                            dbc.Col(
                                                dbc.Input(id="a", type="number", value=0),
                                                width=6,
                                            ),
                                        ],
                                        className="my-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("Nhập b:", html_for="b", width=3),
                                            dbc.Col(
                                                dbc.Input(id="b", type="number", value=0),
                                                width=6,
                                            ),
                                        ],
                                        className="my-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("Nhập x1:", html_for="x1", width=3),
                                            dbc.Col(
                                                dbc.Input(id="x1", type="number"), width=6
                                            ),
                                        ],
                                        className="my-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("Nhập x2:", html_for="x2", width=3),
                                            dbc.Col(
                                                dbc.Input(id="x2", type="number"), width=6
                                            ),
                                        ],
                                        className="my-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label(
                                                "Nhập số điểm:", html_for="x-num", width=3
                                            ),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="x-num",
                                                    type="number",
                                                    min=2,
                                                    value=10,
                                                ),
                                                width=6,
                                            ),
                                        ],
                                        className="my-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label(
                                                "Ngưỡng dưới của nhiễu:",
                                                html_for="epsilon1",
                                                width=3,
                                            ),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="epsilon1", type="number", value=-0.1
                                                ),
                                                width=6,
                                            ),
                                        ],
                                        className="my-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label(
                                                "Ngưỡng trên của nhiễu:",
                                                html_for="epsilon2",
                                                width=3,
                                            ),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="epsilon2", type="number", value=0.1
                                                ),
                                                width=6,
                                            ),
                                        ],
                                        className="my-3",
                                    ),
                                    html.Div(
                                        id="warning",
                                        style={"color": "red", "marginTop": "10px"},
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("Chọn mô hình:", width=3),
                                            dbc.Col(
                                                dcc.Checklist(
                                                    id="model",
                                                    options=[
                                                        {
                                                            "label": "Hồi quy Lasso",
                                                            "value": "lasso",
                                                        },
                                                        {
                                                            "label": "Hồi quy Ridge",
                                                            "value": "ridge",
                                                        },
                                                    ],
                                                    value=[],
                                                    inputStyle={"margin-right": "10px"},
                                                ),
                                                width=6,
                                            ),
                                        ],
                                        className="my-3",
                                    ),
                                ]
                            ),
                            html.Div(
                                dbc.Button("Vẽ!", id="btn", className="mt-3"),
                                className="text-center"
                            )
                        ],
                        width=5,
                        className="justify-content-center align-items-center"
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Biểu đồ"),
                                    dbc.CardBody([dcc.Graph(id="graph")]),
                                ]
                            )
                        ],
                        width=7,
                    ),
                ]
            ),
        ),

        html.Footer(
            html.H2("By Mus", id = "title")
        )
    ]
)


@app.callback(
    Output("warning", "children"),
    Input("x1", "value"),
    Input("x2", "value"),
    Input("epsilon1", "value"),
    Input("epsilon2", "value"),
)
def warn_if_invalid(x1, x2, epsilon1, epsilon2):
    if x1 is None or x2 is None or epsilon1 is None or epsilon2 is None:
        return ""
    if x1 >= x2:
        return "⚠️ Điều kiện x1 < x2 không thỏa mãn!"
    if epsilon1 > epsilon2:
        return "⚠️ Điều kiện epsilon1 < epsilon2 không thỏa mãn!"
    return ""


def create_fig():
    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(title="x"), yaxis=dict(title="y"), title="Biểu đồ hồi quy"
    )

    return fig


def draw_regularizate_regression(x, y, mode):
    fig = create_fig()

    model = None

    if not mode:
        model = LinearRegression()
    elif set(mode) == {"ridge"}:
        model = Ridge()
    elif set(mode) == {"lasso"}:
        model = Lasso()
    elif set(mode) == {"ridge", "lasso"}:
        model = ElasticNet()
    else:
        raise Exception("Không thể đi tới đây được")

    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Thực tế"))

    fig.add_trace(go.Scatter(x=x, y=y_pred, mode="markers", name="Dự đoán"))

    return fig


@app.callback(
    Output("graph", "figure"),
    Input("btn", "n_clicks"),
    Input("model", "value"),
    State("a", "value"),
    State("b", "value"),
    State("x1", "value"),
    State("x2", "value"),
    State("x-num", "value"),
    State("epsilon1", "value"),
    State("epsilon2", "value")
)
def draw_linear_model(btn, name_model, a, b, x1, x2, nums, epsilon1, epsilon2):
    # Kiểm tra các input bắt buộc
    if a is None or b is None or x1 is None or x2 is None:
        return create_fig()
    if x2 - x1 <= 1e-6:
        return create_fig()

    ## Vẽ
    x = np.linspace(x1, x2, nums)
    epsilon = np.random.uniform(low=epsilon1, high=epsilon2, size=nums)
    y = a * x + b + epsilon

    return draw_regularizate_regression(x, y, name_model)


server = app.server
