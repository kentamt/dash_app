import dash
import dash_html_components as html
import dash_core_components as dcc
import flask

import time
import importlib

import numpy as np
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import GPy
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn import datasets
# from sklearn.svm import SVC

import utils.dash_reusable_components as drc
import utils.figures as figs

server = flask.Flask(__name__)

@server.route('/')
def index():
    return '''
<html>
<div>
    <h1>Flask App</h1>
    <a href='dash'>go to dash app</a>
</div>
</html>
'''

app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dash/'
)

app.layout = html.Div(
    children=[
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Gaussian Process Regression (GPR)",
                                    href="https://github.com/plotly/dash-svm",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                            id="banner-logo",
                            children=[
                                html.Img(src=app.get_asset_url("dash-logo-new.png"))
                            ],
                            href="https://plot.ly/products/dash/",
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options=[
                                                {
                                                    "label": "Sin", 
                                                    "value": "sin"
                                                },
                                                {
                                                    "label": "Cos",
                                                    "value": "cos",
                                                },
                                                {
                                                    "label": "Circles",
                                                    "value": "circles",
                                                },
                                            ],
                                            clearable=False,
                                            searchable=False,
                                            value="sin",
                                        ),
                                        drc.NamedSlider(
                                            name="Sample Size",
                                            id="slider-dataset-sample-size",
                                            min=10,
                                            max=50,
                                            step=10,
                                            marks={
                                                str(i): str(i)
                                                for i in [10, 20, 30, 40, 50]
                                            },
                                            value=30,
                                        ),
                                        drc.NamedSlider(
                                            name="Noise Level",
                                            id="slider-dataset-noise-level",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.2,
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                        html.Button(
                                            "Reset Threshold",
                                            id="button-zero-threshold",
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="last-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Kernel",
                                            id="dropdown-svm-parameter-kernel",
                                            options=[
                                                {
                                                    "label": "Radial basis function (RBF)",
                                                    "value": "rbf",
                                                },
                                                {"label": "Linear", "value": "linear"},
                                                {
                                                    "label": "Polynomial",
                                                    "value": "poly",
                                                },
                                                {
                                                    "label": "Sigmoid",
                                                    "value": "sigmoid",
                                                },
                                            ],
                                            value="rbf",
                                            clearable=False,
                                            searchable=False,
                                        ),
                                        drc.NamedSlider(
                                            name="Cost (C)",
                                            id="slider-svm-parameter-C-power",
                                            min=-2,
                                            max=4,
                                            value=0,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-2, 5)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-C-coef",
                                            min=1,
                                            max=9,
                                            value=1,
                                        ),
                                        drc.NamedSlider(
                                            name="Degree",
                                            id="slider-svm-parameter-degree",
                                            min=2,
                                            max=10,
                                            value=3,
                                            step=1,
                                            marks={
                                                str(i): str(i) for i in range(2, 11, 2)
                                            },
                                        ),
                                        drc.NamedSlider(
                                            name="Gamma",
                                            id="slider-svm-parameter-gamma-power",
                                            min=-5,
                                            max=0,
                                            value=-1,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-5, 1)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-gamma-coef",
                                            min=1,
                                            max=9,
                                            value=5,
                                        ),
                                        html.Div(
                                            id="shrinking-container",
                                            children=[
                                                html.P(children="Shrinking"),
                                                dcc.RadioItems(
                                                    id="radio-svm-parameter-shrinking",
                                                    labelStyle={
                                                        "margin-right": "7px",
                                                        "display": "inline-block",
                                                    },
                                                    options=[
                                                        {
                                                            "label": " Enabled",
                                                            "value": "True",
                                                        },
                                                        {
                                                            "label": " Disabled",
                                                            "value": "False",
                                                        },
                                                    ],
                                                    value="True",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
        dcc.Graph(
                id='example-graph',
                figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                }
                }
            )
  ])
@app.callback(
    Output("slider-svm-parameter-gamma-coef", "marks"),
    [Input("slider-svm-parameter-gamma-power", "value")],
)
def update_slider_svm_parameter_gamma_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-svm-parameter-C-coef", "marks"),
    [Input("slider-svm-parameter-C-power", "value")],
)
def update_slider_svm_parameter_C_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
    [State("graph-sklearn-svm", "figure")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks:
        Z = np.array(figure["data"][0]["z"])
        value = -Z.min() / (Z.max() - Z.min())
    else:
        value = 0.4959986285375595
    return value


# Disable Sliders if kernel not in the given list
@app.callback(
    Output("slider-svm-parameter-degree", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_degree(kernel):
    return kernel != "poly"


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_coef(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("slider-svm-parameter-gamma-power", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_power(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-svm-parameter-kernel", "value"),
        Input("slider-svm-parameter-degree", "value"),
        Input("slider-svm-parameter-C-coef", "value"),
        Input("slider-svm-parameter-C-power", "value"),
        Input("slider-svm-parameter-gamma-coef", "value"),
        Input("slider-svm-parameter-gamma-power", "value"),
        Input("dropdown-select-dataset", "value"),
        Input("slider-dataset-noise-level", "value"),
        Input("radio-svm-parameter-shrinking", "value"),
        Input("slider-threshold", "value"),
        Input("slider-dataset-sample-size", "value"),
    ],
)
def update_svm_graph(
    kernel,
    degree,
    C_coef,
    C_power,
    gamma_coef,
    gamma_power,
    dataset,
    noise,
    shrinking,
    threshold,
    sample_size,
):
    t_start = time.time()
    h = 0.3  # step size in the mesh

    x_train = np.random.randint(0, 100, sample_size)
    x_test = np.linspace(0,100,100)
    x_true = np.linspace(0,100,100)


    if dataset == "sin":
        y_train = np.sin(x_train / 100 * 2* np.pi) + np.random.normal(0, noise, len(x_train))
        y_true = np.sin(x_true / 100 * 2* np.pi)
    elif dataset == "cos":
        y_train = np.cos(x_train / 100 * 2* np.pi) + np.random.normal(0, noise, len(x_train))
        y_true = np.cos(x_true / 100 * 2* np.pi)
    else:
        pass

    traces = []

    input_fig=go.Scatter(x=x_train, y=y_train, hovertext=[], mode='markers+text',textposition="bottom center",hoverinfo="text", marker={'size': 10, "color": "blue"})
    traces.append(input_fig)

    gt_fig=go.Scatter(x=x_true, y=y_true,mode='lines',  marker={"color": "blue"}, name='gt')
    traces.append(gt_fig)

    from sklearn import gaussian_process as gp
    from sklearn.gaussian_process import kernels as sk_kern
    kern = sk_kern.ExpSineSquared() * sk_kern.RBF()
    kernel = sk_kern.RBF(1.0, (1e-3, 1e3)) + sk_kern.ConstantKernel(1.0, (1e-3, 1e3)) + sk_kern.WhiteKernel()
    clf = gp.GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10, 
        optimizer="fmin_l_bfgs_b", 
        n_restarts_optimizer=20,
        normalize_y=True)

    clf.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    # clf.kernel_ # < RBF(length_scale=0.374) + 0.0316**2 + WhiteKernel(noise_level=0.00785)
    
    pred_mean, pred_std= clf.predict(x_test.reshape(-1,1), return_std=True)
    pred_mean_fig = go.Scatter(x=x_test, y=pred_mean[:,0], mode='lines', marker=dict(color="darkturquoise"), name='pred-mean')
    traces.append(pred_mean_fig)    
    
    pred_ustd_fig = go.Scatter(x=x_test, y=(pred_mean + 3*pred_std)[:,0], mode='lines', line=dict(color="#888888", dash="dash"), name='pred-3-std')
    pred_lstd_fig = go.Scatter(x=x_test, y=(pred_mean - 3*pred_std)[:,0], mode='lines', line=dict(color="#888888", dash="dash"), name='pred-3-std')
    traces.append(pred_ustd_fig)    
    traces.append(pred_lstd_fig)    
    
    
    figure={
        "data": traces,
        "layout": go.Layout(title='GPR', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 100},
                            clickmode='event+select',
                            plot_bgcolor="#282b38", 
                            paper_bgcolor="#282b38"
                            )
        }

    return [
        html.Div(
            id="svm-graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="graph-sklearn-svm", figure=figure),
                style={"display": "none"},
            ),
        ),
        # html.Div(
        #     id="graphs-container",
        #     children=[
        #         dcc.Loading(
        #             className="graph-wrapper",
        #             children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
        #         ),
        #         dcc.Loading(
        #             className="graph-wrapper",
        #             children=dcc.Graph(
        #                 id="graph-pie-confusion-matrix", figure=confusion_figure
        #             ),
        #         ),
        #     ],
        # ),
    ]



if __name__ == '__main__':
    server.run()# host='0.0.0.0', debug=True, port=80)