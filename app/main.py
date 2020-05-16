import dash
import dash_html_components as html
import dash_core_components as dcc
import flask

import time
import importlib

import numpy as np
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn import datasets
# from sklearn.svm import SVC
from sklearn import gaussian_process as gp
from sklearn.gaussian_process import kernels as sk_kern


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
                                    "Gaussian Process Regression",
                                    style={
                                        # "text-decoration": "none",
                                        "font-family" : "courier",
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
                                                    "label": "Cos * Cos",
                                                    "value": "cos",
                                                },
                                                {
                                                    "label": "log",
                                                    "value": "log",
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
                                        drc.NamedSlider(
                                            name="Frequency",
                                            id="slider-freq",
                                            min=0.1,
                                            max=10,
                                            value=1,
                                            step=0.1,
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
                                                # {
                                                #     "label": "Linear", 
                                                #     "value": "linear"},
                                                # {
                                                #     "label": "Polynomial",
                                                #     "value": "poly",
                                                # },
                                                # {
                                                #     "label": "Sigmoid",
                                                #     "value": "sigmoid",
                                                # },
                                            ],
                                            value="rbf",
                                            clearable=False,
                                            searchable=False,
                                        ),

                                        drc.NamedSlider(
                                            name="Length",
                                            id="slider-svm-parameter-kernel-length",
                                            min=0.1,
                                            max=10,
                                            value=0.5,
                                            marks={
                                                i: f"{i}" for i in range(10)
                                            },
                                            step=0.1,
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
  ])
# @app.callback(
#     Output("slider-svm-parameter-gamma-coef", "marks"),
#     [Input("slider-svm-parameter-gamma-power", "value")],
# )
# def update_slider_svm_parameter_gamma_coef(power):
#     scale = 10 ** power
#     return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


# @app.callback(
#     Output("slider-svm-parameter-C-coef", "marks"),
#     [Input("slider-svm-parameter-C-power", "value")],
# )
# def update_slider_svm_parameter_C_coef(power):
#     scale = 10 ** power
#     return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


# @app.callback(
#     Output("slider-phase", "value"),
#     [Input("button-zero-threshold", "n_clicks")],
#     [State("graph-sklearn-svm", "figure")],
# )
# def reset_threshold_center(n_clicks, figure):
#     if n_clicks:
#         Z = np.array(figure["data"][0]["z"])
#         value = -Z.min() / (Z.max() - Z.min())
#     else:
#         value = 0.4959986285375595
#     return value


# Disable Sliders if kernel not in the given list
@app.callback(
    Output("slider-svm-parameter-degree", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_degree(kernel):
    return kernel != "poly"


# @app.callback(
#     Output("slider-svm-parameter-gamma-coef", "disabled"),
#     [Input("dropdown-svm-parameter-kernel", "value")],
# )
# def disable_slider_param_gamma_coef(kernel):
#     return kernel not in ["rbf", "poly", "sigmoid"]


# @app.callback(
#     Output("slider-svm-parameter-gamma-power", "disabled"),
#     [Input("dropdown-svm-parameter-kernel", "value")],
# )
# def disable_slider_param_gamma_power(kernel):
#     return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-svm-parameter-kernel", "value"),
        Input("dropdown-select-dataset", "value"),
        Input("slider-dataset-noise-level", "value"),
        Input("slider-freq", "value"),
        Input("slider-svm-parameter-kernel-length", "value"),
        Input("slider-dataset-sample-size", "value"),
    ],
)
def update_svm_graph(
    kernel,
    dataset,
    noise,
    freq,
    length,
    sample_size,
):
    t_start = time.time()
    h = 0.3  # step size in the mesh
    xmax = 5
    x_train = np.random.rand(sample_size) * xmax
    x_test = np.linspace(0,xmax,300)
    x_true = np.linspace(0,xmax,300)

    if dataset == "sin":
        y_train = np.sin(x_train * 2* np.pi * freq / xmax) + np.random.normal(0, noise, len(x_train))
        y_true = np.sin(x_true* 2 * np.pi * freq / xmax)
    elif dataset == "cos":
        y_train = np.cos(x_train * 2* np.pi * freq / xmax) * 2.0 * np.sin(x_train * 10 * np.pi * freq / xmax) + np.random.normal(0, noise, len(x_train))
        y_true = np.cos(x_true * 2* np.pi * freq / xmax) * 2.0 * np.sin(x_true * 10 * np.pi * freq / xmax)
    elif dataset == "log":
        y_train = np.log(x_train) + np.random.normal(0, noise, len(x_train))
        y_true = np.log(x_true) 
        
    else:
        exit()
        
    traces = []

    input_fig=go.Scatter(x=x_train, y=y_train, hovertext=[], mode='markers+text',textposition="bottom center",hoverinfo="text", name="observation", marker={'size': 10, "color": "salmon"})
    traces.append(input_fig)

    gt_fig=go.Scatter(x=x_true, y=y_true,mode='lines',  marker={"color": "salmon"}, name='ground truth')
    traces.append(gt_fig)

    kern = sk_kern.ExpSineSquared() * sk_kern.RBF()
    kernel = sk_kern.RBF(length, (1e-3, 1e3)) + sk_kern.ConstantKernel(length, (1e-3, 1e3)) + sk_kern.WhiteKernel()
    
    clf = gp.GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10, 
        optimizer="fmin_l_bfgs_b", 
        n_restarts_optimizer=30,
        normalize_y=True)

    clf.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
        
    pred_mean, pred_std= clf.predict(x_test.reshape(-1,1), return_std=True)
    pred_mean_fig = go.Scatter(x=x_test, y=pred_mean[:,0], mode='lines', marker=dict(color="darkturquoise"), name='pred-mean')
    traces.append(pred_mean_fig)    
    
    pred_ustd_fig = go.Scatter(x=x_test, y=(pred_mean + 3*pred_std)[:,0], mode='lines', line=dict(color="#888888", dash="dash"), name='pred-3-std')
    pred_lstd_fig = go.Scatter(x=x_test, y=(pred_mean - 3*pred_std)[:,0], mode='lines', line=dict(color="#888888", dash="dash"), name='')
    traces.append(pred_ustd_fig)    
    traces.append(pred_lstd_fig)    
    
    
    figure={
        "data": traces,
        "layout": go.Layout(title='GPR', showlegend=True, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 100},
                            clickmode='event+select',
                            plot_bgcolor="#282b38", 
                            paper_bgcolor="#282b38",
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