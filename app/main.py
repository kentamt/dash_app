import dash
import dash_html_components as html
import dash_core_components as dcc
import flask

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

app.layout = html.Div(children=[
  html.Div(children='''
      Dash: A web application framework for Python.
  '''),

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

if __name__ == '__main__':
    server.run()# host='0.0.0.0', debug=True, port=80)