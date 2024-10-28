import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
import io
import base64

app = dash.Dash(__name__)

# Layout du tableau de bord avec des onglets
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='1. Chargement des données', children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Glissez et déposez ou ',
                    html.A('Sélectionnez un fichier', href="#")
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                },
                aria_label="Zone de téléchargement de fichier"
            ),
            html.Div(id='output-data-upload'),
            dcc.Dropdown(id='dropdown-client', placeholder='Sélectionnez un client', aria_label="Sélectionnez un client"),
        ]),
        dcc.Tab(label='2. Crédit Score', children=[
            dcc.Graph(id='gauge-score', aria_label="Graphique jauge score de crédit"),
            html.Div(id='resultat-score', style={'fontSize': '18px'})
        ]),
        dcc.Tab(label='3. Information générale', children=[
            dcc.Dropdown(id='dropdown-variables', multi=True, placeholder='Sélectionnez les variables', aria_label="Sélectionnez les variables"),
            dash_table.DataTable(id='table-info-client', style_data={'fontSize': '16px'})  
        ]),
        dcc.Tab(label='4. Features Importances', children=[
            dcc.Graph(id='global-feature-importance', aria_label="Importance des features globales"),
            dcc.Graph(id='local-feature-importance', aria_label="Importance des features locales"),
            html.Div(id='top-features', style={'fontSize': '18px'})
        ]),
        dcc.Tab(label='5. Comparaison clients', children=[
            dcc.Dropdown(id='dropdown-clients-comparison', multi=True, placeholder='Sélectionnez plusieurs clients', aria_label="Sélectionnez plusieurs clients pour comparaison"),
            dcc.Dropdown(id='dropdown-variables-comparison', multi=True, placeholder='Sélectionnez des variables', aria_label="Sélectionnez des variables pour comparaison"),
            dcc.Graph(id='histogram-comparison', aria_label="Histogramme de comparaison des clients")
        ]),
        dcc.Tab(label='6. Analyse bivariée', children=[
            dcc.Dropdown(id='dropdown-x', placeholder='Sélectionnez la variable X', aria_label="Variable X pour analyse bivariée"),
            dcc.Dropdown(id='dropdown-y', placeholder='Sélectionnez la variable Y', aria_label="Variable Y pour analyse bivariée"),
            dcc.Graph(id='bivariate-analysis', aria_label="Analyse bivariée entre deux variables")
        ])
    ])
])

# Parse CSV file content
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None
    except Exception as e:
        return None
    return df

# Callbacks and functions as defined in the initial code go here

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)  # Port compatible avec Render
