import base64
import io
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import logging
from fastapi import FastAPI, HTTPException
import joblib

# Initialisation de l'application Dash
app = Dash(__name__)

# Chargement du pipeline
try:
    pipeline = joblib.load("pipeline.joblib")
    logging.info("Modèle chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")
    raise

# Layout de l'application
app.layout = html.Div(style={'backgroundColor': '#2a2a2a', 'color': '#FFFFFF'}, children=[
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV', style={'margin': '10px'}),
        multiple=False
    ),
    dcc.Tabs([
        dcc.Tab(label='Client Info', children=[
            dcc.Dropdown(id='client-dropdown'),
            html.Div(id='client-info', style={'margin': '10px'}),
            dcc.Graph(id='acceptance-probability', style={'height': '400px'})
        ]),
        dcc.Tab(label='Feature Comparison', children=[
            dcc.Dropdown(id='comparison-client-dropdown', multi=True),
            dcc.Dropdown(id='comparison-feature-dropdown', multi=True),
            dcc.Graph(id='comparison-graph', style={'height': '400px'})
        ]),
        dcc.Tab(label='Feature Importances', children=[
            dcc.Graph(id='feature-importances-graph', style={'height': '400px'}),
        ]),
        dcc.Tab(label='Bivariate Analysis', children=[
            dcc.Dropdown(id='feature1-dropdown'),
            dcc.Dropdown(id='feature2-dropdown'),
            dcc.Graph(id='bivariate-graph', style={'height': '400px'}),
        ])
    ])
])

# Fonction pour extraire les caractéristiques du DataFrame
def get_feature_importances(df):
    logistic_model = pipeline.named_steps['classifier']
    feature_importances = logistic_model.coef_[0]  # Récupérer les coefficients du modèle
    feature_importances_dict = dict(zip(df.columns, feature_importances))
    return feature_importances_dict

# Callback pour charger les données et remplir le dropdown des clients
@app.callback(
    Output('client-dropdown', 'options'),
    Output('comparison-client-dropdown', 'options'),
    Output('feature1-dropdown', 'options'),
    Output('feature2-dropdown', 'options'),
    Input('upload-data', 'contents')
)
def load_clients(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Lire le CSV
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier CSV : {e}")
            return [], [], [], []

        # Vérifier si le DataFrame est vide
        if df.empty:
            logging.warning("Le fichier CSV est vide.")
            return [], [], [], []

        # Vérifier si SK_ID_CURR existe
        if 'SK_ID_CURR' not in df.columns:
            logging.error("La colonne SK_ID_CURR n'existe pas dans le fichier CSV.")
            return [], [], [], []

        client_options = [{'label': str(client_id), 'value': client_id} for client_id in df['SK_ID_CURR']]
        
        feature_options = [{'label': col, 'value': col} for col in df.columns if col != 'SK_ID_CURR']

        return (
            client_options,
            client_options,
            feature_options,
            feature_options
        )

    return [], [], [], []  # Retourner un message si pas de données

# Callback pour afficher les informations du client sélectionné
@app.callback(
    Output('client-info', 'children'),
    Output('acceptance-probability', 'figure'),
    Input('client-dropdown', 'value'),
    Input('upload-data', 'contents')
)
def display_client_info(selected_client, contents):
    if contents is None or selected_client is None:
        return "Veuillez télécharger un fichier et sélectionner un client.", go.Figure()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Vérifier si le client sélectionné est dans le DataFrame
    if selected_client not in df['SK_ID_CURR'].values:
        return f"Client ID {selected_client} non trouvé.", go.Figure()

    # Extraire les informations du client
    client_info = df[df['SK_ID_CURR'] == selected_client].iloc[0].to_dict()
    info_display = [html.P(f"{key}: {value}") for key, value in client_info.items()]

    # Calculer la probabilité d'acceptation
    features = {col: client_info[col] for col in df.columns if col != 'SK_ID_CURR'}
    X = pd.DataFrame([features])
    prob = pipeline.predict_proba(X)[:, 1][0]  # Probabilité de la classe 1

    # Créer une figure de jauge pour la probabilité
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        title={'text': "Probabilité d'Acceptation"},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "red"},
               'steps': [
                   {'range': [0, 0.53], 'color': "green"},
                   {'range': [0.53, 1], 'color': "red"}],
               'threshold': {'line': {'color': "black", 'width': 4}, 'value': 0.53}}))

    return info_display, fig

# Callback pour la comparaison entre plusieurs clients
@app.callback(
    Output('comparison-graph', 'figure'),
    Input('comparison-client-dropdown', 'value'),
    Input('comparison-feature-dropdown', 'value'),
    Input('upload-data', 'contents')
)
def update_comparison_graph(selected_clients, selected_features, contents):
    if contents is None or selected_clients is None or selected_features is None:
        return go.Figure()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    filtered_df = df[df['SK_ID_CURR'].isin(selected_clients)][['SK_ID_CURR'] + selected_features]
    filtered_df = filtered_df.melt(id_vars=['SK_ID_CURR'], var_name='Feature', value_name='Value')

    fig = px.bar(filtered_df, x='SK_ID_CURR', y='Value', color='Feature', barmode='group')
    fig.update_layout(title='Comparaison des caractéristiques des clients')

    return fig

# Callback pour afficher les importances des caractéristiques
@app.callback(
    Output('feature-importances-graph', 'figure'),
    Input('upload-data', 'contents')
)
def display_feature_importances(contents):
    if contents is None:
        return go.Figure()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    feature_importances_dict = get_feature_importances(df)
    features = list(feature_importances_dict.keys())
    importances = list(feature_importances_dict.values())

    fig = go.Figure(data=[go.Bar(x=features, y=importances)])
    fig.update_layout(title='Importances des Caractéristiques', xaxis_title='Caractéristiques', yaxis_title='Importance')

    return fig

# Callback pour l'analyse bivariée
@app.callback(
    Output('bivariate-graph', 'figure'),
    Input('feature1-dropdown', 'value'),
    Input('feature2-dropdown', 'value'),
    Input('upload-data', 'contents'),
    Input('client-dropdown', 'value')
)
def update_bivariate_graph(feature1, feature2, contents, selected_client):
    if contents is None or feature1 is None or feature2 is None:
        return go.Figure()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    fig = px.scatter(df, x=feature1, y=feature2, title='Analyse Bivariée')
    
    if selected_client is not None and selected_client in df['SK_ID_CURR'].values:
        client_data = df[df['SK_ID_CURR'] == selected_client]
        fig.add_trace(go.Scatter(
            x=client_data[feature1],
            y=client_data[feature2],
            mode='markers+text',
            text=[str(selected_client)],
            textposition="top center",
            marker=dict(color='red', size=10)
        ))

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
