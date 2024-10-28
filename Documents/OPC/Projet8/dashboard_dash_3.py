import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table  # Mise à jour de l'importation
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
                    html.A('Sélectionnez un fichier')
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                },
            ),
            html.Div(id='output-data-upload'),
            dcc.Dropdown(id='dropdown-client', placeholder='Sélectionnez un client'),
        ]),
        dcc.Tab(label='2. Crédit Score', children=[
            dcc.Graph(id='gauge-score'),
            html.Div(id='resultat-score')
        ]),
        dcc.Tab(label='3. Information générale', children=[
            dcc.Dropdown(id='dropdown-variables', multi=True, placeholder='Sélectionnez les variables'),
            dash_table.DataTable(id='table-info-client')  # Utilisation de dash_table ici
        ]),
        dcc.Tab(label='4. Features Importances', children=[
            dcc.Graph(id='global-feature-importance'),
            dcc.Graph(id='local-feature-importance'),
            html.Div(id='top-features')
        ]),
        dcc.Tab(label='5. Comparaison clients', children=[
            dcc.Dropdown(id='dropdown-clients-comparison', multi=True, placeholder='Sélectionnez plusieurs clients'),
            dcc.Dropdown(id='dropdown-variables-comparison', multi=True, placeholder='Sélectionnez des variables'),
            dcc.Graph(id='histogram-comparison')
        ]),
        dcc.Tab(label='6. Analyse bivariée', children=[
            dcc.Dropdown(id='dropdown-x', placeholder='Sélectionnez la variable X'),
            dcc.Dropdown(id='dropdown-y', placeholder='Sélectionnez la variable Y'),
            dcc.Graph(id='bivariate-analysis')
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

# Callback pour charger et afficher les données
# Callback pour charger et afficher les données
@app.callback(
    Output('output-data-upload', 'children'),
    Output('dropdown-client', 'options'),
    Output('dropdown-variables', 'options'),  # Pour les options de variables
    Output('dropdown-variables-comparison', 'options'),  # Pour les options de variables de comparaison
    Output('dropdown-x', 'options'),  # Pour les options de la variable X
    Output('dropdown-y', 'options'),  # Pour les options de la variable Y
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(content, filename):
    if content is None:
        return "Veuillez télécharger un fichier", [], [], [], [], []
    
    df = parse_contents(content, filename)
    
    if df is None:
        return "Erreur lors du chargement des données.", [], [], [], [], []
    
    # Vérification de la colonne SK_ID_CURR
    if 'SK_ID_CURR' not in df.columns:
        return "Erreur : La colonne 'SK_ID_CURR' n'est pas présente dans les données.", [], [], [], [], []
    
    # Créer une liste d'options pour le dropdown des clients
    client_options = [{'label': str(client_id), 'value': client_id} for client_id in df['SK_ID_CURR'].unique() if pd.notnull(client_id)]
    
    # Créer une liste d'options pour les variables
    variable_options = [{'label': col, 'value': col} for col in df.columns if col not in ['TARGET', 'SK_ID_CURR']]
    
    return f"Fichier chargé avec succès: {filename}", client_options, variable_options, variable_options, variable_options, variable_options

# Callback pour afficher le score de crédit sous forme de jauge
@app.callback(
    Output('gauge-score', 'figure'),
    Output('resultat-score', 'children'),
    Input('dropdown-client', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_gauge(client_id, content, filename):
    if client_id is None or content is None:
        return {}, "Veuillez sélectionner un client et télécharger des données."
    
    df = parse_contents(content, filename)
    
    # Filtrer le client sélectionné
    client_data = df[df['SK_ID_CURR'] == client_id]
    
    if client_data.empty:
        return {}, "Client non trouvé dans les données."
    
    # Filtrer les colonnes pour ne pas envoyer `TARGET` et `SK_ID_CURR` à l'API
    client_data_filtered = client_data.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    
    # Convertir les données en format dictionnaire (attendu par l'API)
    features = client_data_filtered.to_dict(orient='records')[0]
    
    try:
        # Appel à l'API pour récupérer la probabilité
        response = requests.post('http://localhost:8000/predict', json=features)
        
        if response.status_code != 200:
            return {}, f"Erreur lors de la récupération des données : {response.status_code}"
        
        data = response.json()  # Obtenir la réponse JSON de l'API
        prob = data.get('probability')
        label = data.get('class')
        
        if prob is None or label is None:
            return {}, "Erreur: Données manquantes dans la réponse de l'API."
        
        figure = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Score Crédit"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "green" if label == "Accepted" else "red"},
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.53}
            }
        ))
        return figure, f"Résultat : {label}"
    
    except Exception as e:
        return {}, f"Erreur lors de l'appel à l'API : {str(e)}"

# Callback pour afficher les informations générales du client
@app.callback(
    Output('table-info-client', 'data'),
    Input('dropdown-client', 'value'),
    Input('dropdown-variables', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_client_info(client_id, variables, content, filename):
    if client_id is None or content is None or variables is None:
        return []
    
    df = parse_contents(content, filename)
    
    client_data = df[df['SK_ID_CURR'] == client_id]
    
    if client_data.empty or not variables:
        return []
    
    return client_data[variables].to_dict('records')

# Callback pour afficher les features importances globales et locales
@app.callback(
    Output('global-feature-importance', 'figure'),
    Output('local-feature-importance', 'figure'),
    Output('top-features', 'children'),
    Input('dropdown-client', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_feature_importances(client_id, content, filename):
    if client_id is None or content is None:
        return {}, {}, "Sélectionnez un client pour afficher les importances des features."
    
    df = parse_contents(content, filename)
    
    # Appel API pour récupérer les features importances globales
    try:
        response = requests.get('http://localhost:8000/feature_importances')
        if response.status_code != 200:
            return {}, {}, "Erreur lors de la récupération des importances."
        
        global_importances = response.json()
        global_importance_series = pd.Series(global_importances)
        top_global_features = global_importance_series.nlargest(3).index.tolist()
        
        # Visualisation des importances globales
        fig_global = px.bar(global_importance_series.sort_values(ascending=False), 
                            title="Importance des Features Globales")
        
        # Importances locales (simulées pour l'exemple)
        client_data = df[df['SK_ID_CURR'] == client_id].drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
        local_importance_series = pd.Series(client_data.values[0], index=client_data.columns)
        top_local_features = local_importance_series.nlargest(3).index.tolist()
        
        # Visualisation des importances locales
        fig_local = px.bar(local_importance_series.sort_values(ascending=False),
                           title="Importance des Features Locales pour le Client")
        
        return fig_global, fig_local, f"Top 3 Global Features: {', '.join(top_global_features)}; Top 3 Local Features: {', '.join(top_local_features)}"
    
    except Exception as e:
        return {}, {}, f"Erreur lors de l'appel à l'API : {str(e)}"

# Callback pour la comparaison des clients
@app.callback(
    Output('histogram-comparison', 'figure'),
    Input('dropdown-clients-comparison', 'value'),
    Input('dropdown-variables-comparison', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_comparison(clients, variables, content, filename):
    if not clients or not variables or content is None:
        return {}
    
    df = parse_contents(content, filename)
    
    # Filtrer les données pour les clients sélectionnés et les variables choisies
    filtered_data = df[df['SK_ID_CURR'].isin(clients)][['SK_ID_CURR'] + variables]
    
    fig = px.histogram(filtered_data.melt(id_vars='SK_ID_CURR'), 
                       x='variable', y='value', color='SK_ID_CURR', barmode='group',
                       title="Comparaison des Clients")
    return fig

# Callback pour l'analyse bivariée avec nuage de points
@app.callback(
    Output('bivariate-analysis', 'figure'),
    Input('dropdown-x', 'value'),
    Input('dropdown-y', 'value'),
    Input('dropdown-client', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_bivariate_analysis(x_var, y_var, client_id, content, filename):
    if not x_var or not y_var or client_id is None or content is None:
        return {}
    
    df = parse_contents(content, filename)
    
    fig = px.scatter(df, x=x_var, y=y_var, title=f"Analyse Bivariée entre {x_var} et {y_var}")
    
    # Ajouter une annotation pour mettre en évidence le client
    client_data = df[df['SK_ID_CURR'] == client_id]
    if not client_data.empty:
        fig.add_trace(go.Scatter(x=client_data[x_var], y=client_data[y_var],
                                 mode='markers', marker=dict(color='red', size=10),
                                 name=f"Client {client_id}"))
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
