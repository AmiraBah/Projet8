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
            dash_table.DataTable(id='table-info-client')  
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
@app.callback(
    Output('output-data-upload', 'children'),
    Output('dropdown-client', 'options'),
    Output('dropdown-variables', 'options'),  
    Output('dropdown-variables-comparison', 'options'),  
    Output('dropdown-x', 'options'),  
    Output('dropdown-y', 'options'),  
    Output('dropdown-clients-comparison', 'options'),  # Ajout des options pour clients comparaison
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(content, filename):
    if content is None:
        return "Veuillez télécharger un fichier", [], [], [], [], [], []
    
    df = parse_contents(content, filename)
    
    if df is None:
        return "Erreur lors du chargement des données.", [], [], [], [], [], []
    
    # Vérification de la colonne SK_ID_CURR
    if 'SK_ID_CURR' not in df.columns:
        return "Erreur : La colonne 'SK_ID_CURR' n'est pas présente dans les données.", [], [], [], [], [], []
    
    # Créer une liste d'options pour le dropdown des clients
    client_options = [{'label': str(client_id), 'value': client_id} for client_id in df['SK_ID_CURR'].unique() if pd.notnull(client_id)]
    
    # Créer une liste d'options pour les variables
    variable_options = [{'label': col, 'value': col} for col in df.columns if col not in ['TARGET', 'SK_ID_CURR']]
    
    return f"Fichier chargé avec succès: {filename}", client_options, variable_options, variable_options, variable_options, variable_options, client_options

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
                            title="Importance des Features Globales",
                            labels={'index': 'Features', 'value': 'Importance'},
                            color='value', color_continuous_scale=px.colors.sequential.Viridis)

        # Appel API pour récupérer les importances locales
        client_data = df[df['SK_ID_CURR'] == client_id].drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
        local_importance_series = pd.Series(client_data.values[0], index=client_data.columns)
        top_local_features = local_importance_series.nlargest(3).index.tolist()
        
        fig_local = px.bar(local_importance_series.sort_values(ascending=False), 
                           title=f"Importance des Features Locales pour le Client {client_id}",
                           labels={'index': 'Features', 'value': 'Valeurs'},
                           color='value', color_continuous_scale=px.colors.sequential.Viridis)
        
        return fig_global, fig_local, f"Top 3 des Features Globales : {', '.join(top_global_features)}"
    
    except Exception as e:
        return {}, {}, f"Erreur lors de l'appel à l'API pour les importances des features : {str(e)}"

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
    
    # Vérifier si les clients sélectionnés existent dans les données
    if not all(client in df['SK_ID_CURR'].values for client in clients):
        return {}
    
    # Filtrer les données pour les clients sélectionnés et les variables choisies
    filtered_data = df[df['SK_ID_CURR'].isin(clients)][['SK_ID_CURR'] + variables]
    
    # Vérifiez si les variables sélectionnées sont dans les colonnes du DataFrame
    missing_columns = [var for var in variables if var not in df.columns]
    if missing_columns:
        return {}

    # Créer le graphique
    fig = px.histogram(filtered_data.melt(id_vars='SK_ID_CURR'), 
                       x='variable', y='value', color='SK_ID_CURR', barmode='group',
                       title="Comparaison des Clients")
    return fig

# Callback pour l'analyse bivariée avec mise en avant du client
@app.callback(
    Output('bivariate-analysis', 'figure'),
    Input('dropdown-client', 'value'),  # Le client étudié
    Input('dropdown-x', 'value'),
    Input('dropdown-y', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_bivariate_analysis(client_id, x_var, y_var, content, filename):
    if not x_var or not y_var or content is None:
        return {}
    
    df = parse_contents(content, filename)
    
    # Vérifier si les variables sont dans le DataFrame
    if x_var not in df.columns or y_var not in df.columns:
        return {}
    
    # Création du graphique de base pour tous les clients
    fig = px.scatter(df, x=x_var, y=y_var, title=f"Analyse Bivariée entre {x_var} et {y_var}",
                     labels={x_var: x_var, y_var: y_var}, opacity=0.5)

    # Mettre en évidence le client sélectionné
    if client_id is not None and client_id in df['SK_ID_CURR'].values:
        client_data = df[df['SK_ID_CURR'] == client_id]
        
        # Ajouter un point distinct pour le client sélectionné
        fig.add_trace(
            go.Scatter(
                x=client_data[x_var],
                y=client_data[y_var],
                mode='markers',
                marker=dict(size=12, color='red', symbol='circle'),
                name=f'Client {client_id}',  # Légende pour le client étudié
                showlegend=True
            )
        )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
