import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
import io
import base64
import os

app = dash.Dash(__name__)
app.title = "Dashboard Crédit Score"

# Layout du tableau de bord avec des onglets et une taille de police par défaut

app.layout = html.Div(style={'fontSize': '20px'},children=[  # Enlever la taille de police par défaut
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
            html.Div(id='output-data-upload'),  # Retirer style pour la taille de police
            dcc.Dropdown(id='dropdown-client', placeholder='Sélectionnez un client'),  # Retirer style pour la taille de police
            # Enlever le label pour la taille de police
        ]),
        dcc.Tab(label='2. Crédit Score', children=[
            dcc.Graph(id='gauge-score'),
            html.Div(id='resultat-score', style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            })  
        ]),
        dcc.Tab(label='3. Information générale', children=[
            dcc.Dropdown(id='dropdown-variables', multi=True, placeholder='Sélectionnez les variables'),  # Retirer style pour la taille de police
            html.Div(id='table-container', style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            }),  
            dash_table.DataTable(id='table-info-client')  # Retirer style pour la taille de police
        ]),
        dcc.Tab(label='4. Features Importances', children=[
            dcc.Graph(id='global-feature-importance'),
            dcc.Graph(id='local-feature-importance'),
            html.Div(id='top-features')
        ]),
        dcc.Tab(label='5. Comparaison clients', children=[
            dcc.Dropdown(id='dropdown-clients-comparison', multi=True, placeholder='Sélectionnez plusieurs clients'),  # Retirer style pour la taille de police
            dcc.Dropdown(id='dropdown-variables-comparison', multi=True, placeholder='Sélectionnez des variables'),  # Retirer style pour la taille de police
            dcc.Graph(id='histogram-comparison')
        ]),
        dcc.Tab(label='6. Analyse bivariée', children=[
            dcc.Dropdown(id='dropdown-x', placeholder='Sélectionnez la variable X'),  # Retirer style pour la taille de police
            dcc.Dropdown(id='dropdown-y', placeholder='Sélectionnez la variable Y'),  # Retirer style pour la taille de police
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
    Output('dropdown-clients-comparison', 'options'),
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

@app.callback(
    Output('gauge-score', 'figure'),
    Output('resultat-score', 'children'),
    Input('dropdown-client', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_gauge(client_id, content, filename):
    # Vérifiez si le client_id ou le contenu sont None
    if client_id is None or content is None:
        return {}, "Veuillez sélectionner un client et télécharger des données."
    
    df = parse_contents(content, filename)
    
    # Vérifiez si les données du client sont disponibles
    client_data = df[df['SK_ID_CURR'] == client_id]
    
    if client_data.empty:
        return {}, "Client non trouvé dans les données."
    
    # Filtrer les données du client et obtenir les caractéristiques
    client_data_filtered = client_data.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    features = client_data_filtered.to_dict(orient='records')[0]
    
    try:
        # Appel à l'API pour obtenir le score de crédit
        response = requests.post('http://127.0.0.1:8000/predict', json=features)
        
        if response.status_code != 200:
            return {}, f"Erreur lors de la récupération des données : {response.status_code}"
        
        data = response.json()
        prob = data.get('probability')
        label = data.get('class')
        
        if prob is None or label is None:
            return {}, "Erreur: Données manquantes dans la réponse de l'API."
        
        # Définir les couleurs en fonction de l'étiquette
        bar_color = "green" if label == "Accepted" else "red"
        
        # Créer le graphique de jauge
        figure = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Score Crédit", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 2},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [0, 0.53], 'color': 'rgba(255,0,0,0.2)', 'name': 'Refusé'},
                    {'range': [0.53, 1], 'color': 'rgba(0,255,0,0.2)', 'name': 'Accepté'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4}, 
                    'thickness': 0.75, 
                    'value': 0.53
                }
            }
        ))

        # Créer l'icône à afficher à côté du résultat
        icon = "✔️" if label == "Accepted" else "❌"
        icon_color = "green" if label == "Accepted" else "red"

        # Renvoie le graphique et le résultat avec l'icône
        result = html.Div(children=[
            html.Span(icon, style={'fontSize': '1.5em', 'color': icon_color, 'marginRight': '10px'}),
            html.Span(f"Résultat : {label}", style={'fontSize': '1.2em'})
        ])
        
        figure.update_layout(title="Jauge du score de crédit, indiquant un score de 0 à 1 pour la probabilité d'acceptation du crédit") 
        return figure, result
    
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
    
    if client_data.empty:
        return []

    return client_data[variables].to_dict(orient='records')

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
                            title="Importance des features globales",
                            labels={'index': 'Features', 'value': 'Importance'},
                            color='value', color_continuous_scale=px.colors.sequential.Viridis)

        # Appel API pour récupérer les importances locales
        client_data = df[df['SK_ID_CURR'] == client_id].drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
        local_importance_series = pd.Series(client_data.values[0], index=client_data.columns)
        top_local_features = local_importance_series.nlargest(3).index.tolist()
        
        fig_local = px.bar(local_importance_series.sort_values(ascending=False), 
                           title=f"Importance des features locales",
                           labels={'index': 'Features', 'value': 'Valeurs'},
                           color='value', color_continuous_scale=px.colors.sequential.Viridis)
        
        return fig_global, fig_local, f"Top 3 features globales : {', '.join(top_global_features)}\nTop 3 features locales : {', '.join(top_local_features)}"
    
    except Exception as e:
        return {}, {}, f"Erreur lors de l'appel à l'API pour les importances des features : {str(e)}"

# Callback pour afficher la comparaison entre les clients
@app.callback(
    Output('histogram-comparison', 'figure'),
    Input('dropdown-clients-comparison', 'value'),
    Input('dropdown-variables-comparison', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_histogram(clients, variables, content, filename):
    if clients is None or variables is None or content is None:
        return {}

    df = parse_contents(content, filename)

    filtered_df = df[df['SK_ID_CURR'].isin(clients)]

    if filtered_df.empty:
        return {}

    # Création de l'histogramme avec mode de barres 'group'
    fig = px.histogram(
        filtered_df,
        x=variables[0],
        color='SK_ID_CURR',
        title="Comparaison des clients",
        barmode='group'  # Utilisation de 'group' pour séparer les barres
    )

    # Simplification de l'axe des x sans valeurs spécifiques
    fig.update_xaxes(title_text=variables[0], showticklabels=False, ticks="", title_font=dict(size=16))

    # Mise à jour de la mise en page pour éviter le chevauchement
    fig.update_layout(xaxis_title_text=variables[0], barmode='group')  # Assurez-vous que le mode de barre est correct

    return fig

# Callback pour l'analyse bivariée
# Callback pour l'analyse bivariée
@app.callback(
    Output('bivariate-analysis', 'figure'),
    Input('dropdown-x', 'value'),
    Input('dropdown-y', 'value'),
    Input('dropdown-client', 'value'),  # Ajout du client sélectionné
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_bivariate_analysis(x_var, y_var, client_id, content, filename):
    if x_var is None or y_var is None or content is None:
        return {}
    
    df = parse_contents(content, filename)
    
    # Ajouter une colonne pour indiquer si chaque ligne est le client sélectionné
    df['client intérêt'] = df['SK_ID_CURR'] == client_id
    
    fig = px.scatter(
        df, 
        x=x_var, 
        y=y_var, 
        color='client intérêt',  # Utiliser la colonne pour la couleur
        title=f"Analyse bivariée entre {x_var} et {y_var}",
        color_discrete_map={True: 'red', False: 'blue'},  # Choisir les couleurs
        labels={'client intérêt': ''}  # Enlever l'étiquette de la légende
    )
    
    # Supprimer la légende
    fig.update_layout(showlegend=False)

    # Ajouter une annotation sous le graphique
    fig.add_annotation(
        text="Client sélectionné en rouge",  # Texte de l'annotation
        xref="paper", yref="paper",  # Références du système de coordonnées
        x=0.5, y=-0.25,  # Position du texte (ajuster y pour descendre sous le graphique)
        showarrow=False,
        font=dict(size=18)  # Taille de la police (ajuster selon vos besoins)
    )
    
    # Ajuster la mise en page pour laisser de l'espace en bas
    fig.update_layout(
        height=500,  # Augmentez la hauteur de la figure
        margin=dict(l=40, r=40, t=40, b=100)  # Ajoute des marges, surtout en bas
    )

    return fig



if __name__ == '__main__':
    app.run_server(port=int(os.environ.get("PORT", 8050)), debug=True)
