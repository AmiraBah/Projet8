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
app.title = "Dashboard Crédit score"


app.layout = html.Div(style={'fontSize': '22px'},children=[  
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
        dcc.Tab(label='2. Crédit score', children=[
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
        dcc.Tab(label='4. Features importances', children=[
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



#_________________________________________________________________________________________________________
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




#_________________________________________________________________________________________________________
# Callback pour le score
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
        response = requests.post('https://scoring-api-y5hk.onrender.com/predict', json=features)
        
        if response.status_code != 200:
            return {}, f"Erreur lors de la récupération des données : {response.status_code}"
        
        data = response.json()
        prob = data.get('probability')
        label = data.get('class')
        
        if prob is None or label is None:
            return {}, "Erreur: Données manquantes dans la réponse de l'API."
        
        # Définir les couleurs en fonction de l'étiquette
        bar_color = "#228B22" if label == "Accepted" else "#B22222"
        
        # Créer le graphique de jauge
        figure = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Score crédit", 'font': {'size': 30, 'weight': 'bold', 'color': 'black'}, 'align': 'center'},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 2},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [0, 0.53], 'color': 'rgba(178,34,34,0.2)', 'name': 'Refusé'},
                    {'range': [0.53, 1], 'color': 'rgba(34,139,34,0.2)', 'name': 'Accepté'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4}, 
                    'thickness': 0.75, 
                    'value': 0.53
                }
            }
        ))

        # Ajouter une annotation pour indiquer la valeur seuil de 0.53
        figure.add_annotation(
            x=0.5,  # Position en X autour du centre de la jauge
            y=0.65,  # Position en Y pour situer le texte sous la jauge
            text="Seuil : 0.53",  # Texte de l'annotation
            showarrow=False,
            font=dict(size=14, color="black")
        )

        # Créer l'icône à afficher à côté du résultat
        icon = "✔️" if label == "Accepted" else "❌"
        
        return figure, f"{icon} {label.capitalize()} avec une probabilité de {prob:.2%}"
    
    except Exception as e:
        return {}, f"Erreur lors de la récupération des données : {str(e)}"




#_________________________________________________________________________________________________________
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

    return client_data[variables].round(3).to_dict(orient='records')



#_________________________________________________________________________________________________________
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
        response = requests.get('https://scoring-api-y5hk.onrender.com/feature_importances')
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

        # Personnaliser le graphique global
        fig_global.update_layout(
            title={'text': "Importance des features globales", 'font': {'size': 24, 'weight': 'bold'}, 'x': 0.5},
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            legend_font=dict(size=14)
        )

        # Appel API pour récupérer les importances locales
        client_data = df[df['SK_ID_CURR'] == client_id].drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
        local_importance_series = pd.Series(client_data.values[0], index=client_data.columns)
        top_local_features = local_importance_series.nlargest(3).index.tolist()
        
        fig_local = px.bar(local_importance_series.sort_values(ascending=False), 
                           title=f"Importance des features locales",
                           labels={'index': 'Features', 'value': 'Valeurs'},
                           color='value', color_continuous_scale=px.colors.sequential.Viridis)

        # Personnaliser le graphique local
        fig_local.update_layout(
            title={'text': "Importance des features locales", 'font': {'size': 24, 'weight': 'bold'}, 'x': 0.5},
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            legend_font=dict(size=14)
        )

        return (
    fig_global,
    fig_local,
    dcc.Markdown(f"**Top 3 features globales** : {', '.join(top_global_features)}\n\n**Top 3 features locales** : {', '.join(top_local_features)}")
)
    
    except Exception as e:
        return {}, {}, f"Erreur lors de l'appel à l'API pour les importances des features : {str(e)}"

    

#_________________________________________________________________________________________________________
# Callback pour comparaison entre clients
@app.callback(
    Output('histogram-comparison', 'figure'),
    Input('dropdown-clients-comparison', 'value'),
    Input('dropdown-variables-comparison', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_histogram(clients, variables, content, filename):
    if clients is None or variables is None or content is None or len(variables) == 0:
        return {}  # Retourne un graphique vide si l'une des valeurs est absente ou que 'variables' est vide

    # Parser les données
    df = parse_contents(content, filename)

    # Filtrer les données pour ne garder que les clients sélectionnés
    filtered_df = df[df['SK_ID_CURR'].isin(clients)]

    if filtered_df.empty:
        return {}

    # Création du graphique avec un histogramme de type 'bar'
    import plotly.express as px

# Création du graphique avec un histogramme de type 'bar'
    fig = px.bar(
    filtered_df,
    x='SK_ID_CURR',  # Axe X : identifiant des clients
    y=variables[0],  # Axe Y : variable choisie à comparer
    title=f"Comparaison de la variable '{variables[0]}' pour les clients sélectionnés",
    barmode='group',  # Utilisation de 'group' pour séparer les barres
    color='SK_ID_CURR',  # Colorer selon l'ID client
    pattern_shape='SK_ID_CURR',  # Motifs distincts selon l'ID client
    opacity=0.75  # Définir la transparence des barres
)

# Mise à jour de la mise en page pour centrer les labels sous chaque barre
    fig.update_layout(
    title=dict(
        text=f"Comparaison de la variable '{variables[0]}' pour les clients sélectionnés",
        font=dict(size=20, weight='bold'),  # Titre en gras et taille 20
        x=0.5,  # Centre le titre
        xanchor='center'
    ),
    showlegend=False,  # Désactive la légende
    xaxis=dict(
        title="SK_ID_CURR",  # Titre de l'axe des X
        type='category',  # Définit l'axe X comme catégorique
        categoryorder='array',  # Utilise un ordre spécifique pour les catégories
        categoryarray=filtered_df['SK_ID_CURR'].tolist(),  # Assure l'ordre exact des ID
        tickvals=filtered_df['SK_ID_CURR'].tolist(),  # Valeurs de chaque tick
        ticktext=[str(i) for i in filtered_df['SK_ID_CURR']],  # Texte de chaque tick
        tickfont=dict(size=16),  # Taille de la police des labels
        title_font=dict(size=18),  # Taille du titre de l'axe
        tickangle=0,  # Garder les labels droits
        automargin=True,  # Ajuste les marges automatiquement
        tickmode='array',  # Assure que les ticks sont définis explicitement
    ),
    yaxis=dict(
        title=variables[0],  # Titre de l'axe des Y
        tickfont=dict(size=16),  # Taille de la police des valeurs de l'axe Y
        title_font=dict(size=18),  # Taille du titre de l'axe Y
    ),
    margin=dict(l=40, r=40, t=70, b=100),  # Augmente la marge inférieure pour les labels inclinés
    bargap=0.2  # Espace entre les barres
)



    return fig




#_________________________________________________________________________________________________________
# Callback pour l'analyse bivariée
@app.callback(
    Output('bivariate-analysis', 'figure'),
    Input('dropdown-x', 'value'),
    Input('dropdown-y', 'value'),
    Input('dropdown-client', 'value'),  
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_bivariate_analysis(x_var, y_var, client_id, content, filename):
    if x_var is None or y_var is None or content is None:
        return {}
    
    df = parse_contents(content, filename)
    
    # Ajouter une colonne pour indiquer si chaque ligne est le client sélectionné
    df['client intérêt'] = df['SK_ID_CURR'] == client_id
    
    # Définir les symboles en fonction de la catégorie
    symbol_map = {True: 'circle', False: 'x'}
    
    fig = px.scatter(
        df, 
        x=x_var, 
        y=y_var, 
        color='client intérêt',
        color_discrete_map={True: 'red', False: 'blue'},  
        labels={'client intérêt': ''},
        symbol='client intérêt',
        symbol_map=symbol_map,  # Utiliser un mapping de symboles
    )
    
    # Supprimer la légende
    fig.update_layout(showlegend=False)

    # Ajouter une annotation sous le graphique
    fig.add_annotation(
        text="Client sélectionné en rouge (●), autres clients en bleu (×)",  
        xref="paper", yref="paper",  
        x=0.5, y=-0.25,  
        showarrow=False,
        font=dict(size=18)  
    )
    
    # Ajuster la mise en page pour laisser de l'espace en bas
    fig.update_layout(
        height=500,  
        margin=dict(l=40, r=40, t=40, b=100),
    )

    # Ajouter le titre du graphique
    fig.update_layout(title=dict(
        text=f"Analyse bivariée entre {x_var} et {y_var}",
        font=dict(size=20, color='black', weight='bold'),  # Titre en gras et taille 20
        xanchor='center',
        x=0.5
    ))

    # Augmenter la taille des titres des axes
    fig.update_xaxes(title_text=x_var, title_font=dict(size=18), tickfont=dict(size=16))  # Ajustez les tailles selon vos besoins
    fig.update_yaxes(title_text=y_var, title_font=dict(size=18), tickfont=dict(size=16))  # Ajustez les tailles selon vos besoins

    # Augmenter la taille des marqueurs dans le graphique
    fig.update_traces(marker=dict(size=10))  # Ajustez la taille des marqueurs ici

    return fig







if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 8050)), debug=True)


