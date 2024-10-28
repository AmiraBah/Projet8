import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import requests

# Fonction pour calculer les valeurs SHAP (explications locales)
def calculate_shap_values(model, client_data, background_data):
    explainer = shap.LinearExplainer(model, background_data, feature_perturbation="interventional")
    shap_values = explainer(client_data)
    return shap_values

# Fonction pour afficher les explications locales des features avec SHAP
def show_local_feature_importance(shap_values, client_data):
    st.write("### Contribution des features à la prédiction (Importance locale)")
    shap.plots.waterfall(shap_values[0], max_display=10)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    st.write("Graphique en cascade représentant la contribution des principales features à la prédiction pour ce client.")

# Fonction pour afficher l'importance des features globales
def show_global_feature_importance(pipeline, X_train):
    st.write("### Importance des features (Globale)")
    model = pipeline.named_steps['classifier']
    feature_names = X_train.columns

    if hasattr(model, 'coef_'):
        importances = model.coef_[0]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Importance des Features Globales")
        st.plotly_chart(fig)
        st.write("Graphique en barres représentant l'importance globale des features dans le modèle.")
    else:
        st.write("Le modèle n'a pas de coefficients (coef_).")

# Fonction pour afficher le score de crédit sous forme de jauge
def show_gauge(probability, threshold=0.53):
    gauge_colors = [
        {"range": [0, threshold * 100], "color": "#69b3a2"},  # Vert pour bleu
        {"range": [threshold * 100, 100], "color": "#ff9933"}  # Rouge pour orange
    ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,  
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
            "steps": gauge_colors
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        title="Score de crédit",
        height=400
    )
    
    st.plotly_chart(fig)
    st.write("Indicateur du score de crédit : un graphique en jauge montrant la probabilité de crédit en pourcentage.")

# Fonction pour analyser deux features en même temps (bi-variée)
def show_bivariate_analysis(data, feature1, feature2, client_data):
    st.write(f"### Analyse bi-variée entre '{feature1}' et '{feature2}'")
    fig = px.scatter(data, x=feature1, y=feature2, title=f"Analyse bi-variée : {feature1} vs {feature2}")
    fig.add_scatter(x=[client_data[feature1]], y=[client_data[feature2]], mode='markers', marker=dict(color='red', size=10), name="Client")
    st.plotly_chart(fig)
    st.write(f"Graphique de dispersion montrant l'analyse bi-variée entre '{feature1}' et '{feature2}' avec le client en surbrillance.")

# Fonction pour aplatir les données avant de les envoyer à l'API
def flatten_data(client_data):
    flattened_data = {key: list(value.values())[0] for key, value in client_data.items()}
    return flattened_data

# Fonction pour appeler l'API et obtenir le score de crédit
def get_credit_score(client_data):
    url = "http://127.0.0.1:8000/predict"  # Remplacez par l'URL de votre API FastAPI

    expected_features = [
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN", "CC_CNT_DRAWINGS_CURRENT_MAX", "BURO_DAYS_CREDIT_MEAN", "CC_AMT_BALANCE_MEAN",
        "DAYS_BIRTH", "PREV_NAME_CONTRACT_STATUS_Refused_MEAN", "BURO_CREDIT_ACTIVE_Active_MEAN", "DAYS_EMPLOYED",
        "REFUSED_DAYS_DECISION_MAX", "CC_AMT_BALANCE_MIN", "ACTIVE_DAYS_CREDIT_MEAN", "CC_CNT_DRAWINGS_ATM_CURRENT_MAX",
        "CC_MONTHS_BALANCE_MEAN", "BURO_STATUS_1_MEAN_MEAN", "CC_CNT_DRAWINGS_ATM_CURRENT_VAR", "REGION_RATING_CLIENT_W_CITY",
        "CC_AMT_DRAWINGS_CURRENT_MEAN", "NAME_INCOME_TYPE_Working", "PREV_NAME_PRODUCT_TYPE_walk_in_MEAN",
        "PREV_CODE_REJECT_REASON_SCOFR_MEAN", "DAYS_LAST_PHONE_CHANGE", "APPROVED_DAYS_DECISION_MIN", "DAYS_ID_PUBLISH",
        "REG_CITY_NOT_WORK_CITY", "REFUSED_HOUR_APPR_PROCESS_START_MIN", "CODE_GENDER", "BURO_STATUS_C_MEAN_MEAN",
        "NAME_EDUCATION_TYPE_Higher_education", "PREV_NAME_CONTRACT_STATUS_Approved_MEAN", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
    ]

    if isinstance(client_data, pd.DataFrame):
        client_data_dict = client_data.squeeze().to_dict()
    else:
        client_data_dict = client_data

    client_data_dict = flatten_data(client_data_dict)

    missing_features = set(expected_features) - set(client_data_dict.keys())
    if missing_features:
        st.error(f"Le fichier est manquant les colonnes suivantes : {', '.join(missing_features)}")
        return None, None
    
    response = requests.post(url, json=client_data_dict)
    st.write(f"Réponse de l'API : {response.text}")

    if response.status_code == 200:
        result = response.json()
        return result['probability'], result['class']
    else:
        st.error(f"Erreur lors de la prédiction : {response.text}")
        return None, None

# Fonction pour modifier les informations du client
def modify_client_data(client_data):
    st.write("### Modification des informations client")
    modified_data = client_data.copy()
    
    for feature in client_data.columns:
        if client_data[feature].dtype == 'int64':
            modified_data[feature] = st.number_input(f"{feature} (actuel : {client_data[feature].values[0]})", 
                                                     value=int(client_data[feature].values[0]))
        elif client_data[feature].dtype == 'float64':
            modified_data[feature] = st.number_input(f"{feature} (actuel : {client_data[feature].values[0]})", 
                                                     value=float(client_data[feature].values[0]))
        else:
            modified_data[feature] = st.text_input(f"{feature} (actuel : {client_data[feature].values[0]})", 
                                                   value=str(client_data[feature].values[0]))

    return modified_data

# Fonction pour comparer plusieurs clients
def compare_clients(data, client_ids, feature_to_compare):
    comparison_data = data[data['SK_ID_CURR'].isin(client_ids)]
    fig = px.histogram(comparison_data, x=feature_to_compare, color='SK_ID_CURR', barmode='overlay',
                       title=f"Comparaison de la feature '{feature_to_compare}' entre plusieurs clients")
    st.plotly_chart(fig)
    st.write("Histogramme représentant la comparaison de la feature sélectionnée entre les clients choisis.")

# Interface principale du dashboard
def main():
    st.title("Dashboard Interactif de Scoring Crédit")

    try:
        pipeline = joblib.load("pipeline.joblib")
        st.success("Modèle chargé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return

    uploaded_file = st.file_uploader("Chargez les données du client", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Données chargées")
        st.dataframe(data)

        client_id = st.selectbox("Sélectionnez un client", data['SK_ID_CURR'])
        client = data[data['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR', 'TARGET'])

        st.write("### Informations du client sélectionné")
        st.markdown("Voici les informations du client sélectionné :")
        st.json(client.to_dict(orient='records')[0])

        # Afficher le score de crédit
        if st.button("Obtenir le score de crédit"):
            probability, label = get_credit_score(client.to_dict())
            if probability is not None:
                st.write(f"Score de crédit : {label} ({probability:.2f})")
                show_gauge(probability)

        # Afficher l'explication locale des features avec SHAP
        if st.button("Voir explication locale SHAP"):
            shap_values = calculate_shap_values(pipeline, client, data.drop(columns=['SK_ID_CURR', 'TARGET']))
            show_local_feature_importance(shap_values, client)

        # Comparaison entre clients
        selected_client_ids = st.multiselect("Sélectionnez plusieurs clients", data['SK_ID_CURR'].unique())
        feature_to_compare = st.selectbox("Sélectionnez une feature à comparer", data.columns)
        if st.button("Comparer les clients"):
            if selected_client_ids and feature_to_compare:
                compare_clients(data, selected_client_ids, feature_to_compare)
            else:
                st.warning("Veuillez sélectionner au moins un client et une feature à comparer.")

        # Modifier les données du client
        modified_client = modify_client_data(client)

        # Recalculer le score avec les nouvelles données
        if st.button("Recalculer le score avec les nouvelles informations"):
            probability, label = get_credit_score(modified_client.to_dict())
            if probability is not None:
                st.write(f"Nouveau score de crédit : {label} ({probability:.2f})")
                show_gauge(probability)

if __name__ == "__main__":
    main()
