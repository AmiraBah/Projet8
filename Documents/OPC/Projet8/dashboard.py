import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LogisticRegression

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
    else:
        st.write("Le modèle n'a pas de coefficients (coef_).")

# Fonction pour comparer l'importance locale et globale des features
def compare_local_global_importance(local_shap_values, global_importances, feature_names):
    st.write("### Comparaison entre l'importance des features locale et globale")

    local_importances = local_shap_values.values[0]

    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance Locale': local_importances,
        'Importance Globale': global_importances
    }).sort_values(by='Importance Globale', ascending=False)

    fig = px.bar(comparison_df, y='Feature', x=['Importance Locale', 'Importance Globale'], barmode='group',
                 title="Comparaison de l'importance des features (locale vs globale)")
    st.plotly_chart(fig)

# Fonction pour afficher le score de crédit sous forme de jauge
def show_gauge(probability, threshold=0.53):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,  
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, threshold * 100], "color": "lightgreen"},
                {"range": [threshold * 100, 100], "color": "red"}
            ]
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        title="Score de crédit",
        height=400
    )
    
    st.plotly_chart(fig)

# Fonction pour afficher la distribution d'une feature sélectionnée
def show_feature_distribution(data, feature, client_value):
    st.write(f"### Distribution de la feature '{feature}' par rapport au client sélectionné")
    fig = px.histogram(data, x=feature, marginal="box", nbins=30, title=f"Distribution de {feature}")
    fig.add_vline(x=client_value, line_dash="dash", line_color="red", annotation_text="Client", annotation_position="top right")
    st.plotly_chart(fig)

# Fonction pour analyser deux features en même temps (bi-variée)
def show_bivariate_analysis(data, feature1, feature2, client_data):
    st.write(f"### Analyse bi-variée entre '{feature1}' et '{feature2}'")
    fig = px.scatter(data, x=feature1, y=feature2, title=f"Analyse bi-variée : {feature1} vs {feature2}")
    fig.add_scatter(x=[client_data[feature1]], y=[client_data[feature2]], mode='markers', marker=dict(color='red', size=10), name="Client")
    st.plotly_chart(fig)

# Fonction pour appeler l'API et obtenir le score de crédit
# Fonction pour appeler l'API et obtenir le score de crédit
def get_credit_score(client_data):
    url = "http://127.0.0.1:8000/predict"  # Remplacez par l'URL de votre API FastAPI

    expected_features = [
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
        "CC_CNT_DRAWINGS_CURRENT_MAX",
        "BURO_DAYS_CREDIT_MEAN",
        "CC_AMT_BALANCE_MEAN",
        "DAYS_BIRTH",
        "PREV_NAME_CONTRACT_STATUS_Refused_MEAN",
        "BURO_CREDIT_ACTIVE_Active_MEAN",
        "DAYS_EMPLOYED",
        "REFUSED_DAYS_DECISION_MAX",
        "CC_AMT_BALANCE_MIN",
        "ACTIVE_DAYS_CREDIT_MEAN",
        "CC_CNT_DRAWINGS_ATM_CURRENT_MAX",
        "CC_MONTHS_BALANCE_MEAN",
        "BURO_STATUS_1_MEAN_MEAN",
        "CC_CNT_DRAWINGS_ATM_CURRENT_VAR",
        "REGION_RATING_CLIENT_W_CITY",
        "CC_AMT_DRAWINGS_CURRENT_MEAN",
        "NAME_INCOME_TYPE_Working",
        "PREV_NAME_PRODUCT_TYPE_walk_in_MEAN",
        "PREV_CODE_REJECT_REASON_SCOFR_MEAN",
        "DAYS_LAST_PHONE_CHANGE",
        "APPROVED_DAYS_DECISION_MIN",
        "DAYS_ID_PUBLISH",
        "REG_CITY_NOT_WORK_CITY",
        "REFUSED_HOUR_APPR_PROCESS_START_MIN",
        "CODE_GENDER",
        "BURO_STATUS_C_MEAN_MEAN",
        "NAME_EDUCATION_TYPE_Higher_education",
        "PREV_NAME_CONTRACT_STATUS_Approved_MEAN",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]

    # Vérification que client_data est un dict
    if isinstance(client_data, dict):
        client_data_dict = client_data  # Si client_data est déjà un dictionnaire
    else:
        client_data_dict = client_data.squeeze().to_dict()  # Sinon, le convertir

    # Vérification des colonnes manquantes
    missing_features = set(expected_features) - set(client_data_dict.keys())
    if missing_features:
        st.error(f"Le fichier est manquant les colonnes suivantes : {', '.join(missing_features)}")
        return None, None

    # Afficher les données envoyées à l'API
    st.write("Données envoyées à l'API :", client_data_dict)  # Log des données envoyées
    
    response = requests.post(url, json=client_data_dict)
    
    # Afficher la réponse de l'API
    st.write(f"Response from API: {response.text}")  
    
    if response.status_code == 200:
        result = response.json()
        return result['probability'], result['class']
    else:
        st.error(f"Erreur lors de la prédiction : {response.text}")
        return None, None

# Interface principale du dashboard
def main():
    st.title("Dashboard Interactif de Scoring Crédit")

    try:
        pipeline = joblib.load("pipeline.joblib")
        st.success("Modèle chargé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return
    
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV contenant les données clients", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Données chargées")
        st.dataframe(data)

        client_id = st.selectbox("Sélectionnez un client", data['SK_ID_CURR'])
        client = data[data['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR', 'TARGET'])

        st.write("### Informations descriptives du client sélectionné")
        st.write(client)

        st.write("### Analyse des contributions des features")
        show_global_feature_importance(pipeline, data.drop(columns=['SK_ID_CURR', 'TARGET']))

        st.write("### Importance locale des features pour le client")
        client_data = client.values.reshape(1, -1)
        background_data = data.drop(columns=['SK_ID_CURR', 'TARGET'])

        shap_values = calculate_shap_values(pipeline.named_steps['classifier'], client_data, background_data)
        show_local_feature_importance(shap_values, client_data)

        global_importances = pipeline.named_steps['classifier'].coef_[0]
        feature_names = data.drop(columns=['SK_ID_CURR', 'TARGET']).columns
        compare_local_global_importance(shap_values, global_importances, feature_names)

        selected_feature = st.selectbox("Choisissez une feature pour voir sa distribution", feature_names)
        show_feature_distribution(data, selected_feature, client[selected_feature].values[0])

        feature1 = st.selectbox("Sélectionnez la première feature", feature_names)
        feature2 = st.selectbox("Sélectionnez la deuxième feature", feature_names)
        show_bivariate_analysis(data, feature1, feature2, client)

        probability, label = get_credit_score(client.to_dict())
        if probability is not None:
            st.write(f"Nouveau score de crédit : {label} ({probability:.2f})")
            show_gauge(probability)
    else:
        st.write("Veuillez télécharger un fichier CSV.")

if __name__ == "__main__":
    main()
