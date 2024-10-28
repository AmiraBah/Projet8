# Importations nécessaires
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import logging
import numpy as np

# Configurer le logging
logging.basicConfig(level=logging.INFO)

# Charger le pipeline 
try:
    pipeline = joblib.load("pipeline.joblib")
    logging.info("Modèle chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")
    raise

# Initialiser l'API
app = FastAPI()

# Définir les noms de caractéristiques attendus
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

# Route pour la racine 
@app.get("/")
def read_root():
    logging.info("Received GET request at /")
    return {"message": "Bienvenue à l'API de scoring"}

# Fonction pour calculer la classe en fonction du seuil
def get_prediction_label(probability, threshold=0.53):
    return "Accepted" if probability < threshold else "Rejected"

# Route API pour la prédiction
@app.post("/predict")
async def predict(features: dict):
    logging.info("Received POST request at /predict")
    logging.info(f"Features received: {features}")
    
    # Vérifiez que toutes les caractéristiques attendues sont présentes
    missing_features = set(expected_features) - set(features.keys())
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
        raise HTTPException(
            status_code=400, 
            detail=f"Missing features: {', '.join(missing_features)}"
        )
    
    # Si toutes les caractéristiques sont présentes, procéder à la prédiction
    X = pd.DataFrame([features], columns=expected_features)
    
    # Imputer les NaN par 0
    X.fillna(0, inplace=True)  # Impute les NaN avec 0

    X_np = X.values
    
    try:
        prob = pipeline.predict_proba(X_np)[:, 1]  # Récupération de la probabilité de la classe 1
        label = get_prediction_label(prob[0], threshold=0.53)  # On passe prob[0] pour le label
        logging.info(f"Prediction made: probability={prob[0]}, class={label}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    return {"probability": prob[0], "class": label}


# Route API pour obtenir les importances des caractéristiques
@app.get("/feature_importances")
async def get_feature_importances():
    try:
        # Extraire le modèle de régression logistique du pipeline
        logistic_model = pipeline.named_steps['classifier']
        feature_importances = logistic_model.coef_[0]  # Récupérer les coefficients du modèle

        # Associer les coefficients aux noms de caractéristiques
        features_importances_dict = dict(zip(expected_features, feature_importances))
        
        return features_importances_dict
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des importances des caractéristiques : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des importances : {str(e)}")
