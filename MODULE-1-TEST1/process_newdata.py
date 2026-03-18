from os.path import join as join

import joblib
import pandas as pd
from models.models import model_predict
from modules.preprocess import preprocessing

# Chargement des datasets
# df_old = pd.read_csv(join("data", "df_old.csv"))

# Charger le préprocesseur
preprocessor_loaded = joblib.load(join("models", "preprocessor.pkl"))

# preprocesser les data
# X, y, _ = preprocessing(df_old)

# split dédié "trainig/validation"
# X_train, X_test, y_train, y_test = split(X, y)


# charger le modèle entrainé précédemment
model_2024_08 = joblib.load(join("models", "model_2026_newmodel.pkl"))

# y_pred = model_predict(model_2024_08, X_train)


########### NOUVELLES DONNES, utilisation en production aka "inference"
"""Si modèle deja entrainé et validé:
- par defaut  pas besoin de splitter les données
 -il est recommandé de garder un petit échantillon de nouvelles données réelles pour un monitoring
  -  et un suivi des performance (librairie MLFLOW)
"""
df_prod = pd.read_csv(join("data", "df_new.csv"))

# preprocesser les data
X_prod, y_prod, _ = preprocessing(df_prod)

# split pour monitoriung : reference vs batch en cours
# X_prod, X_testprod, y_prod, y_testprod = split(Xb, yb)
# reference_data = X_train.sample(frac=0.25, random_state=42)

# Prédiction sur les nouvelles données
y_prod_pred = model_predict(model_2024_08, X_prod)


# mesurer les performances MSE, MAE et R²
# perf = evaluate_performance(y_prod, y_prod_pred)

# print_data(perf)
