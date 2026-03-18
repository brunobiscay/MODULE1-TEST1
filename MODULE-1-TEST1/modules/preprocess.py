from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def preprocessing(df):
    """
    Fonction pour effectuer le prétraitement des données :
    - Imputation des valeurs manquantes.
    - Standardisation des variables numériques.
    - Encodage des variables catégorielles.
    """
    numerical_cols = ["age", "taille", "poids", "revenu_estime_mois"]
    categorical_cols = [
        "sexe",
        "sport_licence",
        "niveau_etude",
        "region",
        "smoker",
        "nationalité_francaise",
    ]

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, numerical_cols), ("cat", cat_pipeline, categorical_cols)]
    )

    # Prétraitement
    X = df.drop(columns=["nom", "prenom", "montant_pret"])
    y = df["montant_pret"]

    X_processed = preprocessor.fit_transform(X)
    """
    a)fit(X) : Calcule les vecteurs propres (eigenvectors) de la matrice de covariance de X en utilisant la décomposition en valeurs propres.
    Après avoir ajusté le PCA avec pca.fit(X), vous pouvez récupérer ces vecteurs propres via pca.components_.
    b)transform(X) : Convertit les données d’entrée depuis l’espace vectoriel initial vers l’espace vectoriel du PCA — c’est‑à‑dire l’espace défini par les vecteurs propres obtenus via l’algorithme PCA.
    Les données transformées sont généralement appelées composantes principales (PCs).
    c)fit_transform(X) : Combine les deux étapes : d’abord le calcul des vecteurs propres, puis la projection des données sur ceux‑ci.
    
    Scikit-learn's terminology: eigenvectors = components_
    """

    return X_processed, y, preprocessor
