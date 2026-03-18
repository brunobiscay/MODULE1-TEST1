from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def create_nn_model(input_dim):
    """
    Fonction pour créer et compiler un modèle de réseau de neurones simple.
    """
    model = Sequential()
    # Première couche dense (fully connected)
    # - 64 neurones
    # - fonction d'activation ReLU
    # - input_dim : taille du vecteur d’entrée (nombre de features)
    model.add(Dense(64, activation="relu", input_dim=input_dim))

    # Deuxième couche dense
    # - 32 neurones
    # - ReLU comme activation (fonction la plus utilisée pour les couches cachées)
    model.add(Dense(32, activation="relu"))

    # Couche de sortie
    # - 1 neurone (cas typique pour une régression)
    # - Pas d’activation => sortie linéaire (par défaut)
    # Dense = couche entièrement connectée (chaque neurone reçoit l'ensemble des entrées).
    # Couche de sortie à 1 neurone :
    # Regression → OK (prédire un nombre).
    model.add(Dense(1))

    # On choisit l'optimiseur ADaptativeMomentEstimation
    # on utilise la fonction de perte Mean Squared Error
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(
    model, X, y, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0
):
    hist = model.fit(
        X,
        y,
        validation_data=(X_val, y_val)
        if X_val is not None and y_val is not None
        else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    return model, hist


# concepts
# Batch size refers to the number of training samples processed before the model's internal parameters are updated (default 32 )


def train_model_avec_earlystopping(
    model, X, y, X_val=None, y_val=None, epochs=150, batch_size=32, verbose=0
):

    # --- Early Stopping ---
    early_stop = EarlyStopping(
        monitor="val_loss",  # surveille la loss de validation
        patience=10,  # arrête après 10 epochs sans amélioration
        restore_best_weights=True,  # récupère les meilleurs poids, avant la derive
        min_delta=1e-5,  # amélioration minimale
        verbose=1,
    )

    # --- Entraînement du modèle ---
    hist = model.fit(
        X,
        y,
        validation_data=(X_val, y_val)
        if X_val is not None and y_val is not None
        else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[early_stop],
    )

    return model, hist


def model_predict(model, X):
    y_pred = model.predict(X).flatten()
    return y_pred
