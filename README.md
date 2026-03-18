main.py
le fichier main.py est paramétré pour créer un nouveau modele sur les données (df_concat.csv= df_old.csv + df_new.csv)
l'entrainement se fait avec la fonction "train_model_avec_earlystopping" (issue de models.py) avec du early-stopping
Le "early stopping" a arrété l'entrainement au 96 eme epocs
Cela créée un nouveau modele "model_2026_newmodel.pkl", qui grace a l'early stopping est meilleur que s'il avait tourné plus longtemps
les affichages obtenus via MLFLOW sont dispos dans le fichier M1B1_results_mlflow_bruno;biscay.pdf

Code commenté

LANCEMENT

dans une fenetre lancer UI MLFLOW = uvx mlflow server --host 127.0.0.1 --port 5000

dans une autre fenetre on lance main.py
