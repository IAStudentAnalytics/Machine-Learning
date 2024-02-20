import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Charger les données
df = pd.read_csv('data_set.csv')

# Fonction pour convertir le temps au format mm:ss en secondes
def convertir_en_secondes(temps):
    minutes, secondes = map(int, temps.split(':'))
    return minutes * 60 + secondes

# Convertir les colonnes 'time_ch_1' à 'time_ch_4' en secondes
for i in range(1, 5):
    df[f'time_ch_{i}'] = df[f'time_ch_{i}'].apply(convertir_en_secondes)

# Modifier la fonction categoriser_performance pour prendre en compte les temps en secondes
def categoriser_performance(row):
    note = float(row['note'])
    time = int(row['time'])
    if note < 2:
        return 'faible'
    elif note >= 2 and note < 4:
        return 'moyen'
    elif note == 5 or (note == 4 and time <= 5):
        return 'excellent'
    else:
        return 'précipité'

# Créez une nouvelle colonne 'performance' en appliquant la fonction aux notes de chaque chapitre
for i in range(1, 5):
    df[f'performance_ch_{i}'] = df.apply(
        lambda row: categoriser_performance({'note': row[f'note_ch_{i}'], 'time': row[f'time_ch_{i}']}), axis=1)

# Prendre toutes les notes et tous les temps en entrée
X = df[[f'note_ch_{i}' for i in range(1, 5)] + [f'time_ch_{i}' for i in range(1, 5)]]

# Encoder les performances en labels numériques
le = LabelEncoder()
# Vous pourriez vouloir prédire la performance sur un chapitre spécifique, ici chapitre 4 par exemple
y = le.fit_transform(df['performance_ch_4'])

# Séparer les données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construire le modèle pour la classification
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(le.classes_), activation='softmax')  # Nombre de classes de sortie représentant les catégories de performance
])

# Compiler le modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Puisque vos labels sont entiers
    metrics=['accuracy']
)

# Entraîner le modèle
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# Évaluation du modèle
_, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy sur le dataset de test: {accuracy}")

# Utiliser le même transformateur StandardScaler pour normaliser les données de test
X_test_normalized = scaler.transform(X_test)

# Diviser les données en sous-ensembles pour chaque chapitre
X_test_chapters = [X_test[:, i*2:(i+1)*2] for i in range(4)]
# Concaténer les données de chaque chapitre pour former une seule entrée par étudiant
X_test_concat = np.concatenate(X_test_chapters, axis=1)

# Prédiction de la performance par chapitre pour chaque étudiant
predictions_chapters = model.predict(X_test_concat)

# Combinaison des prédictions pour chaque étudiant
for i, row in enumerate(predictions_chapters):
    student_predictions = {}
    for j in range(4):
        predicted_class = le.inverse_transform([predictions_chapters[i][j].argmax()])[0]
        student_predictions[f'performance_ch_{j+1}'] = predicted_class
    df.loc[i, 'predictions'] = str(student_predictions)

# Calcul de la performance générale pour chaque étudiant
def performance_generale(row):
    performances = [row[f'performance_ch_{i}'] for i in range(1, 5)]
    if all(performances[i] == 'excellent' for i in range(4)):
        return 'excellent'
    elif any(performances[i] == 'précipité' for i in range(4)):
        return 'précipité'
    elif all(performances[i] == 'moyen' for i in range(4)):
        return 'moyen'
    else:
        return 'faible'

df['performance_generale'] = df.apply(performance_generale, axis=1)

# Données d'un étudiant dans 3 chapitres (à remplacer par les vraies données de l'étudiant)
notes_etudiant = [4, 1, 3]
temps_etudiant = [38, 234, 92]  # Temps en secondes

# Créez un DataFrame pour l'étudiant en incluant toutes les colonnes utilisées lors de l'entraînement du StandardScaler
df_etudiant = pd.DataFrame({
    'note_ch_1': [notes_etudiant[0]],
    'time_ch_1': [temps_etudiant[0]],
    'note_ch_2': [notes_etudiant[1]],
    'time_ch_2': [temps_etudiant[1]],
    'note_ch_3': [notes_etudiant[2]],
    'time_ch_3': [temps_etudiant[2]],
    'note_ch_4': [0],  # Valeur factice pour note_ch_4
    'time_ch_4': [0]   # Valeur factice pour time_ch_4
})


# Prendre les notes et les temps de l'étudiant en entrée
X_etudiant = df_etudiant[[f'note_ch_{i}' for i in range(1, 5)] + [f'time_ch_{i}' for i in range(1, 5)]]

# Normaliser les données de l'étudiant
X_etudiant_normalized = scaler.transform(X_etudiant)

# Prédiction de la performance dans le 4ème chapitre
prediction_etudiant = model.predict(X_etudiant_normalized)

# Affichage de la prédiction
predicted_class_etudiant = le.inverse_transform([prediction_etudiant.argmax()])[0]
print(f"Performance prévue dans le 4ème chapitre : {predicted_class_etudiant}")
