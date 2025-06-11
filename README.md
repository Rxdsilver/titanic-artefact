# 🛳️ Titanic - Machine Learning Pipeline

Un projet Python pour analyser les données du Titanic avec une chaîne complète : chargement des données, nettoyage, préparation et entraînement d’un modèle de machine learning. Inclut aussi un système de sauvegarde/chargement de modèles.

## 📦 Installation

Assurez-vous d’avoir Python 3.10 ou supérieur :

```bash
pip install titanic
```

### Depuis le dépôt GitHub :

```bash
git clone https://github.com/cmercier2/titanic.git
cd titanic
pip install -e .
```

## 🧠 Fonctionnalités

### - 📂 Chargement des données : load_data

### - 🧼 Nettoyage et encodage : clean_data

### - ✂️ Séparation des données : prepare_data

### - 🧪 Entraînement du modèle : train_model

### - 💾 Sauvegarde/Chargement du modèle : save_model / load_model

## 📁 Structure du projet

```bash
titanic/
├── data.py          # Chargement et nettoyage des données
├── train.py         # Entraînement du modèle
├── registry.py      # Sauvegarde/Chargement du modèle
tests/               # Tests unitaires
pyproject.toml       # Configuration du projet
```

## 🧪 Tests

```bash
pytest --cov=titanic
```

### 👤 Auteurs

- Hippolyte Bernard @Rxdsilver
- Clément Mercier @cmercier2