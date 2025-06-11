# Makefile pour le projet Titanic

# Variables
PYTHON := python
PIP := pip
ENV ?= titanic-env
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data
MODELS_DIR := models

.PHONY: help install lint test run clean format

## Affiche la liste des commandes
help:
	@echo "Usage :"
	@echo "  make install     - Installe les dépendances"
	@echo "  make lint        - Lint le code avec flake8"
	@echo "  make format      - Formate le code avec black"
	@echo "  make test        - Lance les tests avec pytest"
	@echo "  make run         - Lance le main.py"
	@echo "  make clean       - Supprime les fichiers temporaires"

## Installe les dépendances
install:
	$(PIP) install -r requirements.txt

## Lint du code
lint:
	flake8 $(SRC_DIR) --max-line-length=100

## Formate le code avec black
format:
	black $(SRC_DIR) $(TEST_DIR)

## Lance les tests avec pytest
test:
	pytest --disable-warnings

## Exécute le script principal
run:
	$(PYTHON) $(SRC_DIR)/main.py

## Nettoie les fichiers temporaires
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf .pytest_cache .mypy_cache .coverage* coverage.xml htmlcov dist build *.egg-info

## Génère un rapport de couverture
test-cov:
	pytest --cov=src/titanic --cov-report=term --cov-report=html tests/
	@echo "🧪 Rapport HTML généré dans htmlcov/index.html"