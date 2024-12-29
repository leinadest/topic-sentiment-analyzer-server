setup:
	poetry install -n

notebook_setup:
	wget -P data/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
	wget -P data/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
	wget -P data/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
	wget -P data/ http://nlp.uoregon.edu/download/embeddings/glove.6B.300d.txt
	poetry run spacy download en_core_web_lg

quality_checks:
	poetry run isort .
	poetry run black .
	poetry run flake8 .

test:
	poetry run pytest

dev:
	poetry run fastapi dev app/main.py
