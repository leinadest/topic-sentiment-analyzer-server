# Social Media Sentiment Analysis Server

## Description

This server analyzes social media sentiment for a given text by scraping Reddit comments and running sentiment analysis using a machine learning pipeline.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview

The project uses Python's [FastAPI](https://fastapi.tiangolo.com/) framework to create a web application.

```
my-fastapi-ml-project/
│
├── app/
│   ├── main.py               # FastAPI app entry point
│   ├── ml_model.py           # Code for loading the ML model and running inference
│   └── scraper.py            # Code for scraping data (could be a wrapper for the API)
│
├── notebooks/
│   └── ml_workflow.ipynb     # Jupyter notebook with the ML workflow (data prep, model training, etc.)
│
├── requirements.txt          # List of Python dependencies
├── Dockerfile                # Docker setup for containerization
├── README.md                 # Project documentation
├── .gitignore                # Git ignore file
└── tests/                    # Test folder for unit tests
    ├── test_main.py          # FastAPI endpoint tests
    └── test_ml_model.py      # Tests for the ML model inference logic
```

## Installation
