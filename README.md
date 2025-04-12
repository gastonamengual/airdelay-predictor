# Flight Delay Predictor

This project is an API designed to predict flight delays based on various input features. It includes two primary endpoints:

## Train Endpoint

The train endpoint loads dummy data from an SQLite database using SQLAlchemy. It then utilizes Polars for efficient data handling and Ray for distributed computation to train an XGBoost model. The features of the data are: *airline* - *flight_number* - *origin* - *destination* - *departure_time* - *weather* - *congestion_level* - *day_of_week*. After the training, the model is saved to a Ray checkpoint for future predictions.

## Predict Endpoint

The predict endpoint loads the trained XGBoost model from a Ray checkpoint and predicts the likelihood of a flight delay for a given flight. The trained model is loaded from the Ray checkpoint when the predict endpoint is called. A user can send data for a specific flight, and the model will return the probability of delay based on the provided input features.

## Summary

Technologies Used
* FastAPI for building the API.
* SQLAlchemy for interacting with the SQLite database.
* Polars for fast data processing.
* Ray for distributed computation and model checkpointing.
* XGBoost for building the machine learning model.

## Next Steps

* The project is ready for deployment on VERCEL, but it cannot be deployed using the free tier (250MB) as the project size is approximately 800MB.
* Experiment tracking with MLFlow to store different model versions and allow users to select them.
* Automated CI/CD pipeline to GitHub Pages for building and deployment.
* 100% test coverage.

## Installation

1.	Clone this repository:

```bash
git clone https://github.com/gastonamengual/airdelay-predictor
cd <repository-directory>
```

2.	Create a virtual environment and install dependencies:

```bash
pipenv install --dev
pipenv shell
```

3. Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The server will be accessible at http://localhost:8080
