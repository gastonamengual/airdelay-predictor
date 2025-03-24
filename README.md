# Flight Delay Predictor

This project is being developed as part of a job interview for an airline company. This project is an API designed to predict flight delays based on various input features. It includes two primary endpoints:

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
