# Ford Car Price Prediction 🚗

## Live Demo

https://fordcarpricepred.streamlit.app/

## Overview

This project predicts Ford car prices using machine learning models trained on historical vehicle data.

The system allows users to input vehicle features and obtain real-time price predictions through an interactive web interface.

## Features

• End-to-end machine learning pipeline  
• Feature engineering with Label Encoding and One-Hot Encoding  
• Multiple regression model comparison  
• Xgboost model selection  
• Interactive Streamlit web application  
• Dataset insights dashboard  
• Model performance visualization

## Models Evaluated

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Xgboost

## Tech Stack

Python  
Pandas  
Scikit-Learn  
Streamlit

## Project Structure

ford-car-price-ml
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── data
├── models
├── reports
├── notebooks
└── assets

## Deployment

The application is deployed using Streamlit Cloud.

Run locally:

streamlit run app.py
