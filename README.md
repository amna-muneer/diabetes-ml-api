## Author
Amna Muneer  
BS Data Science  
Big Data & Machine Learning Project
# Diabetes Prediction API  
**Flask • PySpark • SQLite • Docker**

---

## 📌 Project Overview

This project is a **machine learning–based Diabetes Prediction API** developed as part of a **Big Data course project**.  
The system takes patient health parameters as input, predicts the likelihood of diabetes using a trained ML model, and stores every prediction in a SQLite database for tracking and analysis.

The project demonstrates:
- End-to-end ML pipeline
- API development
- Database logging
- Containerization using Docker

---

## 🎯 Problem Statement

Early detection of diabetes is critical for timely treatment.  
This project provides a REST API that:
- Accepts patient medical data
- Predicts diabetes risk
- Returns prediction probability
- Stores prediction history for future analysis

---

## 🧠 Solution Approach

1. User sends medical data through an API request  
2. Data is processed and passed to a trained ML model  
3. Model returns:
   - Predicted class (0 = Non-Diabetic, 1 = Diabetic)
   - Probability score  
4. Prediction details are stored in a SQLite database  
5. API sends response back to the client

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Framework:** Flask  
- **Big Data Processing:** PySpark  
- **Machine Learning:** Logistic Regression  
- **Database:** SQLite  
- **Containerization:** Docker  
- **Testing Tools:** Postman, curl  
- **Platform:** Linux (WSL – Ubuntu)

---

## 🏗️ System Architecture

Client (Postman / curl)  
⬇  
Flask REST API  
⬇  
PySpark ML Model  
⬇  
Prediction Output  
⬇  
SQLite Database (`predictions.db`)

---

## 📁 Project Structure

```text
bigdataproject/
│── app.py
│── model/
│   └── logistic_regression_model
│── predictions.db
│── requirements.txt
│── Dockerfile
│── README.md

