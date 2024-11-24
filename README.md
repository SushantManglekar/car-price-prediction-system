# Car-Price-Prediction-App

![Flask](https://img.shields.io/badge/Framework-Flask-blue)
![Docker](https://img.shields.io/badge/Containerization-Docker-skyblue)
![AWS ECS](https://img.shields.io/badge/Deployment-AWS--ECS-orange)
![AWS CloudWatch](https://img.shields.io/badge/Monitoring-AWS--CloudWatch-lime)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub--Actions-green)
![MLFlow](https://img.shields.io/badge/Experiment%20Tracking-MLFlow-blue)
![DVC](https://img.shields.io/badge/Dataset%20Versioning-DVC-black)

## Table of Contents
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Project Workflow](#project-workflow)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Feature Engineering](#feature-engineering)
6. [Machine Learning Model](#machine-learning-model)
7. [Dockerization](#dockerization)
8. [Deployment](#deployment)
   - [AWS ECS Deployment](#aws-ecs-deployment)
   - [CI/CD Pipeline](#cicd-pipeline)
9. [How to Run Locally](#how-to-run-locally)
10. [API Endpoints](#api-endpoints)
11. [Future Work](#future-work)
12. [Acknowledgements](#acknowledgements)

---

## Introduction

The **Car-Price-Prediction-App** is a machine learning-based web application that predicts car prices based on user input. This project focuses on:
- Understanding the dataset through **Exploratory Data Analysis (EDA)**.
- Enhancing performance with **Feature Engineering**.
- Employing modern DevOps practices for deployment using **Docker** and **AWS ECS**.
- Streamlining development and deployment using **CI/CD pipelines** via **GitHub Actions**.

---

## Technologies Used

- **Framework:** Flask
- **Version Control:** Git
- **Data Tracking:** DVC
- **Experiment Tracking:** MLFlow
- **Containerization:** Docker
- **Cloud Deployment:** AWS ECS with Fargate
- **CI/CD:** GitHub Actions

---

## Project Workflow

1. **EDA:** Analyze and visualize the dataset to uncover trends and insights.
2. **Feature Engineering:** Transform raw data into meaningful features.
3. **Model Development:** Train and log models using MLFlow.
4. **Dockerization:** Containerize the application.
5. **Deployment:** Deploy the Dockerized app to AWS ECS and configure CI/CD.

---

## Exploratory Data Analysis (EDA)

- **Objective:** Understand the data distribution and identify trends affecting car prices.
- **Techniques Used:**
  - Correlation analysis.
  - Visualizations: scatter plots, histograms, heatmaps.
  - Outlier detection and treatment.

**Key Findings:**
- Engine size and car brand significantly influence car prices.
- Certain features required transformations for better model accuracy.

In this project, a significant amount of time was spent on Exploratory Data Analysis (EDA) to understand the dataset before proceeding to model training. Below are some key visualizations from the EDA process:

### 1. Price Distribution by Category
![Price Distribution by Category](https://github.com/SushantManglekar/car-price-prediction-system/blob/master/public/price_dist-category.png)

### 2. Price Distribution by Model
![Price Distribution by Model](https://github.com/SushantManglekar/car-price-prediction-system/blob/master/public/price_dist_model.png)

### 3. Feature Correlation
![Feature Correlation](https://github.com/SushantManglekar/car-price-prediction-system/blob/master/public/correlation.png)

### 4. Outlier Analysis
![Outlier Analysis](https://github.com/SushantManglekar/car-price-prediction-system/blob/master/public/Outlier_analysis.png)

---

## Feature Engineering

- **Transformations:** Applied log transformations for skewed features.
- **Encoding:** Used one-hot encoding for categorical variables.
- **Scaling:** Standardized numerical features.
- **Feature Selection:** Retained only the most impactful features for prediction.

---

## Machine Learning Model

- **Algorithm:** Random Forest Regressor (or specify your model).
- **Tools:**
  - **DVC:** To track and version raw and processed datasets.
  - **MLFlow:** For tracking model metrics, hyperparameters, and outputs.

---

## Dockerization

The application is containerized using Docker for consistent deployment across environments:
1. Built a Docker image using the `Dockerfile`.
2. Tagged and pushed the image to **Amazon ECR**.
3. Configured the container to serve predictions via Flask.

---

## Deployment

### AWS ECS Deployment

1. **Docker Image:** Hosted on **Amazon ECR**.
2. **Orchestration:** Managed with **AWS ECS Fargate**.
3. **Networking:** Configured security groups and load balancer for external access.
4. **Monitoring:** Logs and metrics tracked via **AWS CloudWatch**.

### CI/CD Pipeline

1. **GitHub Actions Workflow:**
   - Builds the Docker image.
   - Runs health check tests.
   - Deploy the image to ECS upon passing all tests.

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/car-price-prediction-app.git
   cd car-price-prediction-app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run Flask app:
   ```bash
   python app.py
   
## API Endpoints

The following screenshots show the API call examples for predicting car prices and testing the app.

### 1. API Call Screenshot for Price Prediction
![API Call for Price Prediction](https://github.com/SushantManglekar/car-price-prediction-system/blob/master/public/Screenshot%202024-11-24%20093445.png)

### 2. API Call Screenshot for health check Endpoint
![API Call for health check Endpoint](https://github.com/SushantManglekar/car-price-prediction-system/blob/master/public/Screenshot%202024-11-24%20093729.png)

---
## Future Work

- **Model Enhancements:** Experiment with advanced algorithms like Gradient Boosting (XGBoost, LightGBM) or Neural Networks to improve prediction accuracy.
- **Scalability:** 
  - Implement **auto-scaling** in AWS ECS to handle varying traffic loads dynamically.
  - Explore **serverless options** like AWS Lambda for specific components to optimize costs.
- **User Interface Improvements:** 
  - Create an intuitive dashboard for predictions and EDA visualizations using tools like Dash or Streamlit.
  - Add interactive elements for custom data input and insights.
- **Data Pipeline Automation:** Automate data ingestion, preprocessing, and model retraining using AWS Step Functions or Apache Airflow.
- **Monitoring and Alerts:** Integrate a robust monitoring system with tools like Prometheus and Grafana to monitor app performance and receive alerts for failures or anomalies.
- **MLOps Integration:**
  - Implement continuous training pipelines to keep the model updated with new data.
  - Explore feature stores for better feature management and sharing.

---

## Acknowledgements

- **Dataset Source:**  
  The dataset used for this project is publicly available at [https://www.kaggle.com/datasets/mohidabdulrehman/ultimate-car-price-prediction-dataset](#).  

- **Tools and Platforms:**  
  - [Flask](https://flask.palletsprojects.com/) for building the web application.
  - [DVC](https://dvc.org/) for data and model versioning.
  - [MLFlow](https://mlflow.org/) for experiment tracking and model management.
  - [Docker](https://www.docker.com/) for containerizing the application.
  - [AWS ECS](https://aws.amazon.com/ecs/) and [AWS Fargate](https://aws.amazon.com/fargate/) for deployment and orchestration.
  - [GitHub Actions](https://github.com/features/actions) for CI/CD pipeline integration.
---

