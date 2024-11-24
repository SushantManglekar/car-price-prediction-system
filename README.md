# Car-Price-Prediction-App

![Flask](https://img.shields.io/badge/Framework-Flask-blue)
![Docker](https://img.shields.io/badge/Containerization-Docker-blue)
![AWS ECS](https://img.shields.io/badge/Deployment-AWS--ECS-orange)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub--Actions-green)
![MLFlow](https://img.shields.io/badge/Experiment%20Tracking-MLFlow-blue)

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
10. [Future Work](#future-work)
11. [Acknowledgements](#acknowledgements)

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
   - Deploys the image to ECS upon passing all tests.

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/car-price-prediction-app.git
   cd car-price-prediction-app
