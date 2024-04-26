---
author: Tim Manik
date: 2024-04-26
---

## [Internet2 CLASS Advanced Spring 2024 - AI/ML Project](https://github.com/Internet2/class-adv-spr2024-proj2)

## Overview

This repository supports the CLASS Advanced Spring 2024 - AI/ML Project.

This README.md is intended to help provide a starting point for the project. Should your cohort choose this project, you should consider this a proposal and take liberty to modify this project. Ultimately the goal of project week is to present a real-world challenge while reinforcing knowledge through a practical application of learned skills. Feel free to change the scope of this project to something that is achievable by the cohort and can show some working code or proof of concept within the time of the project week (5 days).


### Background

Machine learning has become more and more prevalent in higher education research across various disciplines such as biology, chemistry, social sciences, and beyond. As a result, research teams increasingly rely on machine learning workflows to gain insights from large and complex datasets. This means that more teams will need the necessary support to establish machine learning environments.

### Objective

The objective of this project is to develop a solution that standardizes the creation and management of these machine learning environments for research teams. Specifically, at the end of this project, a collaborative environment and a Machine Learning Operations (MLOps) pipeline will be built and templatized via Infrastructure-as-Code (IaC). This will become an a set of artifacts that institutions worldwide can use to deploy into their environments to support their researchers.


### Requirements

- An account with a Cloud Service Provider (CSP) to leverage cloud-based resources. For this project, it is preferred to use GCP.
- Access to the CSP's machine learning services for model training and deployment. Below are the minimal services that you should have access to. Additional services will be determined as the project is further scoped out.
    - GCP: Vertex AI's suite of services, Google Compute Engine, Google Cloud Storage.
    - AWS: Amazon SageMaker's suite of services, Amazon EC2, S3.
    - Azure: Azure "AI + machine learning"'s suite of services, Azure VM, Azure Blob Storage.
- Proficiency in tools such as Terraform, Python, and Jupyter Notebooks.
- Basic understanding of machine learning concepts including training, feature engineering, model deployment, inference, and model evaluation metrics like accuracy.

### Directions

Pending per finalizing of project direction and scope.


### References

#### GCP
- [GCP's AI and Machine Learning products homepage](https://cloud.google.com/ai)
- [Best practices for implementing machine learning on Google Cloud](https://cloud.google.com/architecture/ml-on-gcp-best-practices)
- [Build and deploy generative AI and machine learning models in an enterprise](https://cloud.google.com/architecture/genai-mlops-blueprint)

#### AWS
- [Amazon SageMaker homepage](https://aws.amazon.com/sagemaker/)
- [Amazon SageMaker for MLOps homepage](https://aws.amazon.com/sagemaker/mlops/)
- [SageMaker Immersion Day](https://catalog.us-east-1.prod.workshops.aws/workshops/63069e26-921c-4ce1-9cc7-dd882ff62575/en-US)

#### Azure
- [Azure Machine Learning homepage](https://azure.microsoft.com/en-us/products/machine-learning)
- [What is Azure Machine Learning?](https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning)
- [Quickstart: Get started with Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-azure-ml-in-a-day)