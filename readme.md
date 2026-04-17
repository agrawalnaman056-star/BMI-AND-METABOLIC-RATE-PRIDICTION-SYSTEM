# 🧬 Nexus AI: BMI & Metabolic Rate Prediction Engine

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![Machine Learning](https://img.shields.io/badge/Scikit--Learn-Random_Forest-orange.svg)
![UI](https://img.shields.io/badge/UI-Glassmorphism-purple.svg)

## 📌 Project Overview
Nexus AI is an enterprise-grade Health-Tech SaaS platform that uses Machine Learning to predict metabolic health risks based on biological data. It goes beyond simple classification by mathematically calculating the Basal Metabolic Rate (BMR) and automatically generating customized, AI-driven nutrition and fitness protocols. 

This project was developed for the Healthcare Data Science Initiative.

## ✨ Key Features
* **Advanced ML Classification:** Utilizes a Random Forest Classifier to accurately categorize patients into 6 metabolic health indices.
* **Deterministic BMR Calculation:** Integrates the clinical Mifflin-St Jeor equation for precise energy expenditure metrics.
* **Generative AI Action Plan:** Dynamically calculates TDEE (Total Daily Energy Expenditure) to prescribe exact macro-nutrient splits (Protein/Carbs/Fats) and targeted exercise minutes.
* **Premium User Interface:** A fully responsive, dark-mode dashboard featuring glassmorphism, neon accents, and interactive 3D hover effects.
* **Secure Authentication:** Built-in user registration and session management powered by SQLite and Passlib encryption.
* **WhatsApp Integration:** One-click automated health report delivery directly to the patient's phone via the Twilio API.

## 🛠️ Technology Stack
* **Backend:** Python, FastAPI, SQLAlchemy
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Frontend:** HTML5, CSS3, Vanilla JavaScript
* **Database:** SQLite
* **External APIs:** Twilio (WhatsApp Messaging)
* **Task Scheduling:** APScheduler

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/nexus-ai-health.git](https://github.com/yourusername/nexus-ai-health.git)
cd nexus-ai-health
