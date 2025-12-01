# 🏥 Healthcare Recommendation System  
An end-to-end AI-based disease prediction and recommendation system using symptoms as input.  
This project includes:

✔ A machine learning inference engine  
✔ A Flask REST API  
✔ A modern, interactive HTML frontend  
✔ Recommendation datasets (description, precautions, medications, diets, workouts)

---

## 🚀 Features

### 🔍 **1. Disease Prediction**
- Input symptoms → System validates → ML model predicts one of 41 diseases  
- Uses one-hot encoding with a trained `svc.pkl` model

### 📦 **2. Recommendations Provided**
For each disease, the system returns:
- Description  
- Precautions  
- Medications  
- Diet recommendations  
- Workout suggestions  

### 🌐 **3. API Backend (Flask)**
- Endpoint: `/api/predict`  
- Cross-Origin enabled  
- Validates symptoms and responds in JSON format  
- Automatically loads datasets from `datasets/`  
- Works directly with the provided HTML interface  

### 🖥️ **4. Web Interface**
- Built using pure HTML + JavaScript  
- Real-time validation of symptoms  
- Mock Mode for testing without API  
- Clean, modern UI design  

### 🧪 **5. CLI Tool**
- Full inference pipeline  
- Predict directly via terminal  
- Saves logs to `inference.log`  
- Interactive mode supported  

---

## 📁 Project Structure

```

📦 Medical-Recommendation-System
│
├── api.py                  # Flask backend API
├── inference.py            # CLI inference tool
├── index.html              # Frontend interface
│
├── models/
│   └── svc.pkl             # Trained ML model (required)
│
├── datasets/
│   ├── description.csv
│   ├── precautions_df.csv
│   ├── medications.csv
│   ├── diets.csv
│   └── workout_df.csv
│
└── README.md

````

---

## 🛠️ Installation & Setup

### 1️⃣ **Install Dependencies**

```bash
pip install flask flask-cors numpy pandas
````

---

## 🌐 Running the Flask API

```bash
python api.py
```

Server starts at:

```
http://localhost:5000
```

API Endpoints:

| Method | Endpoint        | Description                       |
| ------ | --------------- | --------------------------------- |
| GET    | `/api/health`   | Check server health               |
| GET    | `/api/symptoms` | List all valid symptoms           |
| POST   | `/api/predict`  | Predict disease & recommendations |

---

## 🖥️ Running the Web Interface (Frontend)

You can open it directly:

```
index.html
```

### To connect backend:

1. Open the webpage
2. Turn OFF **Mock Mode**
3. Set API URL to:

```
http://localhost:5000/api/predict
```

---

## 🖥️ Running the CLI Tool

```
python inference.py
```

Example:

```
Enter symptoms: itching, skin_rash, headache
```

Outputs prediction + recommendations in the terminal.

---

## 📊 Model & Dataset Notes

* Model used: **SVM (svc.pkl)**
* Input vector length: **132 symptoms**
* Disease classes: **41 diseases**
* All supporting datasets located in `datasets/` folder

---

## 🛡️ Medical Disclaimer

This system is **for educational and research purposes only**.
It does **not** replace professional medical diagnosis or treatment.
Always consult licensed healthcare providers for medical advice.

---

## ❤️ Credits

Developed by **Supriya Mandal, Madana Venkatesh & Biki Haldar**
<br>GitHub: [MSupriya4223](https://github.com/MSupriya4223)
---
All rights reserved.
