# Medical Recommendation System - Inference Scripts Guide

Complete guide for using all inference scripts in this project.

---

## 📁 Project Structure

```
project/
│
├── models/
│   └── svc.pkl                    # Trained model
│
├── datasets/
│   ├── description.csv            # Disease descriptions
│   ├── precautions_df.csv         # Precautions
│   ├── medications.csv            # Medications
│   ├── diets.csv                  # Diet recommendations
│   └── workout_df.csv             # Workout suggestions
│
├── inference.py                   # Interactive CLI inference
├── batch_inference.py             # Batch processing
├── api_inference.py               # REST API
└── main.py                        # Flask web application
```

---

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install numpy pandas scikit-learn flask pickle-mixin
```

---

## 1️⃣ Interactive CLI Inference (`inference.py`)

### Description
Command-line tool for single patient diagnosis with full recommendations.

### Usage

#### Interactive Mode (Recommended)
```bash
python inference.py
```

Then enter symptoms when prompted:
```
Enter symptoms: itching, skin_rash, fever, headache
```

#### Command-Line Mode
```bash
python inference.py itching,skin_rash,fever
```

### Example Output
```
======================================================================
MEDICAL RECOMMENDATION SYSTEM - INFERENCE RESULTS
======================================================================

📋 Symptoms Analyzed: 3
   itching, skin_rash, nodal_skin_eruptions

🔍 PREDICTED DISEASE: Fungal infection

📖 Description:
   Fungal infection is a common skin condition caused by fungi.

🛡️  Precautions:
   1. bath twice
   2. use detol or neem in bathing water
   3. keep infected area dry
   4. use clean cloths

💊 Recommended Medications:
   • Antifungal Cream
   • Fluconazole
   • Terbinafine

🥗 Dietary Recommendations:
   • Antifungal Diet
   • Probiotics
   • Garlic

💪 Lifestyle & Workout Suggestions:
   • Avoid sugary foods
   • Stay hydrated
   • Include yogurt in diet

======================================================================
⚕️  MEDICAL DISCLAIMER
======================================================================
This is an AI-based prediction system for educational purposes only.
It should NOT replace professional medical advice, diagnosis, or treatment.
Always consult with a qualified healthcare provider for medical concerns.
======================================================================
```

### Features
✅ Input validation with error messages  
✅ Invalid symptom detection  
✅ Complete recommendations  
✅ Logging to `inference.log`  
✅ Medical disclaimer  

---

## 2️⃣ Batch Inference (`batch_inference.py`)

### Description
Process multiple patients from a CSV file.

### Input CSV Format

Create a file `patients.csv`:
```csv
patient_id,symptoms
1,"itching,skin_rash,nodal_skin_eruptions"
2,"cough,high_fever,headache,fatigue"
3,"chest_pain,fast_heart_rate,breathlessness"
4,"yellowish_skin,dark_urine,nausea"
```

### Usage

```bash
# Basic usage
python batch_inference.py patients.csv

# Specify output file
python batch_inference.py patients.csv -o results.csv

# Use custom model
python batch_inference.py patients.csv -m custom_model.pkl
```

### Output CSV

Generated file `patients_results.csv`:
```csv
patient_id,symptoms,predicted_disease,prediction_date
1,"itching,skin_rash,nodal_skin_eruptions",Fungal infection,2025-10-26 14:30:00
2,"cough,high_fever,headache,fatigue",Common Cold,2025-10-26 14:30:01
3,"chest_pain,fast_heart_rate,breathlessness",Heart attack,2025-10-26 14:30:02
4,"yellowish_skin,dark_urine,nausea",Jaundice,2025-10-26 14:30:03
```

### Summary Output
```
======================================================================
BATCH INFERENCE SUMMARY
======================================================================
Total records processed: 100
Results saved to: patients_results.csv

Disease distribution:
Common Cold              25
Fungal infection         18
Allergy                  12
Migraine                 10
...
======================================================================
```

---

## 3️⃣ REST API (`api_inference.py`)

### Description
Production-ready REST API for integrating with other applications.

### Start the Server

```bash
python api_inference.py
```

Server will start at: `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "datasets_loaded": true
}
```

#### 2. List All Symptoms
```bash
curl http://localhost:5000/api/symptoms
```

Response:
```json
{
  "total_symptoms": 132,
  "symptoms": [
    "abdominal_pain",
    "abnormal_menstruation",
    "acidity",
    ...
  ]
}
```

#### 3. List All Diseases
```bash
curl http://localhost:5000/api/diseases
```

Response:
```json
{
  "total_diseases": 41,
  "diseases": [
    "AIDS",
    "Acne",
    "Alcoholic hepatitis",
    ...
  ]
}
```

#### 4. Predict Disease (Main Endpoint)

**Without Recommendations:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["itching", "skin_rash", "fever"]
  }'
```

Response:
```json
{
  "success": true,
  "predicted_disease": "Fungal infection",
  "valid_symptoms": ["itching", "skin_rash"],
  "invalid_symptoms": ["fever"],
  "symptom_count": 2,
  "processing_time_ms": 15.42,
  "disclaimer": "This is for educational purposes only. Consult a healthcare professional."
}
```

**With Recommendations:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
    "include_recommendations": true
  }'
```

Response:
```json
{
  "success": true,
  "predicted_disease": "Fungal infection",
  "valid_symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
  "invalid_symptoms": [],
  "symptom_count": 3,
  "processing_time_ms": 18.76,
  "disclaimer": "This is for educational purposes only...",
  "recommendations": {
    "description": "Fungal infection is a common skin condition...",
    "precautions": [
      "bath twice",
      "use detol or neem in bathing water",
      "keep infected area dry",
      "use clean cloths"
    ],
    "medications": [
      "Antifungal Cream",
      "Fluconazole",
      "Terbinafine"
    ],
    "diets": [
      "Antifungal Diet",
      "Probiotics",
      "Garlic"
    ],
    "workouts": [
      "Avoid sugary foods",
      "Stay hydrated",
      "Include yogurt in diet"
    ]
  }
}
```

### Error Responses

**Invalid Symptoms:**
```json
{
  "success": false,
  "error": "No valid symptoms provided",
  "invalid_symptoms": ["xyz", "abc"]
}
```

**Missing Data:**
```json
{
  "success": false,
  "error": "Missing required field: symptoms"
}
```

### Python Client Example

```python
import requests
import json

url = "http://localhost:5000/api/predict"
payload = {
    "symptoms": ["itching", "skin_rash", "fever"],
    "include_recommendations": True
}

response = requests.post(url, json=payload)
result = response.json()

if result['success']:
    print(f"Disease: {result['predicted_disease']}")
    print(f"Description: {result['recommendations']['description']}")
else:
    print(f"Error: {result['error']}")
```

### JavaScript/Fetch Example

```javascript
fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    symptoms: ['itching', 'skin_rash', 'fever'],
    include_recommendations: true
  })
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    console.log('Disease:', data.predicted_disease);
    console.log('Recommendations:', data.recommendations);
  } else {
    console.error('Error:', data.error);
  }
});
```

---

## 4️⃣ Web Application (`main.py`)

### Description
Full-featured web interface with HTML forms.

### Start the Server

```bash
python main.py
```

Visit: `http://localhost:5000`

### Features
- 🏠 Home page with symptom input
- 📊 Results page with predictions
- 📝 About, Contact, Developer pages
- 📰 Blog section

---

## 🔍 Valid Symptoms List

<details>
<summary>Click to expand (132 symptoms)</summary>

```
itching, skin_rash, nodal_skin_eruptions, continuous_sneezing,
shivering, chills, joint_pain, stomach_pain, acidity,
ulcers_on_tongue, muscle_wasting, vomiting, burning_micturition,
spotting_ urination, fatigue, weight_gain, anxiety,
cold_hands_and_feets, mood_swings, weight_loss, restlessness,
lethargy, patches_in_throat, irregular_sugar_level, cough,
high_fever, sunken_eyes, breathlessness, sweating, dehydration,
indigestion, headache, yellowish_skin, dark_urine, nausea,
loss_of_appetite, pain_behind_the_eyes, back_pain, constipation,
abdominal_pain, diarrhoea, mild_fever, yellow_urine,
yellowing_of_eyes, acute_liver_failure, fluid_overload,
swelling_of_stomach, swelled_lymph_nodes, malaise,
blurred_and_distorted_vision, phlegm, throat_irritation,
redness_of_eyes,