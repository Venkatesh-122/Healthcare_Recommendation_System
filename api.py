"""
Project: Healthcare Recommendation System API
Author: Supriya Mandal, Madana Venkatesh & Biki Haldar
GitHub: [MSupriya4223](https://github.com/MSupriya4223)
Year: 2025
"""
"""
🏥 Healthcare Recommendation System
==================================

Welcome to the backend engine of the Healthcare Recommendation System!

This API:
• Validates user-provided symptoms
• Predicts the most likely disease using a trained ML model
• Provides useful recommendations (precautions, medications, diets, workouts)
• Powers the frontend interface (`index.html`) through a clean JSON API

Built with Flask + NumPy + Pandas, this backend is optimized for fast, reliable 
health inference.

⚠ Disclaimer:
This tool is NOT a replacement for real medical diagnosis. Always consult a doctor.
"""


from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) 

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Global variables
model = None
datasets = {}

# Symptom and disease mappings (132 symptoms, 41 diseases)
SYMPTOMS_DICT = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2,
    'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
    'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
    'burning_micturition': 12, 'spotting_ urination': 13,
    'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19,
    'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
    'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25,
    'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31,
    'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34,
    'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39,
    'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47, 'malaise': 48,
    'blurred_and_distorted_vision': 49, 'phlegm': 50,
    'throat_irritation': 51, 'redness_of_eyes': 52,
    'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57,
    'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64,
    'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68,
    'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
    'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74,
    'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76,
    'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79,
    'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84,
    'loss_of_balance': 85, 'unsteadiness': 86,
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
    'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97,
    'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101,
    'dischromic _patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106,
    'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113,
    'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117,
    'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
    'palpitations': 120, 'painful_walking': 121,
    'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
    'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128,
    'blister': 129, 'red_sore_around_nose': 130,
    'yellow_crust_ooze': 131
}

DISEASES_LIST = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD',
    9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis',
    32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice',
    29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid',
    40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C',
    21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
    13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack',
    39: 'Varicose veins', 26: 'Hypothyroidism',
    24: 'Hyperthyroidism', 25: 'Hypoglycemia',
    31: 'Osteoarthristis', 5: 'Arthritis',
    0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}


def load_model():
    """Load the trained ML model"""
    global model
    try:
        model_path = 'models/svc.pkl'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info("✅ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False


def load_datasets():
    """Load all recommendation datasets"""
    global datasets
    try:
        dataset_files = {
            'description': 'datasets/description.csv',
            'precautions': 'datasets/precautions_df.csv',
            'medications': 'datasets/medications.csv',
            'diets': 'datasets/diets.csv',
            'workout': 'datasets/workout_df.csv'
        }
        
        for key, filepath in dataset_files.items():
            if os.path.exists(filepath):
                datasets[key] = pd.read_csv(filepath)
                logger.info(f"✅ Loaded {key}: {len(datasets[key])} records")
            else:
                logger.warning(f"⚠️  Dataset not found: {filepath}")
                datasets[key] = pd.DataFrame()
        
        return True
    except Exception as e:
        logger.error(f"❌ Error loading datasets: {e}")
        return False


def predict_disease(symptoms):
    """
    Predict disease from symptom list
    
    Args:
        symptoms: List of symptom strings
        
    Returns:
        Predicted disease name
    """
    try:
        # Create input vector
        input_vector = np.zeros(len(SYMPTOMS_DICT))
        
        for symptom in symptoms:
            if symptom in SYMPTOMS_DICT:
                input_vector[SYMPTOMS_DICT[symptom]] = 1
        
        # Predict using model
        prediction = model.predict([input_vector])[0]
        disease = DISEASES_LIST.get(prediction, "Unknown Disease")
        
        logger.info(f"🔍 Predicted: {disease} from {len(symptoms)} symptoms")
        return disease
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return None


def get_recommendations(disease):
    """
    Get all recommendations for a disease
    
    Args:
        disease: Disease name
        
    Returns:
        Dictionary with description, precautions, medications, diets, workouts
    """
    recommendations = {
        'description': '',
        'precautions': [],
        'medications': [],
        'diets': [],
        'workouts': []
    }
    
    try:
        # Get description
        if not datasets['description'].empty:
            desc_df = datasets['description'][
                datasets['description']['Disease'] == disease
            ]
            if not desc_df.empty:
                recommendations['description'] = str(desc_df['Description'].values[0])
        
        # Get precautions
        if not datasets['precautions'].empty:
            prec_df = datasets['precautions'][
                datasets['precautions']['Disease'] == disease
            ]
            if not prec_df.empty:
                prec_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
                recommendations['precautions'] = [
                    str(val) for val in prec_df[prec_cols].values[0]
                    if pd.notna(val) and str(val) != 'nan'
                ]
        
        # Get medications
        if not datasets['medications'].empty:
            med_df = datasets['medications'][
                datasets['medications']['Disease'] == disease
            ]
            if not med_df.empty:
                recommendations['medications'] = [
                    str(m) for m in med_df['Medication'].tolist()
                ]
        
        # Get diets
        if not datasets['diets'].empty:
            diet_df = datasets['diets'][
                datasets['diets']['Disease'] == disease
            ]
            if not diet_df.empty:
                recommendations['diets'] = [
                    str(d) for d in diet_df['Diet'].tolist()
                ]
        
        # Get workouts
        if not datasets['workout'].empty:
            workout_df = datasets['workout'][
                datasets['workout']['disease'] == disease
            ]
            if not workout_df.empty:
                recommendations['workouts'] = [
                    str(w) for w in workout_df['workout'].tolist()
                ]
        
    except Exception as e:
        logger.error(f"❌ Error getting recommendations: {e}")
    
    return recommendations


# ==================== API ENDPOINTS ====================

@app.route('/')
def index():
    """Serve the HTML interface"""
    try:
        return send_file('inference_demo.html')
    except:
        return jsonify({
            'message': 'Healthcare Recommendation API',
            'version': '1.0',
            'status': 'running',
            'endpoints': {
                '/api/predict': 'POST - Main prediction endpoint',
                '/api/health': 'GET - Health check',
                '/api/symptoms': 'GET - List all valid symptoms'
            }
        })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'datasets_loaded': len(datasets) > 0
    })


@app.route('/api/symptoms', methods=['GET'])
def list_symptoms():
    """Return list of all valid symptoms"""
    return jsonify({
        'total': len(SYMPTOMS_DICT),
        'symptoms': sorted(list(SYMPTOMS_DICT.keys()))
    })


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Main prediction endpoint - designed for HTML interface
    
    Expected JSON:
    {
        "symptoms": ["itching", "skin_rash", "fever"]
    }
    
    Returns JSON matching HTML interface expectations:
    {
        "disease": "Fungal infection",
        "description": "...",
        "precautions": [...],
        "medications": [...],
        "diets": [...],
        "workouts": [...]
    }
    """
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'disease': 'Error'
            }), 400
        
        # Validate symptoms field
        if 'symptoms' not in data:
            return jsonify({
                'error': 'Missing required field: symptoms',
                'disease': 'Error'
            }), 400
        
        symptoms = data['symptoms']
        
        if not isinstance(symptoms, list):
            return jsonify({
                'error': 'symptoms must be an array',
                'disease': 'Error'
            }), 400
        
        if len(symptoms) == 0:
            return jsonify({
                'error': 'symptoms array is empty',
                'disease': 'Error'
            }), 400
        
        # Validate each symptom
        valid_symptoms = []
        invalid_symptoms = []
        
        for symptom in symptoms:
            symptom_clean = str(symptom).strip().lower().replace(' ', '_')
            if symptom_clean in SYMPTOMS_DICT:
                valid_symptoms.append(symptom_clean)
            else:
                invalid_symptoms.append(symptom)
        
        if len(valid_symptoms) == 0:
            return jsonify({
                'error': 'No valid symptoms provided',
                'disease': 'Unknown',
                'description': 'Please enter valid symptoms from the symptom list.',
                'precautions': [],
                'medications': [],
                'diets': [],
                'workouts': [],
                'invalid_symptoms': invalid_symptoms
            }), 400
        
        # Predict disease
        disease = predict_disease(valid_symptoms)
        
        if not disease:
            return jsonify({
                'error': 'Prediction failed',
                'disease': 'Error'
            }), 500
        
        # Get recommendations
        recommendations = get_recommendations(disease)
        
        # Build response in format expected by HTML interface
        response_data = {
            'disease': disease,
            'description': recommendations['description'],
            'precautions': recommendations['precautions'],
            'medications': recommendations['medications'],
            'diets': recommendations['diets'],
            'workouts': recommendations['workouts'],
            'valid_symptoms': valid_symptoms,
            'invalid_symptoms': invalid_symptoms,
            'symptom_count': len(valid_symptoms),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Prediction successful: {disease}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ API error: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'disease': 'Error'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/api/predict', '/api/symptoms', '/api/health']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error'
    }), 500


# ==================== STARTUP ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 HEALTHCARE RECOMMENDATION SYSTEM - API SERVER")
    print("="*70)
    print("\nStarting server for HTML inference interface...")
    
    # Load model and datasets
    print("\n📦 Loading resources...")
    
    model_loaded = load_model()
    datasets_loaded = load_datasets()
    
    if not model_loaded:
        print("\n❌ CRITICAL ERROR: Model could not be loaded!")
        print("   Make sure 'models/svc.pkl' exists")
        exit(1)
    
    if not datasets_loaded:
        print("\n⚠️  WARNING: Some datasets could not be loaded")
        print("   Recommendations may be incomplete")
    
    print("\n✅ All resources loaded successfully!")
    print("\n" + "="*70)
    print("📍 API Endpoints:")
    print("   - Main prediction: POST http://localhost:5000/api/predict")
    print("   - Health check:    GET  http://localhost:5000/api/health")
    print("   - List symptoms:   GET  http://localhost:5000/api/symptoms")
    print("\n🌐 HTML Interface:")
    print("   - Open: http://localhost:5000")
    print("   - Or open inference_demo.html directly in browser")
    print("   - Set API URL to: http://localhost:5000/api/predict")
    print("   - Uncheck 'Mock Mode' to use real predictions")
    print("="*70 + "\n")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to True only for development
        threaded=True
    )

