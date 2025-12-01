"""
Project: Healthcare Recommendation System API
Author: Supriya Mandal
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


import numpy as np
import pandas as pd
import pickle
import sys
import os
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MedicalInferenceSystem:
    """
    A complete inference system for disease prediction and recommendations
    """
    
    def __init__(self, model_path: str = 'models/svc.pkl'):
        """
        Initialize the inference system
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.datasets = {}
        
        # Symptom to index mapping (132 symptoms)
        self.symptoms_dict = {
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
        
        # Disease index to name mapping (41 diseases)
        self.diseases_list = {
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
        
        logger.info("Inference system initialized")
    
    
    def load_model(self) -> bool:
        """
        Load the trained model
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    
    def load_datasets(self, datasets_dir: str = 'datasets') -> bool:
        """
        Load all recommendation datasets
        
        Args:
            datasets_dir: Directory containing CSV files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            dataset_files = {
                'description': 'description.csv',
                'precautions': 'precautions_df.csv',
                'medications': 'medications.csv',
                'diets': 'diets.csv',
                'workout': 'workout_df.csv'
            }
            
            for key, filename in dataset_files.items():
                filepath = os.path.join(datasets_dir, filename)
                
                if not os.path.exists(filepath):
                    logger.warning(f"Dataset file not found: {filepath}")
                    self.datasets[key] = pd.DataFrame()
                else:
                    self.datasets[key] = pd.read_csv(filepath)
                    logger.info(f"Loaded {key} dataset: {len(self.datasets[key])} records")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            return False
    
    
    def validate_symptoms(self, symptoms: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate user-provided symptoms
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            Tuple of (valid_symptoms, invalid_symptoms)
        """
        valid_symptoms = []
        invalid_symptoms = []
        
        for symptom in symptoms:
            # Clean the symptom string
            cleaned = symptom.strip().lower().replace(' ', '_')
            
            if cleaned in self.symptoms_dict:
                valid_symptoms.append(cleaned)
            else:
                invalid_symptoms.append(symptom)
        
        return valid_symptoms, invalid_symptoms
    
    
    def predict_disease(self, symptoms: List[str]) -> Optional[str]:
        """
        Predict disease based on symptoms
        
        Args:
            symptoms: List of valid symptom strings
            
        Returns:
            Predicted disease name or None if error
        """
        try:
            if not self.model:
                logger.error("Model not loaded. Call load_model() first.")
                return None
            
            if not symptoms:
                logger.error("No valid symptoms provided")
                return None
            
            # Create input vector
            input_vector = np.zeros(len(self.symptoms_dict))
            for symptom in symptoms:
                if symptom in self.symptoms_dict:
                    input_vector[self.symptoms_dict[symptom]] = 1
            
            # Predict
            prediction = self.model.predict([input_vector])[0]
            disease = self.diseases_list.get(prediction, "Unknown Disease")
            
            logger.info(f"Predicted disease: {disease} (code: {prediction})")
            return disease
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None
    
    
    def get_recommendations(self, disease: str) -> Dict[str, any]:
        """
        Get health recommendations for a predicted disease
        
        Args:
            disease: Disease name
            
        Returns:
            Dictionary containing all recommendations
        """
        recommendations = {
            'disease': disease,
            'description': '',
            'precautions': [],
            'medications': [],
            'diets': [],
            'workouts': []
        }
        
        try:
            # Get description
            if not self.datasets['description'].empty:
                desc_df = self.datasets['description'][
                    self.datasets['description']['Disease'] == disease
                ]
                if not desc_df.empty:
                    recommendations['description'] = desc_df['Description'].values[0]
            
            # Get precautions
            if not self.datasets['precautions'].empty:
                prec_df = self.datasets['precautions'][
                    self.datasets['precautions']['Disease'] == disease
                ]
                if not prec_df.empty:
                    prec_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
                    recommendations['precautions'] = [
                        str(val) for val in prec_df[prec_cols].values[0] 
                        if pd.notna(val) and str(val) != 'nan'
                    ]
            
            # Get medications
            if not self.datasets['medications'].empty:
                med_df = self.datasets['medications'][
                    self.datasets['medications']['Disease'] == disease
                ]
                if not med_df.empty:
                    recommendations['medications'] = med_df['Medication'].tolist()
            
            # Get diets
            if not self.datasets['diets'].empty:
                diet_df = self.datasets['diets'][
                    self.datasets['diets']['Disease'] == disease
                ]
                if not diet_df.empty:
                    recommendations['diets'] = diet_df['Diet'].tolist()
            
            # Get workouts
            if not self.datasets['workout'].empty:
                workout_df = self.datasets['workout'][
                    self.datasets['workout']['disease'] == disease
                ]
                if not workout_df.empty:
                    recommendations['workouts'] = workout_df['workout'].tolist()
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
        
        return recommendations
    
    
    def display_results(self, recommendations: Dict[str, any], 
                       valid_symptoms: List[str], 
                       invalid_symptoms: List[str]):
        """
        Display results in a formatted way
        
        Args:
            recommendations: Dictionary of recommendations
            valid_symptoms: List of valid symptoms used
            invalid_symptoms: List of invalid symptoms
        """
        print("\n" + "="*70)
        print("HEALTHCARE RECOMMENDATION SYSTEM - INFERENCE RESULTS")
        print("="*70)
        
        # Display warnings
        if invalid_symptoms:
            print("\n⚠️  WARNING: The following symptoms were not recognized:")
            for symptom in invalid_symptoms:
                print(f"   - {symptom}")
        
        # Display input symptoms
        print(f"\n📋 Symptoms Analyzed: {len(valid_symptoms)}")
        print(f"   {', '.join(valid_symptoms)}")
        
        # Display prediction
        print(f"\n🔍 PREDICTED DISEASE: {recommendations['disease']}")
        
        # Display description
        if recommendations['description']:
            print(f"\n📖 Description:")
            print(f"   {recommendations['description']}")
        
        # Display precautions
        if recommendations['precautions']:
            print(f"\n🛡️  Precautions:")
            for i, precaution in enumerate(recommendations['precautions'], 1):
                print(f"   {i}. {precaution}")
        
        # Display medications
        if recommendations['medications']:
            print(f"\n💊 Recommended Medications:")
            for medication in recommendations['medications']:
                print(f"   • {medication}")
        
        # Display diets
        if recommendations['diets']:
            print(f"\n🥗 Dietary Recommendations:")
            for diet in recommendations['diets']:
                print(f"   • {diet}")
        
        # Display workouts
        if recommendations['workouts']:
            print(f"\n💪 Lifestyle & Workout Suggestions:")
            for workout in recommendations['workouts']:
                print(f"   • {workout}")
        
        # Medical disclaimer
        print("\n" + "="*70)
        print("⚕️  MEDICAL DISCLAIMER")
        print("="*70)
        print("This is an AI-based prediction system for educational purposes only.")
        print("It should NOT replace professional medical advice, diagnosis, or treatment.")
        print("Always consult with a qualified healthcare provider for medical concerns.")
        print("="*70 + "\n")
    
    
    def run_inference(self, symptoms_input: str):
        """
        Run complete inference pipeline
        
        Args:
            symptoms_input: Comma-separated string of symptoms
        """
        # Parse input
        symptoms_list = [s.strip() for s in symptoms_input.split(',')]
        
        # Validate symptoms
        valid_symptoms, invalid_symptoms = self.validate_symptoms(symptoms_list)
        
        if not valid_symptoms:
            print("\n❌ ERROR: No valid symptoms provided!")
            print("Please check your symptom names and try again.")
            print("\nExample valid symptoms:")
            print("  itching, skin_rash, fever, headache, cough")
            return
        
        # Predict disease
        disease = self.predict_disease(valid_symptoms)
        
        if not disease:
            print("\n❌ ERROR: Could not predict disease.")
            return
        
        # Get recommendations
        recommendations = self.get_recommendations(disease)
        
        # Display results
        self.display_results(recommendations, valid_symptoms, invalid_symptoms)


def main():
    """
    Main function for command-line usage
    """
    print("\n" + "="*70)
    print("HEALTHCARE RECOMMENDATION SYSTEM - INFERENCE TOOL")
    print("="*70)
    
    # Initialize system
    system = MedicalInferenceSystem()
    
    # Load model
    if not system.load_model():
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    # Load datasets
    if not system.load_datasets():
        print("⚠️  Warning: Some datasets could not be loaded.")
        print("Recommendations may be incomplete.")
    
    # Interactive mode
    if len(sys.argv) > 1:
        # Command-line argument mode
        symptoms_input = ' '.join(sys.argv[1:])
        system.run_inference(symptoms_input)
    else:
        # Interactive mode
        print("\nEnter symptoms separated by commas")
        print("Example: itching, skin_rash, fever, headache")
        print("(Type 'quit' to exit)\n")
        
        while True:
            try:
                symptoms_input = input("Enter symptoms: ").strip()
                
                if symptoms_input.lower() in ['quit', 'exit', 'q']:
                    print("Exiting... Stay healthy! 👋")
                    break
                
                if not symptoms_input:
                    print("⚠️  Please enter at least one symptom.\n")
                    continue
                
                system.run_inference(symptoms_input)
                print("\n" + "-"*70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting... Stay healthy! 👋")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                print(f"\n❌ An error occurred: {str(e)}\n")


if __name__ == "__main__":
    main()
