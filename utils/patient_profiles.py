import json

PATIENT_PROFILES = {
    "elena": {
        "name": "Elena Rodriguez",
        "language": "Spanish",
        "age": 65,
        "diagnoses": ["Type 2 Diabetes", "Hypertension", "Hyperlipidemia", "Osteoarthritis"],
        "medications": ["Metformin", "Lisinopril", "Atorvastatin"],
        "adherence": "often forgets to take Metformin",
        "culture": "Believes in traditional herbal remedies like aloe vera, cinnamon, bitter melon",
        "social": "Sedentary lifestyle, prefers natural remedies, widowed, lives in Little Havana"
    },
    "miguel": {
        "name": "Miguel Hernandez",
        "language": "Spanish",
        "age": 16,
        "diagnoses": ["Type 2 Diabetes"],
        "medications": ["Metformin"],
        "adherence": "struggles with adherence due to side effects",
        "culture": "Family uses remedies like manzanilla, influenced by folk illnesses like empacho",
        "social": "Teenager, prefers fast food, plays video games, limited physical activity"
    },
    "carmen": {
        "name": "Carmen Rivera",
        "language": "Spanish",
        "age": 48,
        "diagnoses": ["Asthma", "Hypertension", "Type 2 Diabetes"],
        "medications": ["Albuterol", "Fluticasone", "Lisinopril", "Metformin"],
        "adherence": "occasionally misses doses",
        "culture": "Believes in hot/cold food theory, uses herbal teas",
        "social": "Works as cashier, lives in high-pollution area, caregiver to elderly mother"
    }
}

def get_patient_profile(key: str):
    return PATIENT_PROFILES.get(key.lower())
