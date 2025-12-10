import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

# Test 1: Même compteur, heures différentes (doit varier !)
print("=" * 60)
print("TEST 1: Variations horaires (même compteur)")
print("=" * 60)

test_cases_hours = [
    {"hour": 2, "label": "2h du matin"},
    {"hour": 8, "label": "8h (pointe matin)"},
    {"hour": 12, "label": "12h (midi)"},
    {"hour": 18, "label": "18h (pointe soir)"},
    {"hour": 23, "label": "23h (nuit)"},
]

counter_id = "urn:ngsi-ld:EcoCounter:X2H22104769"
predictions_hours = []

for test in test_cases_hours:
    params = {
        "year": 2025,
        "month": 12,
        "day": 10,
        "hour": test["hour"],
        "weekday": 2,  # Mercredi
        "counter_id": counter_id
    }
    
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        pred = response.json().get("prediction", 0)
        predictions_hours.append(pred)
        print(f"  {test['label']:20s} → {pred:.6f}")
    else:
        print(f"  {test['label']:20s} → ERREUR {response.status_code}")

# Vérifier si les prédictions varient
if len(set(predictions_hours)) == 1:
    print("\n❌ PROBLÈME: Toutes les prédictions sont identiques !")
    print("   → Le modèle n'utilise PAS la feature 'hour'")
else:
    print(f"\n✅ OK: {len(set(predictions_hours))} valeurs différentes détectées")

# Test 2: Même heure, jours différents (doit varier !)
print("\n" + "=" * 60)
print("TEST 2: Variations selon le jour de la semaine")
print("=" * 60)

test_cases_days = [
    {"weekday": 0, "label": "Lundi"},
    {"weekday": 2, "label": "Mercredi"},
    {"weekday": 4, "label": "Vendredi"},
    {"weekday": 5, "label": "Samedi"},
    {"weekday": 6, "label": "Dimanche"},
]

predictions_days = []

for test in test_cases_days:
    params = {
        "year": 2025,
        "month": 12,
        "day": 10,
        "hour": 12,
        "weekday": test["weekday"],
        "counter_id": counter_id
    }
    
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        pred = response.json().get("prediction", 0)
        predictions_days.append(pred)
        print(f"  {test['label']:20s} → {pred:.6f}")
    else:
        print(f"  {test['label']:20s} → ERREUR {response.status_code}")

if len(set(predictions_days)) == 1:
    print("\n❌ PROBLÈME: Toutes les prédictions sont identiques !")
    print("   → Le modèle n'utilise PAS la feature 'weekday'")
else:
    print(f"\n✅ OK: {len(set(predictions_days))} valeurs différentes détectées")

# Test 3: Compteurs différents (doit varier !)
print("\n" + "=" * 60)
print("TEST 3: Variations selon le compteur")
print("=" * 60)

counters = [
    "urn:ngsi-ld:EcoCounter:X2H22104769",
    "urn:ngsi-ld:EcoCounter:X2H22043032",
    "urn:ngsi-ld:EcoCounter:X2H22043031",
]

predictions_counters = []

for counter in counters:
    params = {
        "year": 2025,
        "month": 12,
        "day": 10,
        "hour": 12,
        "weekday": 2,
        "counter_id": counter
    }
    
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        pred = response.json().get("prediction", 0)
        predictions_counters.append(pred)
        print(f"  {counter[-10:]:20s} → {pred:.6f}")
    else:
        print(f"  {counter[-10:]:20s} → ERREUR {response.status_code}")

if len(set(predictions_counters)) == 1:
    print("\n❌ PROBLÈME: Toutes les prédictions sont identiques !")
    print("   → Le modèle n'utilise PAS la feature 'counter_id'")
else:
    print(f"\n✅ OK: {len(set(predictions_counters))} valeurs différentes détectées")

# Résumé
print("\n" + "=" * 60)
print("📊 RÉSUMÉ DU DIAGNOSTIC")
print("=" * 60)

all_predictions = predictions_hours + predictions_days + predictions_counters
unique_predictions = len(set(all_predictions))

if unique_predictions == 1:
    print("❌ CRITIQUE: Le modèle retourne TOUJOURS la même valeur !")
    print("   Causes possibles:")
    print("   1. Les features ne sont pas passées au modèle")
    print("   2. Le modèle n'a pas convergé (retourne la moyenne)")
    print("   3. Le preprocessing n'est pas appliqué")
    print("\n🔧 Actions:")
    print("   - Vérifiez votre code d'API (endpoint /predict)")
    print("   - Assurez-vous que le modèle est bien chargé")
    print("   - Testez le modèle directement (sans API)")
elif unique_predictions < 5:
    print(f"⚠️ ATTENTION: Seulement {unique_predictions} valeurs uniques détectées")
    print("   Le modèle varie peu, vérifiez les features importantes")
else:
    print(f"✅ OK: {unique_predictions} valeurs uniques détectées")
    print("   Le modèle semble prendre en compte les features !")
    print(f"   Plage: {min(all_predictions):.6f} → {max(all_predictions):.6f}")