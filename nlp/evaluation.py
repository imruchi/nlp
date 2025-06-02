import pandas as pd
import re
import json
from json_repair import repair_json
from transformers import pipeline
from sklearn.metrics import f1_score


# --- Helper Functions ---

def fix_json_with_library(broken_json_text):
    try:
        fixed_json = repair_json(broken_json_text)
        return json.loads(fixed_json)
    except Exception as e:
        print(f"Could not repair JSON: {e}")
        return None


def fixed_json(df):
    """Extract and fix malformed JSON from model responses."""
    print("Fixing JSON responses...")
    all_fixed_jsons = []

    # Filter rows where predicted_direction is not increase/decrease
    filtered_df = df[
        ~df['predicted_direction'].str.lower().isin(['increase', 'decrease'])
    ]

    for i, row in filtered_df.iterrows():
        full_response_text = row.full_model_response
        last_text = full_response_text.split('assistant')[-1]

        # Try primary pattern (```json ... ```)
        pattern1 = r'```json\s*\n(.*?)\n```'
        matches1 = re.findall(pattern1, last_text, re.DOTALL)

        if matches1:
            json_block = matches1[0]
        else:
            # Fallback: basic {...} block
            pattern2 = r'\{[^{}]*\}'
            matches2 = re.findall(pattern2, last_text, re.DOTALL)

            if matches2:
                json_block = matches2[0]
            else:
                print(f"No JSON found for row {i}")
                continue  # Skip if no JSON found

        fixed = fix_json_with_library(json_block)

        # Store metadata alongside fixed JSON
        all_fixed_jsons.append({
            "row_index": i,
            "company_id": row.get("company_id", "Unknown"),
            "year": row.get("year", "Unknown"),
            "fixed_json": fixed
        })

    print(f"Total JSONs fixed: {len(all_fixed_jsons)}")
    return all_fixed_jsons


def zero_shot(rows):
    """Classify using reason or direction_of_eps_change via zero-shot classification."""
    print("Running Zero-Shot Classification...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    predictions = []

    for idx, row in enumerate(rows):
        print(f"Processing Row {idx + 1}/{len(rows)}")

        fixed_json = row.get("fixed_json", {})
        reason = fixed_json.get("reason", "").strip()
        fallback = fixed_json.get("direction_of_eps_change", "").strip()

        if reason:
            text_to_classify = reason
            source = "reason"
        elif fallback:
            text_to_classify = fallback
            source = "direction_of_eps_change"
        else:
            print(f"Skipping row {idx}: No text to classify.")
            predictions.append({
                "row_index": row["row_index"],
                "predicted_label": "Unknown"
            })
            continue

        try:
            result = classifier(text_to_classify, candidate_labels=["Increase", "Decrease"])
            predicted_label = result["labels"][0].lower()  # normalize to lowercase
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            predicted_label = "Error"

        predictions.append({
            "row_index": row["row_index"],
            "predicted_label": predicted_label
        })

    return predictions


# --- Main Workflow ---

# Load evaluation data
json_path = "predictions_checkpoint_1800.json"
eval_df = pd.read_json(json_path)

# Step 1: Fix malformed JSONs
rows = fixed_json(eval_df)

# Step 2: Run zero-shot classification
predictions = zero_shot(rows)

# Step 3: Update original DataFrame with new predictions
pred_df = pd.DataFrame(predictions).set_index("row_index")
eval_df = eval_df.copy()  # Avoid SettingWithCopyWarning
eval_df.loc[pred_df.index, 'predicted_direction'] = pred_df['predicted_label'].values

# Step 4: Clean up labels
eval_df['predicted_direction'] = eval_df['predicted_direction'].astype(str).str.strip().str.lower()
eval_df['actual_label'] = eval_df['actual_label'].astype(str).str.strip().str.lower()

# Step 5: Remove neutral and default cases
eval_df = eval_df[eval_df['predicted_direction'] != "<increase/decrease>"]
eval_df = eval_df[eval_df['actual_label'] != "neutral"]

eval_df['actual_label'].value_counts()
print(eval_df['predicted_direction'].value_counts())

# Step 6: Compute F1 Score
try:
    f1 = f1_score(
        eval_df['actual_label'],
        eval_df['predicted_direction'],
        pos_label='increase',
        average='binary'
    )
    print(f"🔍 Final F1 Score (for label='increase'): {f1:.4f}")
except ValueError as ve:
    print(f"⚠️ Error computing F1 score: {ve}")