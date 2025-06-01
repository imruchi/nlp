import pandas as pd
eval_df = pd.read_json("predictions_checkpoint_1800.json")

import re
import json
from json_repair import repair_json


def fix_json_with_library(broken_json_text):
    """
    Use json-repair library to fix common JSON issues.
    """
    try:
        # Attempt to repair the JSON
        fixed_json = repair_json(broken_json_text)
        return json.loads(fixed_json)
    except Exception as e:
        print(f"Could not repair JSON: {e}")
        return None
    
    for i, row in df.iterrows():
        is_error = (row.predicted_direction == "error")
        if not is_error:
            continue

def fixed_json(row):
    all_fixed_jsons = []

    
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
            print(f"❌ No JSON found for row {i}")
            

    fixed = fix_json_with_library(json_block)

    # ✅ Store metadata alongside fixed JSON
    all_fixed_jsons.append({
        "row_index": i,
        "company_id": row.get("company_id", "Unknown"),
        "year": row.get("year", "Unknown"),
        "fixed_json": fixed
    })

return all_fixed_jsons
