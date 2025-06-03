import pandas as pd
import re
import json
import logging
from json_repair import repair_json
from transformers import pipeline
from sklearn.metrics import f1_score
import sys

# --- Logging Setup ---

log_formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = logging.FileHandler('evaluation.log')
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# --- Helper Functions ---

def fix_json_with_library(broken_json_text):
    try:
        fixed_json = repair_json(broken_json_text)
        return json.loads(fixed_json)
    except Exception as e:
        logging.error(f"Could not repair JSON: {e}")
        return None


def fixed_json(df):
    logging.info("Starting JSON repair process...")
    all_fixed_jsons = []

    filtered_df = df[
        ~df['predicted_direction'].str.lower().isin(['increase', 'decrease'])
    ]
    logging.info(f"Filtering complete. Total rows needing repair: {len(filtered_df)}")

    for i, row in filtered_df.iterrows():
        full_response_text = row.full_model_response
        last_text = full_response_text.split('assistant')[-1]

        if last_text.strip():
            pattern1 = r'```json\s*\n(.*?)\n```'
            matches1 = re.findall(pattern1, last_text, re.DOTALL)

            if matches1:
                json_block = matches1[0]
            else:
                pattern2 = r'\{[^{}]*\}'
                matches2 = re.findall(pattern2, last_text, re.DOTALL)

                if matches2:
                    json_block = matches2[0]
                else:
                    company_id = row.get("company_id")
                    year = row.get("year")
                    logging.warning(f"No JSON found for company_id={company_id}, year={year}")
                    continue

            fixed = fix_json_with_library(json_block)

            all_fixed_jsons.append({
                "row_index": i,
                "company_id": row.get("company_id", "Unknown"),
                "year": row.get("year", "Unknown"),
                "fixed_json": fixed
            })
        else:
            logging.warning(f"No text after 'assistant' for row {i}")

    logging.info(f"Total JSONs fixed: {len(all_fixed_jsons)}")
    return all_fixed_jsons


def zero_shot(rows):
    logging.info("Starting Zero-Shot Classification...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    predictions = []
    skipped_count = 0

    for idx, row in enumerate(rows):
        logging.debug(f"Processing Row {idx + 1}/{len(rows)}")

        fixed_json = row.get("fixed_json", {})
        reason = fixed_json.get("reason", "").strip()
        fallback = fixed_json.get("direction_of_eps_change", "").strip()

        if reason:
            text_to_classify = reason
        elif fallback:
            text_to_classify = fallback
        else:
            logging.warning(f"Skipping row {idx}: No text to classify.")
            skipped_count += 1
            predictions.append({
                "row_index": row["row_index"],
                "predicted_label": "unknown"
            })
            continue

        try:
            result = classifier(text_to_classify, candidate_labels=["Increase", "Decrease"])
            predicted_label = result["labels"][0].lower()
        except Exception as e:
            logging.error(f"Classification error at row {idx}: {str(e)}")
            predicted_label = "error"

        predictions.append({
            "row_index": row["row_index"],
            "predicted_label": predicted_label
        })

    logging.info(f"Classification complete for {len(predictions)} rows.")
    logging.info(f"Skipped {skipped_count} rows due to missing classification text.")
    return predictions


def evaluate_predictions(json_path, fixed_json_func, zero_shot_func):
    logging.info(f"Loading evaluation data from {json_path}...")
    eval_df = pd.read_json(json_path)
    logging.info(f"Loaded {len(eval_df)} rows.")

    rows = fixed_json_func(eval_df)
    predictions = zero_shot_func(rows)

    logging.info("Merging predictions into original dataframe...")
    pred_df = pd.DataFrame(predictions).set_index("row_index")
    eval_df = eval_df.copy()
    eval_df.loc[pred_df.index, 'predicted_direction'] = pred_df['predicted_label'].values

    eval_df['predicted_direction'] = eval_df['predicted_direction'].astype(str).str.strip().str.lower()
    eval_df['actual_label'] = eval_df['actual_label'].astype(str).str.strip().str.lower()

    before = len(eval_df)
    logging.info(f"Filtering invalid and neutral rows... Total before: {before}")
    eval_df = eval_df[eval_df['predicted_direction'].isin(['increase', 'decrease'])]
    eval_df = eval_df[eval_df['actual_label'] != "neutral"]
    after = len(eval_df)
    logging.info(f"Remaining rows after filtering: {after} (Removed: {before - after})")

    logging.info("Label distribution:")
    logging.info(f"Actual Labels:\n{eval_df['actual_label'].value_counts()}")
    logging.info(f"Predicted Directions:\n{eval_df['predicted_direction'].value_counts()}")

    # Save 10 negative examples
    negative_df = eval_df[eval_df['predicted_direction'] != eval_df['actual_label']].head(10)
    with open("positive_samples.json", "w", encoding='utf-8') as f:
        json.dump(negative_df.to_dict(orient='records'), f, indent=4)
    logging.info("Saved 10 negative (misclassified) samples to positive_samples.json")

    try:
        f1 = f1_score(
            eval_df['actual_label'],
            eval_df['predicted_direction'],
            pos_label='increase',
            average='binary'
        )
        logging.info(f"Final F1 Score (label='increase'): {f1:.4f}")
    except ValueError as ve:
        logging.error(f"Error computing F1 score: {ve}")
        f1 = None

    return eval_df, f1


# --- Run Evaluation ---

if __name__ == "__main__":
    logging.info("Starting Evaluation Pipeline...")
    eval_df, final_f1 = evaluate_predictions(
        json_path="nlp/output/predictions.json",
        fixed_json_func=fixed_json,
        zero_shot_func=zero_shot
    )
    logging.info("Evaluation Pipeline Complete.")
