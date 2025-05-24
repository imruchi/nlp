import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import yaml
import os

# --- Configuration Loading ---
CONFIG_PATH = "config/model_config.yaml"
PROMPTS_PATH = "config/prompts.yaml"
DATA_PATH = "nlp/data/process_data.json" # Adjusted path
OUTPUT_DIR = "nlp/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions.json")

# Load configurations globally
with open(CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)
with open(PROMPTS_PATH, "r") as f:
    prompts_config = yaml.safe_load(f)

# --- Model and Tokenizer Initialization ---
model_name = model_config['model_name']
print(f"Loading tokenizer and model for: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Handles device placement
    torch_dtype=torch.float16, # Keeping float16 as per original
    trust_remote_code=True
).eval()
print(f"Model loaded on device: {model.device}")

# Inference parameters from model_config.yaml
INFERENCE_PARAMS = model_config.get('inference_config', {})
# Ensure essential parameters have defaults if not in config
INFERENCE_PARAMS.setdefault('max_new_tokens', 512)
INFERENCE_PARAMS.setdefault('do_sample', True) # Defaulting to True as in example config
INFERENCE_PARAMS.setdefault('num_beams', 1) # Defaulting as in example config

def build_prompt(balance_income_sheet: str) -> str:
    """
    Loads prompt_1 from prompts.yaml and formats it with the provided financial statement text.
    Args:
        balance_income_sheet (str): The full financial statement text to insert into the prompt.
    Returns:
        str: The formatted prompt string.
    Raises:
        ValueError: If prompt_1 is missing or not properly formatted in prompts.yaml.
    """
    prompt_template_list = prompts_config.get('prompt_1')
    if not prompt_template_list or not isinstance(prompt_template_list, list):
        raise ValueError("prompt_1 not found or not a list in prompts.yaml")
    prompt_template = prompt_template_list[0]
    try:
        return prompt_template.format(balance_income_sheet=balance_income_sheet)
    except KeyError as e:
        raise ValueError(f"Missing placeholder for {e} in prompt_1. Provided: balance_income_sheet")

def run_inference(prompt: str):
    print("RUNNING INFERENCE")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # Move inputs to model's device

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"), # Include attention_mask if available
            **INFERENCE_PARAMS # Spread inference parameters from config
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def parse_response(output: str):
    print("PARSING RESPONSE")
    # Regex to find JSON object, accounts for nested structures and varying whitespace
    match = re.search(r'\{\s*("Direction of EPS Change"\s*:.+?)\s*\}', output, re.DOTALL | re.IGNORECASE)
    
    if match:
        json_str_dirty = match.group(0) # Get the entire matched JSON block
        
        # Clean up common issues:
        # 1. Remove trailing commas before '}'
        json_str_cleaned = re.sub(r',\s*(\}|\])', r'\1', json_str_dirty)
        # 2. Ensure newlines within strings are escaped or handled if JSON parser is strict
        #    (Python's json.loads is generally okay with actual newlines if they are part of the string value)
        #    For this specific structure, replacing newlines not inside quotes with spaces might be safer
        #    if the LLM tends to break lines unexpectedly. However, the primary issue is often trailing commas.
        
        try:
            # Attempt to parse the cleaned JSON string
            data = json.loads(json_str_cleaned)
            return {
                "direction": data.get("Direction of EPS Change", "Unknown"),
                "magnitude": data.get("Magnitude of Change", "Unknown"),
                "certainty": data.get("Certainty of Assessment", "Unknown"),
                "reason": data.get("Reason", "No reason provided."),
                "full_response": output # Always return the full raw output
            }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Problematic JSON string (cleaned): '{json_str_cleaned}'")
            return {
                "direction": "Error",
                "magnitude": "Error",
                "certainty": "Error",
                "reason": f"Error in parsing JSON: {str(e)}. Original match: {json_str_dirty}",
                "full_response": output
            }
    else:
        print("No valid JSON object found in the response.")
        return {
            "direction": "No JSON",
            "magnitude": "No JSON",
            "certainty": "No JSON",
            "reason": "No valid JSON object found in the model response.",
            "full_response": output
        }

def process_financial_data():
    """
    Loads financial data, runs inference for each entry, and saves results.
    """
    print(f"Loading financial data from {DATA_PATH}")
    try:
        with open(DATA_PATH, 'r') as f:
            financial_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {DATA_PATH}")
        return

    all_results = []
    count = 0

    for company_id, company_data in financial_data.items():
        if "std_dev_sales" in company_data: # Skip metadata like std_dev_sales
            # Iterate over years which are keys like "1973.0"
            for year, year_data in company_data.items():
                if year == "std_dev_sales": # Double check to skip
                    continue
                if isinstance(year_data, dict) and "description" in year_data and "label" in year_data:
                    count += 1
                    print(f"\nProcessing Company ID: {company_id}, Year: {year} ({count})")
                    
                    description = year_data['description']
                    actual_label = year_data['label'].strip().lower()

                    prompt = build_prompt(description)
                    # print(f"Generated Prompt:\n{prompt[:500]}...") # Optional: print part of the prompt

                    response_text = run_inference(prompt)
                    print(f"Raw Model Response:\n{response_text}") # Optional: print raw response

                    parsed_output = parse_response(response_text)
                    predicted_direction = str(parsed_output.get("direction", "Unknown")).strip().lower()
                    
                    is_match = False
                    if predicted_direction in ["increase", "decrease", "stay the same"]: 
                        is_match = (predicted_direction == actual_label)
                    else:
                        print(f"Warning: Predicted direction '{predicted_direction}' is not a standard value.")


                    result_entry = {
                        "company_id": company_id,
                        "year": year,
                        "actual_label": actual_label,
                        "predicted_direction": predicted_direction,
                        "magnitude": parsed_output.get("magnitude", "Unknown"),
                        "certainty": parsed_output.get("certainty", "Unknown"),
                        "reason": parsed_output.get("reason", "Unknown"),
                        "is_match": is_match,
                        "full_model_response": parsed_output['full_response'] # Storing the raw response from parse_output
                    }
                    all_results.append(result_entry)
                    
                    # Optional: print immediate result for monitoring
                    print(f"Company: {company_id}, Year: {year}, Actual: {actual_label}, Predicted: {predicted_direction}, Match: {is_match}")

    # Save results
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"\nSaving all results to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Processing complete. {len(all_results)} entries processed and saved.")

if __name__ == "__main__":
    process_financial_data()

