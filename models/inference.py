import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from data import process_data
import yaml



model_path = "./qwen-7b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval()

device = "mps" 


def parse_description(description):
    """
    Extracts balance sheet and income statement using regex.
    Returns two strings: balance_sheet_text, income_statement_text
    """
    sections = re.split(r'Below is the (Balance Sheet:|Income Statement:)', description)

    balance_sheet = ""
    income_statement = ""

    for i in range(1, len(sections), 2):
        title = sections[i].strip()
        table_text = sections[i + 1].strip()
        parsed_table = parse_table_string(table_text)
        if 'Balance Sheet' in title:
            balance_sheet = parsed_table
        elif 'Income Statement' in title:
            income_statement = parsed_table

    return balance_sheet, income_statement


def parse_table_string(table_str):
    """
    Converts pipe-separated table into key-value format
    """
    lines = [line.strip() for line in table_str.split("\n") if line.strip()]
    result = []

    for line in lines:
        if '|' not in line:
            continue
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(cells) < 2:
            continue
        key = cells[0]
        values = " | ".join(cells[1:])
        result.append(f"{key}: {values}")

    return "\n".join(result)


def build_prompt(balance_sheet_text, income_statement_text):
    with open("nlp/config/prompts.yaml", "r") as file:
        prompts = yaml.safe_load(file)
        prompt = prompts['prompt_1']
    return prompt


def run_inference(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=500,
            do_sample=False,
            num_beams=1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def parse_response(output):
    match = re.search(r'\{(?:[^{}]|(?R))*\}', output, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str.replace('\n', ' '))
            return {
                "direction": data.get("Direction of EPS Change", "Unknown"),
                "magnitude": data.get("Magnitude of Change", "Unknown"),
                "certainty": data.get("Certainty of Assessment", "Unknown"),
                "reason": data.get("Reason", "No reason provided."),
                "full_response": output
            }
        except Exception as e:
            print("JSON parsing error:", str(e))
            return {
                "direction": "Unknown",
                "magnitude": "Unknown",
                "certainty": "Unknown",
                "reason": "Error in parsing JSON.",
                "full_response": output
            }
    else:
        return {
            "direction": "Unknown",
            "magnitude": "Unknown",
            "certainty": "Unknown",
            "reason": "No valid JSON found.",
            "full_response": output
        }

