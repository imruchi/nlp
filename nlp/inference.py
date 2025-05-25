import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import yaml
import os
from huggingface_hub import login # snapshot_download removed
import logging # Added for more verbose logging

# --- Global Constants ---
# SCRIPT_DIR can be global as it defines the base for all relative paths within the script's module.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Setup Logging ---
# Configure root logger to show INFO messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# Set transformers logger to DEBUG for more verbose output during model loading
logging.getLogger("transformers").setLevel(logging.DEBUG)
# Silence some overly verbose loggers if needed, e.g., from huggingface_hub
logging.getLogger("huggingface_hub.file_download").setLevel(logging.INFO)

try:
    login(token="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")
    print("✓ Successfully logged into HuggingFace!")
except:
    print("Login Failed !")

# --- Function Definitions ---

def build_prompt(balance_income_sheet: str, prompts_config: dict) -> str:
    """
    Loads prompt_1 from prompts_config and formats it with the provided financial statement text.
    Args:
        balance_income_sheet (str): The full financial statement text to insert into the prompt.
        prompts_config (dict): The loaded prompts configuration.
    Returns:
        str: The formatted prompt string.
    Raises:
        ValueError: If prompt_1 is missing or not properly formatted in prompts_config.
    """
    prompt_template = prompts_config.get('prompt_1')
    # if not prompt_template_list or not isinstance(prompt_template_list, list):
    #     raise ValueError("prompt_1 not found or not a list in prompts_config")
    # prompt_template = prompt_template_list[0]
    try:
        return prompt_template.format(balance_income_sheet=balance_income_sheet)
    except KeyError as e:
        raise ValueError(f"Missing placeholder for {e} in prompt_1. Provided: balance_income_sheet")

def run_inference(prompt: str, tokenizer, model, inference_params: dict):
    """
    Runs inference using the provided tokenizer, model, and parameters.
    Args:
        prompt (str): The input prompt for the model.
        tokenizer: The initialized Hugging Face tokenizer.
        model: The initialized Hugging Face model.
        inference_params (dict): Dictionary of parameters for model.generate().
    Returns:
        str: The model's response text.
    """
    print("RUNNING INFERENCE")
    
    
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Set to False for non-thinking mode
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # Move inputs to model's device

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            **inference_params # Spread inference parameters from config
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def parse_response(output: str):
    """
    Parses the model's raw output string to extract a JSON object from ```json code blocks.
    Args:
        output (str): The raw output string from the model.
    Returns:
        dict: A dictionary containing parsed data or error information.
    """
    print("PARSING RESPONSE")
    
    # Look for JSON between ```json and ```
    # Use re.findall to get all matches and then take the last one,
    # ensuring we get the JSON output from the model, not from an echoed prompt.
    matches = re.findall(r'```json\s*(.*?)\s*```', output, re.DOTALL | re.IGNORECASE)
    
    if matches:
        json_str = matches[-1].strip() # Get the content of the last JSON block
        
        # Remove block comments /* ... */
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        # Remove line comments // ...
        json_str = re.sub(r'//.*', '', json_str)
        
        # Remove trailing commas from objects and arrays
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
        
        try:
            # Set strict=False to allow unescaped control characters like newlines within strings
            data = json.loads(json_str, strict=False)
            
            # Handle both formats
            direction = data.get("direction_of_eps_change") or data.get("Direction of EPS Change", "Unknown")
            magnitude = data.get("magnitude_of_change") or data.get("Magnitude of Change", "Unknown") 
            certainty = data.get("certainty_of_assessment") or data.get("Certainty of Assessment", "Unknown")
            reason = data.get("reason") or data.get("Reason", "No reason provided.")
            
            return {
                "direction": direction,
                "magnitude": magnitude,
                "certainty": certainty,
                "reason": reason,
                "full_response": output
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            return {
                "direction": "Error", "magnitude": "Error", "certainty": "Error",
                "reason": f"Error in parsing JSON: {str(e)}",
                "full_response": output
            }
    else:
        print("No ```json code block found in the response.")
        return {
            "direction": "No JSON", "magnitude": "No JSON", "certainty": "No JSON",
            "reason": "No ```json code block found in the model response.",
            "full_response": output
        }

def process_financial_data(data_path: str, output_dir: str, output_file: str, 
                           prompts_config: dict, tokenizer, model, inference_params: dict):
    """
    Loads financial data, runs inference for each entry, and saves results.
    Args:
        data_path (str): Path to the financial data JSON file.
        output_dir (str): Directory to save output files.
        output_file (str): Path to save the predictions JSON file.
        prompts_config (dict): Loaded prompts configuration.
        tokenizer: Initialized Hugging Face tokenizer.
        model: Initialized Hugging Face model.
        inference_params (dict): Parameters for model inference.
    """
    logging.info(f"Loading financial data from {data_path}")
    try:
        with open(data_path, 'r') as f:
            financial_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_path}")
        return

    all_results = []
    count = 0

    for company_id, company_data in financial_data.items():
        for year, year_data in company_data.items():
            if year == "std_dev_sales": continue # Skip non-data entries
            
            if isinstance(year_data, dict) and "description" in year_data and "label" in year_data:
                count += 1
                # logging.info(f"\nProcessing Company ID: {company_id}, Year: {year} ({count})")
                
                description = year_data['description']
                actual_label = year_data['label'].strip().lower()
                
                # Call build_prompt with prompts_config
                prompt = build_prompt(description, prompts_config)
                
                # Call run_inference with tokenizer, model, and inference_params
                response_text = run_inference(prompt, tokenizer, model, inference_params)
                # print(f"Raw Model Response:\n{response_text}")
                
                parsed_output = parse_response(response_text)
                predicted_direction = str(parsed_output.get("direction", "Unknown")).strip().lower()
                
                is_match = (predicted_direction == actual_label) if predicted_direction in ["increase", "decrease", "stay the same"] else False
                if predicted_direction not in ["increase", "decrease", "stay the same"]:
                    print(f"Warning: Predicted direction '{predicted_direction}' is not a standard value.")
                    
                result_entry = {
                    "company_id": company_id, "year": year, "actual_label": actual_label,
                    "predicted_direction": predicted_direction, "magnitude": parsed_output.get("magnitude", "Unknown"),
                    "certainty": parsed_output.get("certainty", "Unknown"), "reason": parsed_output.get("reason", "Unknown"),
                    "is_match": is_match, "full_model_response": parsed_output['full_response']
                }
                all_results.append(result_entry)
                logging.info(f"Company: {company_id}, Year: {year}, Actual: {actual_label}, Predicted: {predicted_direction}, Match: {is_match}")
                print(f"Inference COMPLETE")

                # Save a checkpoint every 100 predictions
                if count % 100 == 0 and count > 0:
                    checkpoint_file_name = f"predictions_checkpoint_{count}.json"
                    checkpoint_file_path = os.path.join(output_dir, checkpoint_file_name)
                    logging.info(f"Saving checkpoint at count {count} to {checkpoint_file_path}")
                    try:
                        with open(checkpoint_file_path, 'w') as cp_f:
                            json.dump(all_results, cp_f, indent=4)
                        logging.info(f"Checkpoint saved successfully to {checkpoint_file_path}")
                    except Exception as e:
                        logging.error(f"Failed to save checkpoint at count {count}: {e}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.info(f"\nSaving all results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logging.info(f"Processing complete. {len(all_results)} entries processed and saved.")

def check_gpu_memory():
    """Check and display GPU memory information"""
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
    else:
        print("CUDA not available")

def main():
    """
    Main function to orchestrate configuration, model initialization, and data processing.
    """
    # --- GPU Memory Check ---
    check_gpu_memory()
    
    # --- Path Definitions ---
    # Paths are defined relative to SCRIPT_DIR for robustness.
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "model_config.yaml")
    PROMPTS_PATH = os.path.join(SCRIPT_DIR, "config", "prompts.yaml")
    DATA_PATH = os.path.join(SCRIPT_DIR, "data", "process_data.json")
    PATHS_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "paths_config.yaml") # Renamed for clarity

    # Determine output directory for results (predictions.json and checkpoints)
    # Priority: Environment variable INFERENCE_RESULTS_DIR from submission script
    # Fallback: SCRIPT_DIR / "outputs" (relative to this script's location)
    env_results_dir = os.getenv("INFERENCE_RESULTS_DIR")
    if env_results_dir:
        # Use the path from the environment variable.
        # This path is expected to be absolute or resolvable by the system.
        OUTPUT_DIR = env_results_dir
        logging.info(f"Using results output directory from INFERENCE_RESULTS_DIR: {OUTPUT_DIR}")
    else:
        # Fallback to a directory named "outputs" relative to the script's location
        OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
        logging.info(f"INFERENCE_RESULTS_DIR not set. Using default results output directory: {OUTPUT_DIR}")

    # Ensure the determined output directory exists.
    # The submission script should ideally create this, but this is a safeguard.
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info(f"Ensured results output directory exists: {OUTPUT_DIR}")
    except OSError as e:
        logging.error(f"Could not create results output directory {OUTPUT_DIR}: {e}. Check permissions and path.", exc_info=True)
        # If directory creation fails, it's a critical issue.
        print(f"CRITICAL ERROR: Failed to create output directory {OUTPUT_DIR}. Exiting.")
        return # Exit main() if output directory cannot be prepared.

    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions.json")
    logging.info(f"Final predictions will be saved to: {OUTPUT_FILE}")
    
    # test_o = parse_response(output="""system\nYou are a helpful assistant.\nuser\nBased on the following financial statements, assess whether EPS will increase or decrease in the next year:\n\nBelow is the Balance Sheet:\n------------------------------------------------------------------------------------------\n| Account Items                                                  | 1972 | 1973 |\n------------------------------------------------------------------------------------------\n| Total Asset                                                  | 19.907 | 21.771 |\n| Current Assets                                               | 11.326 | 12.969 |\n| Current Liabilities                                          | 3.990 | 4.642 |\n| Cash and Short-Term Investments                              | 2.027 | 1.357 |\n| Receivables                                                  | 4.186 | 5.604 |\n| Inventories                                                  | 4.743 | 5.683 |\n| Other Current Assets                                         | 0.370 | 0.325 |\n| Property, Plant, and Equipment (Net)                         | 7.013 | 7.408 |\n| Investment and Advances (equity)                             | 1.147 | 1.016 |\n| Investment Total (short-term)(Other Investments)             | 1.225 | 1.100 |\n| Intangible Assets                                            | 0.170 | 0.152 |\n| Other Assets                                                 | 0.251 | 0.226 |\n| Total Liabilities                                            | 12.886 | 13.204 |\n| Debt in Current Liabilities                                  | 0.000 | 0.000 |\n| Account Payable                                              | 2.768 | 2.789 |\n| Income Taxes Payable                                         | 0.050 | 1.026 |\n| Other Current Liabilities                                    | 1.172 | 0.826 |\n| Long-term Debt                                               | 7.000 | 7.000 |\n| Deferred Taxes and Investment Tax Credit                     | 0.288 | 0.231 |\n| Other Liabilities                                            | 1.608 | 1.331 |\n| Preferred Stock                                              | 0.000 | 0.000 |\n| Common Stock                                                 | 2.902 | 2.840 |\n| Shareholders' Equity                                         | 7.021 | 8.567 |\n| Stockholders' Equity Total                                   | 0.000 | 0.000 |\n| Noncontrolling Interest                                      | 0.000 | 0.000 |\n| Total Liabilities and Shareholders' Equity                   | 19.907 | 21.771 |\n------------------------------------------------------------------------------------------\n\n\n\nBelow is the Income Statement:\n------------------------------------------------------------------------------------------\n| Account Items                                                  | 1971 | 1972 | 1973 |\n------------------------------------------------------------------------------------------\n| Sales (net)                                                  | 47.033 | 34.362 | 37.750 |\n| Cost of Goods Sold                                           | 33.973 | 22.702 | 24.704 |\n| Gross Profit                                                 | 13.060 | 11.660 | 13.046 |\n| Selling, General and Administrative Expenses                 | 10.548 | 7.551 | 8.532 |\n| Operating Income Before Depreciation                         | 2.512 | 4.109 | 4.514 |\n| Depreciation and Amortization                                | 1.399 | 1.200 | 1.237 |\n| Operating Income After Depreciation                          | 1.113 | 2.909 | 3.277 |\n| Interest and Related Expense                                 | 1.117 | 0.784 | 0.705 |\n| Nonoperating Income (excluding interest income)              | 0.142 | 0.577 | 0.307 |\n| Total Interest Income                                        | 0.000 | 0.000 | 0.000 |\n| Special Items                                                | 0.000 | 0.000 | 0.000 |\n| Pretax Income                                                | 0.138 | 2.702 | 2.879 |\n| Income Taxes (current)                                       | 0.000 | 0.000 | 0.000 |\n| Income Taxes (deferred)                                      | 0.000 | 0.288 | -0.057 |\n| Income Taxes (other)                                         | 0.000 | 0.000 | 0.000 |\n| Income Before Extraordinary Items and Noncontrolling Interest | 0.000 | 0.000 | 0.000 |\n| Noncontrolling Interest                                      | 0.000 | 0.000 | 0.000 |\n| Income Before Extraordinary Items                            | 0.138 | 1.554 | 1.863 |\n| Dividends - Total                                            | 0.000 | 0.000 | 0.000 |\n| Income Before Extraordinary Items - Availiabe Common Stock   | 0.138 | 1.554 | 1.863 |\n| Common Stock Equivalents - Dollar Savings                    | 0.000 | 0.000 | 0.000 |\n| Income Before Extraordinary Items - Adjusted for Common Stock E | 0.138 | 1.554 | 1.863 |\n| Extraordinary Items and Discontinued Operations              | -2.456 | 0.671 | 0.000 |\n| Net Income (Loss)                                            | -2.318 | 2.225 | 1.863 |\n| Earnings per Share - Basic Excluding Extraordinary Items     | 0.040 | 0.500 | 0.640 |\n| Earnings per Share - Diluted Excluding Extraordinary Items   | 0.040 | 0.500 | 0.620 |\n------------------------------------------------------------------------------------------\n\n\nSolve this problem step by step:\n1. Analyze the Balance Sheet:\n   1A. Extract balance sheet data including Year-over-Year percentage changes for each item.\n   1B. Analyze the balance sheet data for key insights and trends.\n\n2. Perform a Cash Flow Analysis:\n   2A. Analyze liquidity and cash movement based on the balance sheet and income statement trends.\n\n3. Analyze the Income Statement:\n   3A. Extract income statement data including Year-over-Year percentage changes for each item.\n   3B. Analyze the income statement data for key insights and trends.\n\n4. Financial Ratio Analysis:\n   4A. Using the balance sheet and income statement data, calculate key financial ratios for the two most recent periods.\n   4B. Analyze the calculated financial ratios for key insights and trends.\n\n5. Final Summary and Future Outlook:\n   5A. Based on the balance sheet, income statement, and financial ratio analysis, produce a comprehensive assessment of the company's financial health and future profitability.\n\n6. EPS Prediction for Next Year:\n   6A. Assess whether EPS is likely to increase or decrease based on findings.\n   6B. First provide the step-wise reasoning and then return the results in JSON format only:\n   {\n     \"direction_of_eps_change\": \"Increase or Decrease\",\n     \"magnitude_of_change\": \"description of magnitude\",\n     \"certainty_of_assessment\": \"description of certainty level\",\n     \"reason\": \"explanation for the prediction\"\n   }\n\nassistant\nTo analyze the potential change in earnings per share (EPS) over the coming year, we'll follow these steps using provided financial information from the previous years: \n\n### Step-by-step Reasoning\n\n#### **Balance Sheet Analysis**\n\n**Year-over-year percentages**: \n- **Total assets**: Increases by \\( \\frac{21,771}{19,907} * 100 = 9\\% \\)\n- **Current assets**: Increases slightly (\\( \\approx 5\\%\\))\n- **Receivables**, **Inventories**: Increase by about \\( \\ \\approx 16\\%\\) respectively\n- **Property, plant & equipment net**: Grows approximately \\( \\ \\approx 4\\%\\)\n\n**Insights**: \n- The overall asset base has grown significantly; however, current liabilities also increased at a similar pace potentially leading to higher debt levels but not necessarily an immediate threat to solvency as they're within manageable limits if sales remain steady or grow with it. The current growth might indicate expansion plans that could improve efficiency or market position once operational costs stabilize post-expansion phase.\n\n#### **Income Statement Analysis**\n\n**Year-over-year percentages**:\n- **Sales/Net revenue**: Decreases by around \\( \\ \\approx 26\\%\\).\n- **Costs of goods sold**: Shrinks more slowly than revenues (\\( \\ \\approx 16\\%\\)), indicating margin improvement.\n- **Gross profit**: Drops due to lower sales despite cost control efforts, suggesting either volume reduction or pricing pressures.\n- **Operating income before depreciation**: Improves with increasing margins possibly indicating efficient operations despite reduced activity.\n- **Depreciation expenses**: Remains relatively constant.\n- **Interest expense**: Reduces which improves operating leverage.\n  \n**Key Insights**: \n- Margins seem to have improved during fiscal year '73, even though sales decreased. This indicates better management of fixed-cost structures like selling, general, and administrative expenses, although there were special extraordinary items negatively impacting pre-tax incomes.\n\n#### **Financial Ratios Calculation**\nFor simplicity here, let\u2019s consider just three common indicators:\n1. **Return on Equity (ROE** : Helps understand how efficiently profits are being converted into shareholder equity.\n2. **Debt-to-Equity Ratio**: Indicates reliance on creditors\u2019 funds vs shareholders'.\n3. **Price/Earnings (P/E Ratio**: Measures valuation relative to earnings.\n\nAssuming ROE grows because of better margin performance while maintaining reasonable D/E ratio ( suggests healthy use of borrowed capital without compromising long term stability, P/E might reflect market expectations correctly given the positive signs in other indicators.\n\n#### **Final Assessment & Prediction**\n\nThe balance sheet shows investments aligning well with growing total assets at a rate comparable to liability increases, implying prudent financial planning supporting business expansion rather than speculative ventures. On the income side, gross profit margins have shown resilience under declining sales volumes, hinting towards good cost controls especially considering the reduction in interest expenses.\n\nGiven the trends analyzed above, assuming stable to improving underlying fundamentals such as continued margin improvements after managing through the dip period, and no significant negative surprises arising out of the currently unaccounted-for special items, one can reasonably infer:\n\n```json\n{\n    \"direction_of_eps_change\": \"Increase\",\n    \"magnitude_of_change\": \"Expected moderate to high single-digit percent increase depending on sales recovery rates.\",\n    \"certainty_of_assessment\": \"Moderate\", /* Considering economic uncertainties and industry-specific risks */\n    \"reason\": \"Improved efficiency and effective cost controls suggest a path toward higher profitability, bolstered further by reduced interest expenses, suggesting potential for an upward trend in EPS.\"\n}\n```\n\nThis conclusion draws upon a scenario where the company successfully navigates through any short-term challenges, focusing on expanding its product range or market presence alongside optimizing existing operations effectively. It assumes sales may recover along with broader economic conditions, benefiting from the strong foundational metrics identified throughout our analysis.""")
    # print(f'JSON parse test: {test_o}')
    

    # --- Configuration Loading ---
    print("Loading configurations...")
    with open(PATHS_CONFIG_PATH, "r") as f:
        path_config = yaml.safe_load(f)
    with open(CONFIG_PATH, "r") as f:
        model_config = yaml.safe_load(f)
    with open(PROMPTS_PATH, "r") as f:
        prompts_config = yaml.safe_load(f)
    print("Configurations loaded.")
    # print(prompts_config.get('prompt_1'))

    # --- Environment Setup & Model Download ---
    # These operations are performed only when main() is executed.
    os.environ['HF_HOME'] = path_config['cache_directory']
    os.environ['TRANSFORMERS_CACHE'] = path_config['cache_directory']

    # The cache_directory will be used by from_pretrained implicitly due to env vars,
    # but we can also pass it explicitly.
    hf_cache_dir = path_config['cache_directory']
    os.makedirs(hf_cache_dir, exist_ok=True) # Ensure cache directory exists
    
    # Removed snapshot_download block and essential_files check

    # --- Model and Tokenizer Initialization ---
    model_hub_id = model_config['model_name'] # This is the repo_id from Hugging Face Hub
    
    logging.info(f"Loading tokenizer for: {model_hub_id} using cache: {hf_cache_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_hub_id,
            trust_remote_code=True,
            cache_dir=hf_cache_dir
        )
        logging.info(f"Tokenizer for {model_hub_id} Loaded Successfully !")
    except Exception as e:
        logging.error(f"Failed to load tokenizer for {model_hub_id}: {e}", exc_info=True)
        logging.error("Exiting script - cannot continue without tokenizer.")
        return

    # H100 optimizations for Qwen models
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True # Corrected typo
    
    model = None  # Initialize to None for error checking
    try:
        logging.info(f"Attempting to load Qwen model ({model_hub_id}) with eager attention, torch_dtype=torch.bfloat16, device_map={{'': 0}}, using cache: {hf_cache_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            model_hub_id, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,       # Explicitly bfloat16 for H100
            device_map={"": 0},               # Explicitly map to GPU 0
            attn_implementation="eager",      # Reverted to eager attention
            cache_dir=hf_cache_dir
        )
        logging.info(f"Model ({model_hub_id}) pre-loaded. Current device: {model.device if model else 'None'}. Calling .eval()...")
        model = model.eval() 
        logging.info(f"Model ({model_hub_id}) loaded and .eval() completed successfully. Device: {model.device}")
        
    except Exception as e:
        logging.error(f"Failed loading model ({model_hub_id}) with eager_attention/bfloat16/device_map: {e}", exc_info=True) # Updated log message
        logging.info("Attempting fallback: progressive loading (CPU then GPU) with eager attention...")
        try:
            # Fallback still uses bfloat16 and eager attention as these are generally good for H100
            logging.info(f"Trying progressive loading (CPU then GPU) for {model_hub_id} with eager attention, torch_dtype=torch.bfloat16, using cache: {hf_cache_dir}")
            model = AutoModelForCausalLM.from_pretrained(
                model_hub_id, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16, # Fallback uses bfloat16
                device_map="cpu", # Load to CPU first
                attn_implementation="eager", # Fallback uses eager
                cache_dir=hf_cache_dir
            )
            logging.info(f"Model ({model_hub_id}) loaded to CPU. Moving model to GPU...")
            model = model.to("cuda:0").eval() # Then move to GPU
            logging.info(f"Model ({model_hub_id}) loaded with progressive approach on: {model.device}")
            
        except Exception as e2:
            logging.error(f"All Qwen model ({model_hub_id}) loading attempts failed: {e2}", exc_info=True)
            print("Exiting script - cannot continue without model")
            return  # Exit main() if model loading fails
    
    # Check if model was successfully loaded
    if model is None:
        print("ERROR: Model failed to load. Exiting.")
        return

    # --- Inference Parameters ---
    # These are specific to the inference process.
    inference_params = model_config.get('inference_config', {})
    inference_params.setdefault('max_new_tokens', 1024)
    inference_params.setdefault('do_sample', True)
    inference_params.setdefault('num_beams', 1)
    logging.info("Inference parameters configured. Inference Params: {inference_params}")

    # --- Data Processing ---
    # Pass all necessary initialized objects and configurations.
    try:
        process_financial_data(
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            output_file=OUTPUT_FILE,
            prompts_config=prompts_config,
            tokenizer=tokenizer,
            model=model,
            inference_params=inference_params
        )
    except Exception as e:
        print(f"Error {e}")

if __name__ == "__main__":
    # This block ensures main() is called only when the script is executed directly.
    main()
