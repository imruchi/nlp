import json
import pandas as pd
from models.inference import parse_description, build_prompt, run_inference, parse_response

if __name__ == "__main__":
    with open("process_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        data = data[:100]

    results = []

    with open("debug_output.txt", "w", encoding="utf-8") as log_file:
        for firm_id, years in data.items():
            for year, entry in years.items():
                if not isinstance(entry, dict):
                    continue  # skip non-dictionary entries like std_dev_sales
                description = entry.get("description", "")
                label = entry.get("label", "")

                balance_sheet_text, income_statement_text = parse_description(description)
                prompt = build_prompt(balance_sheet_text, income_statement_text)

                print(f"\nRunning inference for Firm {firm_id}, Year {year}...")
                response = run_inference(prompt)
                parsed_result = parse_response(response)

                # Log full response
                log_file.write(f"Firm: {firm_id}, Year: {year}\n")
                log_file.write(prompt + "\n")
                log_file.write(response + "\n")
                log_file.write("-" * 80 + "\n")

                # Add metadata
                parsed_result.update({
                    "firm_id": firm_id,
                    "year": year,
                    "label": label
                })

                results.append(parsed_result)

                print("Prediction:", parsed_result["direction"])
                print("Label:", label)
                print("-" * 50)

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("inference_results.csv", index=False)
    print("Inference completed. Results saved to 'inference_results.csv'.")