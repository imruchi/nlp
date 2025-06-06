import json
import pandas as pd
from nlp.nlp.inference import parse_description, build_prompt, run_inference, parse_response
from itertools import islice
from sklearn.metrics import f1_score,precision_score, recall_score

if __name__ == "__main__":
    with open("data/process_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        data = dict(list(islice(data.items(), 100)))
        # print(data)

    results = []


    # print(data.items()[0])
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
            print("**********************************",response)
            parsed_result = parse_response(response)


            #build the csv
            rows = {
                "firm_id" : firm_id,
                "year" : year,
                "true_label" : label,
                "balance_sheet_text": balance_sheet_text,
                "income_statement_text": income_statement_text,
                "response": response,
                "direction": parsed_result['direction'],
                "magnitude": parsed_result['magnitude'],
                "confidence": parsed_result['confidence']
            }

            results.append(rows)

            print("Prediction:", parsed_result["direction"])
            print("Label:", label)
            print("-" * 50)

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("inference_results.csv", index=False)
    print("Inference completed. Results saved to 'inference_results.csv'.")
    y_true = df_results['true_label']
    y_pred = df_results['direction']
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    print(f"F1 SCORE: {f1:.4f}")
    print(f"Recall SCORE: {precision:.4f}")
    print(f"Precision SCORE: {recall:.4f}")
    





    #f1 score calculation

    