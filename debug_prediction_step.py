#!/usr/bin/env python3
"""
Debug script to test prediction-step attention extraction
"""

import json
import sys
import os

# Add the scripts directory to the path
sys.path.append('scripts')

from mechanistic_interpretability import FinancialAttentionAnalyzer

def debug_prediction_attention():
    """Debug the prediction attention extraction with detailed output."""
    
    print("🔍 Debugging prediction-step attention extraction...")
    
    # Initialize analyzer
    config_path = "nlp/config/interpretability_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return False
    
    # Load predictions
    predictions_file = "nlp/outputs/predictions_checkpoint_1800.json"
    
    if not os.path.exists(predictions_file):
        print(f"Error: Predictions file not found: {predictions_file}")
        return False
    
    print("Loading analyzer...")
    analyzer = FinancialAttentionAnalyzer(config_path)
    
    # Load a single sample
    with open(predictions_file, 'r') as f:
        all_data = json.load(f)
    
    # Get one correct and one incorrect sample
    correct_samples = [d for d in all_data if d.get('is_match', False) and d.get('full_model_response')]
    incorrect_samples = [d for d in all_data if not d.get('is_match', False) and d.get('full_model_response')]
    
    if not correct_samples:
        print("No correct samples found!")
        return False
        
    sample = correct_samples[0]
    print(f"\n📊 Testing with sample: {sample['company_id']}_{sample['year']}")
    print(f"   Actual label: {sample['actual_label']}")
    print(f"   Predicted: {sample['predicted_direction']}")
    print(f"   Is correct: {sample['is_match']}")
    
    try:
        # Test the new attention extraction
        print("\n🔬 Extracting attention at prediction step...")
        attention_results = analyzer.extract_attention_with_predictions([sample])
        
        if attention_results['correct_predictions']:
            result = attention_results['correct_predictions'][0]
            
            print(f"\n✅ Attention extraction successful!")
            print(f"   Sample ID: {result['sample_id']}")
            print(f"   Total tokens: {result['token_count']}")
            print(f"   Input length: {result['input_length']}")
            print(f"   Generated tokens: {len(result['generated_tokens'])}")
            
            # Show prediction step info
            if 'prediction_step_info' in result:
                step_info = result['prediction_step_info']
                print(f"\n🎯 Prediction Step Info:")
                print(f"   Step: {step_info['step']}")
                print(f"   Token: '{step_info['token']}'")
                print(f"   Total steps: {step_info['total_steps']}")
            
            # Show first few generated tokens
            print(f"\n📝 Generated tokens:")
            for i, token in enumerate(result['generated_tokens'][:10]):
                marker = " <-- PREDICTION" if (i == result['prediction_step_info']['step']) else ""
                print(f"   {i}: '{token}'{marker}")
            
            if len(result['generated_tokens']) > 10:
                print(f"   ... and {len(result['generated_tokens']) - 10} more tokens")
            
            # Show layers analyzed
            print(f"\n🧠 Layers analyzed: {list(result['layers'].keys())}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error during attention extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_prediction_attention()
    if success:
        print("\n🎉 Debug successful! Prediction-step attention extraction is working.")
    else:
        print("\n❌ Debug failed. Please check the error messages above.") 