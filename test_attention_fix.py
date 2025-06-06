#!/usr/bin/env python3
"""
Test script to verify the attention dimension fix
"""

import json
import sys
import os

# Add the scripts directory to the path
sys.path.append('scripts')

from step_by_step_analysis import StepByStepAnalyzer

def test_attention_extraction():
    """Test the attention extraction with a small sample."""
    
    print("Testing attention extraction fix...")
    
    # Initialize analyzer
    config_path = "nlp/config/interpretability_config.yaml"
    analyzer = StepByStepAnalyzer(config_path, debug=True)
    
    # Load predictions
    predictions_file = "nlp/outputs/predictions_checkpoint_1800.json"
    
    if not os.path.exists(predictions_file):
        print(f"Error: Predictions file not found: {predictions_file}")
        print("Please run the model first to generate predictions.")
        return False
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Please ensure the config file exists.")
        return False
    
    try:
        # Run step 2 with a very small sample
        print("Running Step 2: Extract attention (1 sample only)...")
        analyzer.step2_extract_attention(predictions_file, max_samples=2, save_intermediate=True)
        print("✅ Success! Attention extraction completed without dimension errors.")
        return True
        
    except Exception as e:
        print(f"❌ Error during attention extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_attention_extraction()
    if success:
        print("\n🎉 Test passed! The dimension error has been fixed.")
        print("You can now run: python scripts/step_by_step_analysis.py --step extract --max_samples 5")
    else:
        print("\n❌ Test failed. Please check the error messages above.") 