#!/usr/bin/env python3
"""
Step-by-Step Mechanistic Interpretability Analysis
==================================================

This script provides a step-by-step approach to mechanistic interpretability analysis,
similar to working through Jupyter notebooks but designed for cluster execution.

Each step builds on the previous one and saves intermediate results for inspection.

Usage Examples:
    # Step 1: Explore your data first
    python scripts/step_by_step_analysis.py --step explore --predictions nlp/outputs/predictions_checkpoint_1800.json
    
    # Step 2: Extract attention for a few samples
    python scripts/step_by_step_analysis.py --step extract --max_samples 5 --save_intermediate
    
    # Step 3: Analyze a single sample in detail
    python scripts/step_by_step_analysis.py --step single --sample_id 1000_1973.0
    
    # Step 4: Compare correct vs incorrect patterns
    python scripts/step_by_step_analysis.py --step compare
    
    # Step 5: Create visualizations
    python scripts/step_by_step_analysis.py --step visualize
    
    # Step 6: Run full analysis (equivalent to the original script)
    python scripts/step_by_step_analysis.py --step full --max_samples 20
"""

import json
import os
import argparse
import sys
import numpy as np
from typing import Dict, List

# Import the main analyzer class
from mechanistic_interpretability import FinancialAttentionAnalyzer

class StepByStepAnalyzer:
    """Wrapper class for step-by-step analysis using the main FinancialAttentionAnalyzer."""
    
    def __init__(self, config_path: str, debug: bool = False):
        self.debug = debug
        self.config_path = config_path
        self.intermediate_dir = "intermediate_results"
        os.makedirs(self.intermediate_dir, exist_ok=True)
        
        # We'll initialize the main analyzer when needed to save memory
        self.analyzer = None
    
    def _get_analyzer(self):
        """Initialize analyzer only when needed."""
        if self.analyzer is None:
            print("Initializing FinancialAttentionAnalyzer...")
            self.analyzer = FinancialAttentionAnalyzer(self.config_path)
        return self.analyzer
    
    def step1_explore_data(self, predictions_file: str):
        """Step 1: Explore the prediction data structure and quality."""
        print("="*60)
        print("STEP 1: DATA EXPLORATION")
        print("="*60)
        
        # Load data
        print(f"\nLoading data from: {predictions_file}")
        with open(predictions_file, 'r') as f:
            data = json.load(f)
        
        print(f"Total samples: {len(data)}")
        
        # Basic statistics
        correct = [d for d in data if d.get('is_match', False)]
        incorrect = [d for d in data if not d.get('is_match', False)]
        
        print(f"\nBasic Statistics:")
        print(f"  Correct predictions: {len(correct)} ({len(correct)/len(data)*100:.1f}%)")
        print(f"  Incorrect predictions: {len(incorrect)} ({len(incorrect)/len(data)*100:.1f}%)")
        
        # Sample structure analysis
        if data:
            sample = data[0]
            print(f"\nSample Data Structure:")
            print(f"  Keys: {list(sample.keys())}")
            
            # Check data quality
            has_full_response = len([d for d in data if d.get('full_model_response')])
            print(f"  Samples with full_model_response: {has_full_response}/{len(data)}")
            
            if has_full_response < len(data):
                print(f"  WARNING: {len(data) - has_full_response} samples missing full responses")
        
        # Save exploration results
        exploration_results = {
            'total_samples': len(data),
            'correct_count': len(correct),
            'incorrect_count': len(incorrect),
            'usable_samples': len([d for d in data if d.get('full_model_response')]),
            'sample_ids': [f"{d['company_id']}_{d['year']}" for d in data[:10]]  # First 10 sample IDs
        }
        
        with open(f'{self.intermediate_dir}/step1_exploration.json', 'w') as f:
            json.dump(exploration_results, f, indent=2)
        
        print(f"\nExploration results saved to: {self.intermediate_dir}/step1_exploration.json")
        print(f"\nNext step: python scripts/step_by_step_analysis.py --step extract --max_samples 5")
        
        return exploration_results
    
    def step2_extract_attention(self, predictions_file: str, max_samples: int = 5, save_intermediate: bool = True):
        """Step 2: Extract attention for a small number of samples."""
        print("="*60)
        print("STEP 2: ATTENTION EXTRACTION (SMALL BATCH)")
        print("="*60)
        
        # Load data
        with open(predictions_file, 'r') as f:
            all_predictions = json.load(f)
        
        # Sample data
        correct_predictions = [p for p in all_predictions if p.get('is_match', False) and p.get('full_model_response')]
        incorrect_predictions = [p for p in all_predictions if not p.get('is_match', False) and p.get('full_model_response')]
        
        # Take small sample
        max_per_category = max_samples // 2
        sample_data = (
            correct_predictions[:max_per_category] + 
            incorrect_predictions[:max_per_category]
        )
        
        print(f"Extracting attention for {len(sample_data)} samples:")
        for sample in sample_data:
            print(f"  - {sample['company_id']}_{sample['year']} (correct: {sample['is_match']})")
        
        # Initialize analyzer and extract attention
        analyzer = self._get_analyzer()
        print(f"\nExtracting attention patterns...")
        attention_results = analyzer.extract_attention_with_predictions(sample_data)
        
        if save_intermediate:
            # Save intermediate results (without full attention matrices to save space)
            intermediate_results = {
                'sample_count': len(sample_data),
                'correct_count': len(attention_results['correct_predictions']),
                'incorrect_count': len(attention_results['incorrect_predictions']),
                'sample_summaries': []
            }
            
            # Create summaries for each sample
            for sample_id, attention_data in attention_results['attention_patterns'].items():
                summary = {
                    'sample_id': sample_id,
                    'is_correct': attention_data['is_correct'],
                    'token_count': attention_data['token_count'],
                    'layers_analyzed': list(attention_data['layers'].keys()),
                    'financial_terms_found': {}
                }
                
                # Summarize financial attention across layers
                for layer_idx, layer_data in attention_data['layers'].items():
                    for category, fin_data in layer_data['financial_attention'].items():
                        if fin_data['term_count'] > 0:
                            if category not in summary['financial_terms_found']:
                                summary['financial_terms_found'][category] = 0
                            summary['financial_terms_found'][category] += fin_data['term_count']
                
                intermediate_results['sample_summaries'].append(summary)
            
            with open(f'{self.intermediate_dir}/step2_attention_extraction.json', 'w') as f:
                json.dump(intermediate_results, f, indent=2)
            
            print(f"\nIntermediate results saved to: {self.intermediate_dir}/step2_attention_extraction.json")
        
        # Store full results for next steps
        self.attention_results = attention_results
        
        print(f"\nSUMMARY:")
        print(f"  Samples processed: {len(sample_data)}")
        print(f"  Correct predictions: {len(attention_results['correct_predictions'])}")
        print(f"  Incorrect predictions: {len(attention_results['incorrect_predictions'])}")
        
        if self.debug:
            # Print sample token analysis
            if attention_results['correct_predictions']:
                sample = attention_results['correct_predictions'][0]
                print(f"\nDEBUG - Sample token analysis ({sample['sample_id']}):")
                print(f"  Total tokens: {sample['token_count']}")
                print(f"  Layers analyzed: {list(sample['layers'].keys())}")
                
                # Show financial terms found
                for layer_idx, layer_data in sample['layers'].items():
                    financial_terms = []
                    for category, fin_data in layer_data['financial_attention'].items():
                        if fin_data['term_count'] > 0:
                            financial_terms.append(f"{category}({fin_data['term_count']})")
                    if financial_terms:
                        print(f"  Layer {layer_idx} financial terms: {', '.join(financial_terms)}")
        
        print(f"\nNext step: python scripts/step_by_step_analysis.py --step single --sample_id {sample_data[0]['company_id']}_{sample_data[0]['year']}")
        
        return attention_results
    
    def step3_analyze_single_sample(self, predictions_file: str, sample_id: str):
        """Step 3: Deep dive into a single sample's attention patterns."""
        print("="*60)
        print(f"STEP 3: SINGLE SAMPLE ANALYSIS - {sample_id}")
        print("="*60)
        
        # Load data and find the specific sample
        with open(predictions_file, 'r') as f:
            all_predictions = json.load(f)
        
        target_sample = None
        for pred in all_predictions:
            if f"{pred['company_id']}_{pred['year']}" == sample_id:
                target_sample = pred
                break
        
        if not target_sample:
            print(f"ERROR: Sample {sample_id} not found!")
            available_samples = [f"{p['company_id']}_{p['year']}" for p in all_predictions[:10]]
            print(f"Available samples (first 10): {available_samples}")
            return None
        
        print(f"Found sample: {sample_id}")
        print(f"  Actual label: {target_sample['actual_label']}")
        print(f"  Predicted: {target_sample['predicted_direction']}")
        print(f"  Correct: {target_sample['is_match']}")
        
        # Extract attention for this single sample
        analyzer = self._get_analyzer()
        attention_results = analyzer.extract_attention_with_predictions([target_sample])
        
        # Get the attention data for this sample
        sample_attention = attention_results['attention_patterns'][sample_id]
        
        print(f"\nATTENTION ANALYSIS:")
        print(f"  Total tokens: {sample_attention['token_count']}")
        print(f"  Layers analyzed: {list(sample_attention['layers'].keys())}")
        
        # Analyze token-level attention
        detailed_analysis = self._analyze_sample_tokens(sample_attention)
        
        # Save detailed analysis
        with open(f'{self.intermediate_dir}/step3_single_sample_{sample_id.replace(".", "_")}.json', 'w') as f:
            json.dump({
                'sample_info': {
                    'sample_id': sample_id,
                    'actual_label': target_sample['actual_label'],
                    'predicted_direction': target_sample['predicted_direction'],
                    'is_correct': target_sample['is_match']
                },
                'attention_analysis': detailed_analysis
            }, f, indent=2)
        
        print(f"\nDetailed analysis saved to: {self.intermediate_dir}/step3_single_sample_{sample_id.replace('.', '_')}.json")
        print(f"\nNext step: python scripts/step_by_step_analysis.py --step compare")
        
        return detailed_analysis
    
    def _analyze_sample_tokens(self, sample_attention: Dict) -> Dict:
        """Analyze token-level attention for a single sample."""
        tokens = sample_attention['tokens']
        analysis = {
            'top_attended_tokens': [],
            'financial_tokens': [],
            'attention_by_layer': {},
            'head_patterns_summary': {}
        }
        
        # Analyze each layer
        for layer_idx, layer_data in sample_attention['layers'].items():
            attention_matrix = np.array(layer_data['attention_matrix'])
            avg_attention = np.mean(attention_matrix, axis=0)  # Average across heads
            
            # Token attention scores
            token_attentions = []
            for i, token in enumerate(tokens):
                attention_received = float(np.mean(avg_attention[:, i]))
                token_attentions.append({
                    'position': i,
                    'token': token,
                    'attention': attention_received
                })
            
            # Sort by attention
            token_attentions.sort(key=lambda x: x['attention'], reverse=True)
            
            analysis['attention_by_layer'][layer_idx] = {
                'top_10_tokens': token_attentions[:10],
                'layer_stats': {
                    'mean_attention': float(np.mean(avg_attention)),
                    'max_attention': float(np.max(avg_attention)),
                    'attention_entropy': float(self._compute_entropy(avg_attention))
                }
            }
            
            # Financial terms analysis for this layer
            financial_attention = layer_data['financial_attention']
            financial_summary = {}
            for category, fin_data in financial_attention.items():
                if fin_data['term_count'] > 0:
                    financial_summary[category] = {
                        'mean_attention': fin_data['mean_attention'],
                        'term_count': fin_data['term_count'],
                        'positions': fin_data['term_positions'][:5]  # First 5 positions
                    }
            
            analysis['financial_tokens'] = financial_summary
            
            # Head patterns summary
            head_patterns = layer_data['head_patterns']
            pattern_summary = {}
            for pattern_type in ['self_attention', 'causal_attention', 'financial_focus', 'distributed']:
                scores = [head_patterns[head][pattern_type] for head in head_patterns.keys() 
                         if pattern_type in head_patterns[head]]
                if scores:
                    pattern_summary[pattern_type] = {
                        'mean': float(np.mean(scores)),
                        'max': float(np.max(scores)),
                        'std': float(np.std(scores))
                    }
            
            analysis['head_patterns_summary'][layer_idx] = pattern_summary
        
        # Overall top tokens across all layers
        all_token_scores = {}
        for layer_idx, layer_analysis in analysis['attention_by_layer'].items():
            for token_data in layer_analysis['top_10_tokens']:
                pos = token_data['position']
                if pos not in all_token_scores:
                    all_token_scores[pos] = {
                        'token': token_data['token'],
                        'position': pos,
                        'attention_scores': [],
                        'avg_attention': 0
                    }
                all_token_scores[pos]['attention_scores'].append(token_data['attention'])
        
        # Calculate average attention across layers
        for pos, data in all_token_scores.items():
            data['avg_attention'] = np.mean(data['attention_scores'])
        
        # Sort by average attention
        sorted_tokens = sorted(all_token_scores.values(), 
                              key=lambda x: x['avg_attention'], reverse=True)
        
        analysis['top_attended_tokens'] = sorted_tokens[:20]
        
        # Print summary
        print(f"\n  TOP 10 MOST ATTENDED TOKENS:")
        for i, token_data in enumerate(sorted_tokens[:10]):
            print(f"    {i+1:2d}. Position {token_data['position']:4d}: '{token_data['token'][:20]:20s}' "
                  f"(avg attention: {token_data['avg_attention']:.4f})")
        
        if analysis['financial_tokens']:
            print(f"\n  FINANCIAL TERMS FOUND:")
            for category, data in analysis['financial_tokens'].items():
                if data['term_count'] > 0:
                    print(f"    {category}: {data['term_count']} terms, "
                          f"avg attention: {data['mean_attention']:.4f}")
        
        return analysis
    
    def _compute_entropy(self, attention_vector: np.ndarray) -> float:
        """Compute entropy of attention distribution."""
        # Add small epsilon to avoid log(0)
        normalized = attention_vector + 1e-10
        normalized = normalized / normalized.sum()
        entropy = -np.sum(normalized * np.log(normalized))
        return entropy
    
    def step4_compare_patterns(self):
        """Step 4: Compare attention patterns between correct and incorrect predictions."""
        print("="*60)
        print("STEP 4: COMPARE CORRECT VS INCORRECT PATTERNS")
        print("="*60)
        
        if not hasattr(self, 'attention_results'):
            print("ERROR: No attention results found. Run step2 first.")
            print("Usage: python scripts/step_by_step_analysis.py --step extract --max_samples 10")
            return None
        
        analyzer = self._get_analyzer()
        comparison_results = analyzer.compare_correct_vs_incorrect(self.attention_results)
        
        # Print key findings
        print(f"\nCOMPARISON RESULTS:")
        
        if 'financial_attention_differences' in comparison_results:
            print(f"\nFinancial Attention Differences:")
            fin_diff = comparison_results['financial_attention_differences']
            for category, data in sorted(fin_diff.items(), 
                                       key=lambda x: abs(x[1]['difference']), 
                                       reverse=True)[:5]:
                diff = data['difference']
                direction = "higher" if diff > 0 else "lower"
                print(f"  {category}: Correct predictions show {direction} attention "
                      f"(difference: {diff:.4f})")
        
        if 'head_specialization_differences' in comparison_results:
            print(f"\nHead Pattern Differences:")
            head_diff = comparison_results['head_specialization_differences']
            for pattern, data in sorted(head_diff.items(), 
                                      key=lambda x: abs(x[1]['effect_size']), 
                                      reverse=True):
                effect_size = data['effect_size']
                direction = "higher" if data['difference'] > 0 else "lower"
                effect_desc = "large" if abs(effect_size) > 0.8 else \
                             "medium" if abs(effect_size) > 0.5 else \
                             "small" if abs(effect_size) > 0.2 else "negligible"
                
                print(f"  {pattern}: {direction} in correct predictions "
                      f"(effect size: {effect_size:.3f} - {effect_desc})")
        
        # Save comparison results
        with open(f'{self.intermediate_dir}/step4_comparison_results.json', 'w') as f:
            json.dump(analyzer._make_serializable(comparison_results), f, indent=2)
        
        self.comparison_results = comparison_results
        
        print(f"\nComparison results saved to: {self.intermediate_dir}/step4_comparison_results.json")
        print(f"\nNext step: python scripts/step_by_step_analysis.py --step visualize")
        
        return comparison_results
    
    def step5_create_visualizations(self):
        """Step 5: Create visualizations of attention patterns."""
        print("="*60)
        print("STEP 5: CREATE VISUALIZATIONS")
        print("="*60)
        
        if not hasattr(self, 'attention_results') or not hasattr(self, 'comparison_results'):
            print("ERROR: Missing attention or comparison results. Run steps 2 and 4 first.")
            return None
        
        analyzer = self._get_analyzer()
        
        # Create visualizations
        print("Creating visualizations...")
        analyzer.create_visualizations(self.attention_results, self.comparison_results)
        
        print(f"\nVisualizations created in: {analyzer.output_dir}/visualizations/")
        print(f"Generated files:")
        viz_dir = f"{analyzer.output_dir}/visualizations"
        if os.path.exists(viz_dir):
            for file in os.listdir(viz_dir):
                if file.endswith('.png'):
                    print(f"  - {file}")
        
        print(f"\nNext step: python scripts/step_by_step_analysis.py --step full  # For complete analysis")
        
        return True
    
    def step6_full_analysis(self, predictions_file: str, max_samples: int = 20):
        """Step 6: Run the complete analysis (equivalent to original script)."""
        print("="*60)
        print("STEP 6: FULL ANALYSIS")
        print("="*60)
        
        analyzer = self._get_analyzer()
        attention_results, comparison_results = analyzer.run_full_analysis(
            predictions_file, max_samples
        )
        
        return attention_results, comparison_results

def main():
    parser = argparse.ArgumentParser(description='Step-by-step Mechanistic Interpretability Analysis')
    
    # Step selection
    parser.add_argument('--step', choices=[
        'explore', 'extract', 'single', 'compare', 'visualize', 'full'
    ], required=True, help='Analysis step to run')
    
    # Common arguments
    parser.add_argument('--config', default='nlp/config/interpretability_config.yaml',
                       help='Path to interpretability config file')
    parser.add_argument('--predictions', default='nlp/outputs/predictions_checkpoint_1800.json',
                       help='Path to predictions JSON file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    # Step-specific arguments
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples for extract/full steps')
    parser.add_argument('--sample_id', help='Specific sample ID for single step (e.g., 1000_1973.0)')
    parser.add_argument('--save_intermediate', action='store_true', default=True,
                       help='Save intermediate results')
    
    args = parser.parse_args()
    
    # Initialize step-by-step analyzer
    step_analyzer = StepByStepAnalyzer(args.config, args.debug)
    
    try:
        if args.step == 'explore':
            step_analyzer.step1_explore_data(args.predictions)
        
        elif args.step == 'extract':
            step_analyzer.step2_extract_attention(args.predictions, args.max_samples, args.save_intermediate)
        
        elif args.step == 'single':
            if not args.sample_id:
                print("ERROR: --sample_id required for single step")
                print("Usage: python scripts/step_by_step_analysis.py --step single --sample_id 1000_1973.0")
                return
            step_analyzer.step3_analyze_single_sample(args.predictions, args.sample_id)
        
        elif args.step == 'compare':
            step_analyzer.step4_compare_patterns()
        
        elif args.step == 'visualize':
            step_analyzer.step5_create_visualizations()
        
        elif args.step == 'full':
            step_analyzer.step6_full_analysis(args.predictions, args.max_samples)
        
        print(f"\nStep '{args.step}' completed successfully!")
        
    except Exception as e:
        print(f"ERROR in step '{args.step}': {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()