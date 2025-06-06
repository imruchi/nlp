#!/usr/bin/env python3
"""
Mechanistic Interpretability Analysis for Qwen2-7B Financial Predictions
=========================================================================

This script performs detailed attention analysis on the Qwen2-7B-Instruct model
to understand how attention patterns differ between correct and incorrect 
financial EPS predictions.

Key Analysis:
- Attention pattern extraction across all layers and heads
- Financial vocabulary attention mapping  
- Correct vs incorrect prediction comparison
- Head specialization detection
- Attention flow visualization

Usage:
    python scripts/mechanistic_interpretability.py --config nlp/config/interpretability_config.yaml
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

class FinancialAttentionAnalyzer:
    """Main class for mechanistic interpretability analysis of financial predictions."""
    
    def __init__(self, config_path: str):
        """Initialize analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.model_name = self.config['model_config']['name']
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        ).eval()
        
        # Financial vocabulary from config
        self.financial_vocab = self._build_financial_vocabulary()
        
        # Results storage
        self.attention_data = {}
        self.analysis_results = {}
        
        # Create output directory
        self.output_dir = self.config['output_config']['output_directory']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/attention_matrices", exist_ok=True)
        os.makedirs(f"{self.output_dir}/visualizations", exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_financial_vocabulary(self) -> Dict[str, List[str]]:
        """Build comprehensive financial vocabulary from config."""
        vocab = {}
        fin_vocab = self.config['financial_vocabulary']
        
        for category, terms in fin_vocab.items():
            vocab[category] = []
            for term in terms:
                # Add original term and variations
                vocab[category].extend([
                    term, term.capitalize(), term.upper(),
                    f" {term}", f"{term} ", f" {term} "
                ])
        
        # Add prediction targets
        vocab['prediction_targets'] = [
            'increase', 'decrease', 'Increase', 'Decrease',
            'INCREASE', 'DECREASE', 'up', 'down', 'rise', 'fall'
        ]
        
        return vocab
    
    def extract_attention_with_predictions(self, sample_data: List[Dict]) -> Dict:
        """
        Extract attention patterns for both correct and incorrect predictions.
        
        Args:
            sample_data: List of prediction samples with prompts and labels
            
        Returns:
            Dictionary containing attention matrices and metadata
        """
        results = {
            'correct_predictions': [],
            'incorrect_predictions': [],
            'attention_patterns': {},
            'token_mappings': {}
        }
        
        for i, sample in enumerate(sample_data):
            print(f"Processing sample {i+1}/{len(sample_data)}: {sample['company_id']}-{sample['year']}")
            
            # Build prompt from sample data
            prompt = self._build_prompt_from_sample(sample)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                  max_length=2048).to(self.device)
            
            print("=" * 80)
            print(f"FULL INPUT PROMPT:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)
            
            # Generate with attention capture
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=self.config['generation_config']['max_new_tokens'],
                    temperature=self.config['generation_config']['temperature'],
                    do_sample=self.config['generation_config']['do_sample'],
                    output_attentions=True,
                    return_dict_in_generate=True
                )
            
            # Decode the full generated sequence
            full_generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Extract only the newly generated part (response)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs.sequences[0][input_length:]
            generated_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"FULL GENERATED TEXT:")
            print("=" * 80)
            print(full_generated_text)
            print("=" * 80)
            print(f"GENERATED RESPONSE ONLY: '{generated_response}'")
            print(f"EXPECTED LABEL: {sample.get('actual_label', 'Unknown')}")
            print(f"PREDICTED LABEL: {sample.get('predicted_direction', 'Unknown')}")
            print(f"IS CORRECT: {sample.get('is_match', 'Unknown')}")
            print("=" * 80)
            
            # Extract attention matrices specifically when prediction is made
            attention_data = self._extract_attention_at_prediction(
                outputs.attentions, outputs.sequences, inputs["input_ids"], sample, 
                full_generated_text, generated_response, prompt
            )
            
            # Categorize by correctness
            if sample['is_match']:
                results['correct_predictions'].append(attention_data)
            else:
                results['incorrect_predictions'].append(attention_data)
                
            # Store individual patterns
            sample_id = f"{sample['company_id']}_{sample['year']}"
            results['attention_patterns'][sample_id] = attention_data
            
        return results
    
    def _build_prompt_from_sample(self, sample: Dict) -> str:
        """Build prompt from sample data."""
        # For this analysis, we'll reconstruct the prompt from the full_model_response
        # Extract the user prompt portion
        full_response = sample.get('full_model_response', '')
        
        # Find the user portion between "user\n" and "\nassistant"
        user_start = full_response.find("user\n")
        assistant_start = full_response.find("\nassistant")
        
        if user_start != -1 and assistant_start != -1:
            prompt = full_response[user_start + 5:assistant_start].strip()
            return prompt
        
        # Fallback: use a basic template
        return f"Based on financial data, assess whether EPS will increase or decrease in the next year. Return your prediction as Increase or Decrease."
    
    def _extract_attention_at_prediction(self, attentions: Tuple, sequences: torch.Tensor, 
                                       input_ids: torch.Tensor, sample: Dict,
                                       full_generated_text: str = "", generated_response: str = "",
                                       original_prompt: str = "") -> Dict:
        """Extract attention matrices specifically when prediction tokens are generated."""
        
        # Convert input_ids to tokens (original prompt)
        original_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Convert full sequence to tokens (prompt + generated)
        full_sequence = sequences[0]  # Remove batch dimension
        all_tokens = self.tokenizer.convert_ids_to_tokens(full_sequence)
        
        # Find where generation starts (after the input)
        input_length = input_ids.shape[1]
        generated_tokens = all_tokens[input_length:]
        
        attention_data = {
            'sample_id': f"{sample['company_id']}_{sample['year']}",
            'is_correct': sample['is_match'],
            'actual_label': sample['actual_label'],
            'predicted_label': sample['predicted_direction'],
            'original_prompt': original_prompt,
            'full_generated_text': full_generated_text,
            'generated_response': generated_response,
            'tokens': all_tokens,
            'original_tokens': original_tokens,
            'generated_tokens': generated_tokens,
            'token_count': len(all_tokens),
            'input_length': input_length,
            'layers': {},
            'prediction_step_info': {}
        }
        
        # Find the generation step where prediction tokens appear
        prediction_step = self._find_prediction_step(generated_tokens, attentions)
        
        if prediction_step is not None:
            print(f"Found prediction at generation step {prediction_step}")
            attention_data['prediction_step_info'] = {
                'step': prediction_step,
                'token': generated_tokens[prediction_step] if prediction_step < len(generated_tokens) else 'unknown',
                'total_steps': len(attentions)
            }
            
            # Use attention from the prediction step
            step_attentions = attentions[prediction_step]
            
            # Process each layer's attention
            layers_to_analyze = self.config['attention_analysis']['layers_to_analyze']
            
            for layer_idx in range(len(step_attentions)):
                if layer_idx in layers_to_analyze:
                    layer_attention = step_attentions[layer_idx]
                    
                    # Remove batch dimension if present
                    if layer_attention.dim() == 4:  # [batch, heads, seq, seq]
                        layer_attention = layer_attention[0]  # Remove batch dimension
                    
                    # Ensure we have the right dimensions: [heads, seq, seq]
                    if layer_attention.dim() != 3:
                        print(f"Warning: Unexpected attention tensor shape for layer {layer_idx}: {layer_attention.shape}")
                        continue
                    
                    attention_data['layers'][layer_idx] = {
                        'attention_matrix': layer_attention.cpu().numpy(),
                        'head_patterns': self._analyze_head_patterns(layer_attention, all_tokens),
                        'financial_attention': self._compute_financial_attention(layer_attention, all_tokens),
                        'prediction_attention': self._compute_prediction_attention(layer_attention, all_tokens)
                    }
        else:
            # Fallback: use last step if no prediction token found
            print("Warning: No prediction token found, using last generation step")
            attention_data['prediction_step_info'] = {
                'step': len(attentions) - 1,
                'token': 'last_step_fallback',
                'total_steps': len(attentions)
            }
            
            if len(attentions) > 0:
                last_step_attentions = attentions[-1]
                layers_to_analyze = self.config['attention_analysis']['layers_to_analyze']
                
                for layer_idx in range(len(last_step_attentions)):
                    if layer_idx in layers_to_analyze:
                        layer_attention = last_step_attentions[layer_idx]
                        
                        if layer_attention.dim() == 4:
                            layer_attention = layer_attention[0]
                        
                        if layer_attention.dim() != 3:
                            print(f"Warning: Unexpected attention tensor shape for layer {layer_idx}: {layer_attention.shape}")
                            continue
                        
                        attention_data['layers'][layer_idx] = {
                            'attention_matrix': layer_attention.cpu().numpy(),
                            'head_patterns': self._analyze_head_patterns(layer_attention, all_tokens),
                            'financial_attention': self._compute_financial_attention(layer_attention, all_tokens),
                            'prediction_attention': self._compute_prediction_attention(layer_attention, all_tokens)
                        }
            else:
                print("Warning: No attention data found in model output")
        
        return attention_data
    
    def _find_prediction_step(self, generated_tokens: List[str], attentions: Tuple) -> int:
        """Find the generation step where prediction tokens (increase/decrease) appear."""
        
        # Prediction keywords to look for
        prediction_keywords = ['increase', 'decrease', 'Increase', 'Decrease', 
                             'inc', 'dec', 'up', 'down', 'rise', 'fall']
        
        # Look through generated tokens to find prediction keywords
        for step, token in enumerate(generated_tokens):
            token_lower = token.lower().strip()
            
            # Check if this token contains a prediction keyword
            for keyword in prediction_keywords:
                if keyword.lower() in token_lower:
                    # Make sure we have attention data for this step
                    if step < len(attentions):
                        print(f"Found prediction token '{token}' at step {step}")
                        return step
        
        # Alternative: look for tokens that might be part of prediction words
        for step, token in enumerate(generated_tokens):
            token_clean = token.lower().strip().replace('▁', '').replace('Ġ', '')
            
            # Check partial matches for tokenized words
            if any(keyword in token_clean for keyword in ['increa', 'decrea', 'rise', 'fall']):
                if step < len(attentions):
                    print(f"Found potential prediction token '{token}' at step {step}")
                    return step
        
        return None
    
    def _analyze_head_patterns(self, layer_attention: torch.Tensor, tokens: List[str]) -> Dict:
        """Analyze attention patterns for each head in a layer."""
        num_heads = layer_attention.shape[0]
        head_patterns = {}
        
        for head_idx in range(num_heads):
            head_attn = layer_attention[head_idx].cpu().numpy()
            
            # Detect pattern types
            patterns = {
                'self_attention': np.mean(np.diag(head_attn)),
                'causal_attention': self._compute_causal_attention(head_attn),
                'beginning_focus': np.mean(head_attn[:, :5]) if head_attn.shape[1] > 5 else 0,
                'distributed': self._compute_attention_entropy(head_attn),
                'financial_focus': self._compute_financial_focus(head_attn, tokens)
            }
            
            head_patterns[head_idx] = patterns
            
        return head_patterns
    
    def _compute_causal_attention(self, attention_matrix: np.ndarray) -> float:
        """Compute attention to previous tokens (causal pattern)."""
        if attention_matrix.ndim != 2:
            print(f"Warning: Expected 2D attention matrix, got {attention_matrix.ndim}D with shape {attention_matrix.shape}")
            return 0.0
            
        seq_len = attention_matrix.shape[0]
        if seq_len <= 1:
            return 0.0
            
        causal_scores = []
        
        for i in range(1, seq_len):
            # Attention to all previous tokens
            prev_attention = np.sum(attention_matrix[i, :i])
            causal_scores.append(prev_attention)
            
        return np.mean(causal_scores) if causal_scores else 0.0
    
    def _compute_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Compute entropy of attention distribution (higher = more distributed)."""
        if attention_matrix.ndim != 2:
            print(f"Warning: Expected 2D attention matrix, got {attention_matrix.ndim}D with shape {attention_matrix.shape}")
            return 0.0
            
        entropies = []
        for row in attention_matrix:
            # Add small epsilon to avoid log(0)
            row_normalized = row + 1e-10
            if row_normalized.sum() == 0:
                entropies.append(0.0)
                continue
            row_normalized = row_normalized / row_normalized.sum()
            entropy = -np.sum(row_normalized * np.log(row_normalized))
            entropies.append(entropy)
        return np.mean(entropies) if entropies else 0.0
    
    def _compute_financial_attention(self, layer_attention: torch.Tensor, tokens: List[str]) -> Dict:
        """Compute attention scores to financial vocabulary terms."""
        financial_scores = {}
        
        for category, vocab_terms in self.financial_vocab.items():
            category_scores = []
            
            # Find positions of financial terms
            financial_positions = []
            for i, token in enumerate(tokens):
                if any(term in token.lower() for term in vocab_terms):
                    financial_positions.append(i)
            
            if financial_positions:
                # Average attention to financial terms across all heads
                for head in range(layer_attention.shape[0]):
                    head_attn = layer_attention[head].cpu().numpy()
                    
                    # Check bounds and filter valid positions
                    valid_positions = [pos for pos in financial_positions if pos < head_attn.shape[1]]
                    if valid_positions:
                        financial_attention = np.mean(head_attn[:, valid_positions])
                        category_scores.append(financial_attention)
                
                financial_scores[category] = {
                    'mean_attention': np.mean(category_scores),
                    'std_attention': np.std(category_scores),
                    'term_positions': financial_positions,
                    'term_count': len(financial_positions)
                }
            else:
                financial_scores[category] = {
                    'mean_attention': 0.0,
                    'std_attention': 0.0,
                    'term_positions': [],
                    'term_count': 0
                }
        
        return financial_scores
    
    def _compute_financial_focus(self, attention_matrix: np.ndarray, tokens: List[str]) -> float:
        """Compute how much this head focuses on financial terms."""
        if attention_matrix.ndim != 2:
            print(f"Warning: Expected 2D attention matrix, got {attention_matrix.ndim}D with shape {attention_matrix.shape}")
            return 0.0
            
        financial_positions = []
        
        for i, token in enumerate(tokens):
            if i >= attention_matrix.shape[1]:  # Check bounds
                break
            for category, vocab_terms in self.financial_vocab.items():
                if any(term in token.lower() for term in vocab_terms):
                    financial_positions.append(i)
                    break
        
        if not financial_positions:
            return 0.0
            
        # Average attention to financial terms
        try:
            financial_attention = np.mean(attention_matrix[:, financial_positions])
            return float(financial_attention)
        except IndexError as e:
            print(f"Warning: Index error in financial attention computation: {e}")
            return 0.0
    
    def _compute_prediction_attention(self, layer_attention: torch.Tensor, tokens: List[str]) -> Dict:
        """Compute attention patterns around prediction tokens."""
        prediction_patterns = {}
        target_tokens = self.config['analysis_focus']['prediction_analysis']['target_tokens']
        
        for target in target_tokens:
            target_positions = [i for i, token in enumerate(tokens) 
                              if target.lower() in token.lower()]
            
            if target_positions:
                attention_scores = []
                for head in range(layer_attention.shape[0]):
                    head_attn = layer_attention[head].cpu().numpy()
                    
                    for pos in target_positions:
                        # Check bounds
                        if pos < head_attn.shape[0] and pos < head_attn.shape[1]:
                            # Attention TO this prediction token
                            incoming_attention = np.mean(head_attn[:, pos])
                            # Attention FROM this prediction token  
                            outgoing_attention = np.mean(head_attn[pos, :])
                            
                            attention_scores.append({
                                'incoming': float(incoming_attention),
                                'outgoing': float(outgoing_attention),
                                'position': pos
                            })
                
                prediction_patterns[target] = attention_scores
        
        return prediction_patterns
    
    def _find_financial_data_region(self, tokens: List[str]) -> Tuple[Optional[int], Optional[int]]:
        """Find the token range containing financial statement data."""
        
        # Look for start markers
        start_markers = ["Balance", "Sheet", "Income", "Statement", "Account", "Items"]
        end_markers = ["Solve", "this", "problem", "step", "by", "step"]
        
        start_pos = None
        end_pos = None
        
        # Find start of financial data
        for i, token in enumerate(tokens):
            if any(marker.lower() in token.lower() for marker in start_markers):
                start_pos = max(0, i - 5)  # Start a bit before the marker
                break
        
        # Find end of financial data (where analysis instructions begin)
        for i, token in enumerate(tokens):
            if any(marker.lower() in token.lower() for marker in end_markers):
                end_pos = min(len(tokens), i + 5)  # End a bit after the marker
                break
        
        # If we found both markers, return the range
        if start_pos is not None and end_pos is not None and end_pos > start_pos:
            # Limit to reasonable size for visualization
            max_region_size = 200
            if end_pos - start_pos > max_region_size:
                end_pos = start_pos + max_region_size
            return start_pos, end_pos
        
        # Fallback: look for numerical data patterns
        numeric_positions = []
        for i, token in enumerate(tokens):
            # Look for tokens that contain numbers or financial indicators
            if any(char.isdigit() for char in token) or \
               any(financial_word in token.lower() for financial_word in 
                   ['asset', 'liability', 'sales', 'revenue', 'income', 'cash', 'debt']):
                numeric_positions.append(i)
        
        if len(numeric_positions) > 10:  # If we found enough financial data
            start_pos = max(0, min(numeric_positions) - 10)
            end_pos = min(len(tokens), max(numeric_positions) + 10)
            
            # Limit to reasonable size
            max_region_size = 200
            if end_pos - start_pos > max_region_size:
                end_pos = start_pos + max_region_size
            
            return start_pos, end_pos
        
        return None, None
    
    def compare_correct_vs_incorrect(self, attention_results: Dict) -> Dict:
        """Compare attention patterns between correct and incorrect predictions."""
        print("Analyzing differences between correct and incorrect predictions...")
        
        comparison_results = {
            'statistical_differences': {},
            'pattern_differences': {},
            'financial_attention_differences': {},
            'head_specialization_differences': {}
        }
        
        correct_data = attention_results['correct_predictions']
        incorrect_data = attention_results['incorrect_predictions']
        
        if len(correct_data) == 0 or len(incorrect_data) == 0:
            print("Warning: No data for comparison - need both correct and incorrect predictions")
            return comparison_results
        
        # Compare financial attention patterns
        comparison_results['financial_attention_differences'] = self._compare_financial_attention(
            correct_data, incorrect_data
        )
        
        # Compare head specialization patterns
        comparison_results['head_specialization_differences'] = self._compare_head_specialization(
            correct_data, incorrect_data
        )
        
        # Statistical significance tests
        comparison_results['statistical_differences'] = self._statistical_comparison(
            correct_data, incorrect_data
        )
        
        return comparison_results
    
    def _compare_financial_attention(self, correct_data: List, incorrect_data: List) -> Dict:
        """Compare financial vocabulary attention between correct and incorrect predictions."""
        results = {}
        
        for category in self.financial_vocab.keys():
            correct_scores = []
            incorrect_scores = []
            
            # Collect scores for correct predictions
            for sample in correct_data:
                for layer_idx, layer_data in sample['layers'].items():
                    if category in layer_data['financial_attention']:
                        correct_scores.append(layer_data['financial_attention'][category]['mean_attention'])
            
            # Collect scores for incorrect predictions  
            for sample in incorrect_data:
                for layer_idx, layer_data in sample['layers'].items():
                    if category in layer_data['financial_attention']:
                        incorrect_scores.append(layer_data['financial_attention'][category]['mean_attention'])
            
            if correct_scores and incorrect_scores:
                results[category] = {
                    'correct_mean': np.mean(correct_scores),
                    'incorrect_mean': np.mean(incorrect_scores),
                    'difference': np.mean(correct_scores) - np.mean(incorrect_scores),
                    'correct_std': np.std(correct_scores),
                    'incorrect_std': np.std(incorrect_scores)
                }
        
        return results
    
    def _compare_head_specialization(self, correct_data: List, incorrect_data: List) -> Dict:
        """Compare head specialization patterns."""
        pattern_types = ['self_attention', 'causal_attention', 'beginning_focus', 
                        'distributed', 'financial_focus']
        
        results = {}
        
        for pattern_type in pattern_types:
            correct_scores = []
            incorrect_scores = []
            
            # Collect pattern scores
            for sample in correct_data:
                for layer_idx, layer_data in sample['layers'].items():
                    for head_idx, head_patterns in layer_data['head_patterns'].items():
                        if pattern_type in head_patterns:
                            correct_scores.append(head_patterns[pattern_type])
            
            for sample in incorrect_data:
                for layer_idx, layer_data in sample['layers'].items():
                    for head_idx, head_patterns in layer_data['head_patterns'].items():
                        if pattern_type in head_patterns:
                            incorrect_scores.append(head_patterns[pattern_type])
            
            if correct_scores and incorrect_scores:
                results[pattern_type] = {
                    'correct_mean': np.mean(correct_scores),
                    'incorrect_mean': np.mean(incorrect_scores),
                    'difference': np.mean(correct_scores) - np.mean(incorrect_scores),
                    'effect_size': (np.mean(correct_scores) - np.mean(incorrect_scores)) / 
                                 np.sqrt((np.var(correct_scores) + np.var(incorrect_scores)) / 2)
                }
        
        return results
    
    def _statistical_comparison(self, correct_data: List, incorrect_data: List) -> Dict:
        """Perform statistical tests on attention differences."""
        from scipy import stats
        
        results = {}
        
        # Compare overall attention entropy
        correct_entropies = []
        incorrect_entropies = []
        
        for sample in correct_data:
            for layer_idx, layer_data in sample['layers'].items():
                for head_idx, head_patterns in layer_data['head_patterns'].items():
                    if 'distributed' in head_patterns:
                        correct_entropies.append(head_patterns['distributed'])
        
        for sample in incorrect_data:
            for layer_idx, layer_data in sample['layers'].items():
                for head_idx, head_patterns in layer_data['head_patterns'].items():
                    if 'distributed' in head_patterns:
                        incorrect_entropies.append(head_patterns['distributed'])
        
        if len(correct_entropies) > 3 and len(incorrect_entropies) > 3:
            t_stat, p_value = stats.ttest_ind(correct_entropies, incorrect_entropies)
            results['attention_entropy'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.config['comparative_analysis']['statistical_tests']['significance_level']
            }
        
        return results
    
    def create_visualizations(self, attention_results: Dict, comparison_results: Dict):
        """Create comprehensive visualizations of attention patterns."""
        print("Creating visualizations...")
        
        # 1. Attention heatmaps for sample cases
        self._create_attention_heatmaps(attention_results)
        
        # 2. Financial attention comparison
        self._create_financial_attention_comparison(comparison_results)
        
        # 3. Head specialization analysis
        self._create_head_specialization_plots(comparison_results)
        
        # 4. Layer-wise attention patterns
        self._create_layer_analysis_plots(attention_results)
        
        # 5. Financial term attention flow
        self._create_financial_attention_flow(attention_results)
        
        # 6. All token types attention analysis
        self._create_token_type_analysis(attention_results)
        
        print(f"Visualizations saved to {self.output_dir}/visualizations/")
    
    def _create_attention_heatmaps(self, attention_results: Dict):
        """Create attention heatmaps for sample predictions."""
        
        # Select representative samples
        correct_samples = attention_results['correct_predictions'][:2]
        incorrect_samples = attention_results['incorrect_predictions'][:2]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Attention Patterns: Correct vs Incorrect Predictions', fontsize=16)
        
        samples = [
            (correct_samples, "Correct Predictions"),
            (incorrect_samples, "Incorrect Predictions")
        ]
        
        for col, (sample_list, title) in enumerate(samples):
            for row, sample in enumerate(sample_list[:2]):
                if row < 2:  # Only plot first 2 samples
                    ax = axes[row, col]
                    
                    # Get attention from middle layer
                    layer_keys = list(sample['layers'].keys())
                    mid_layer = layer_keys[len(layer_keys)//2] if layer_keys else 0
                    
                    if mid_layer in sample['layers']:
                        # Average across heads
                        attention_matrix = sample['layers'][mid_layer]['attention_matrix']
                        avg_attention = np.mean(attention_matrix, axis=0)
                        tokens = sample['tokens']
                        
                        # Find financial statement region
                        financial_start, financial_end = self._find_financial_data_region(tokens)
                        
                        if financial_start is not None and financial_end is not None:
                            # Show attention within financial data region
                            viz_attention = avg_attention[financial_start:financial_end, financial_start:financial_end]
                            
                            sns.heatmap(viz_attention, ax=ax, cmap='Blues', 
                                      cbar=row==0 and col==1)
                            
                            ax.set_title(f'{title}\n{sample["sample_id"]} (Layer {mid_layer})\nFinancial Data Region: {financial_start}-{financial_end}')
                            ax.set_xlabel('Token Position (Financial Data)')
                            ax.set_ylabel('Token Position (Financial Data)')
                        else:
                            # Fallback to first 50 tokens
                            max_tokens = min(50, avg_attention.shape[0])
                            viz_attention = avg_attention[:max_tokens, :max_tokens]
                            
                            sns.heatmap(viz_attention, ax=ax, cmap='Blues', 
                                      cbar=row==0 and col==1)
                            
                            ax.set_title(f'{title}\n{sample["sample_id"]} (Layer {mid_layer})\nFirst 50 tokens')
                            ax.set_xlabel('Token Position')
                            ax.set_ylabel('Token Position')
        
        # Remove empty subplots
        for row in range(2):
            for col in range(2):
                if (col == 0 and len(correct_samples) <= row) or \
                   (col == 1 and len(incorrect_samples) <= row):
                    axes[row, col].remove()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/visualizations/attention_heatmaps.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_financial_attention_comparison(self, comparison_results: Dict):
        """Create bar plot comparing financial attention between correct/incorrect."""
        
        fin_diff = comparison_results.get('financial_attention_differences', {})
        
        if not fin_diff:
            return
        
        categories = list(fin_diff.keys())
        correct_means = [fin_diff[cat]['correct_mean'] for cat in categories]
        incorrect_means = [fin_diff[cat]['incorrect_mean'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, correct_means, width, label='Correct Predictions', 
                      color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, incorrect_means, width, label='Incorrect Predictions', 
                      color='red', alpha=0.7)
        
        ax.set_xlabel('Financial Vocabulary Categories')
        ax.set_ylabel('Mean Attention Score')
        ax.set_title('Financial Vocabulary Attention: Correct vs Incorrect Predictions')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/visualizations/financial_attention_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_head_specialization_plots(self, comparison_results: Dict):
        """Create plots showing head specialization differences."""
        
        head_diff = comparison_results.get('head_specialization_differences', {})
        
        if not head_diff:
            return
        
        patterns = list(head_diff.keys())
        differences = [head_diff[pattern]['difference'] for pattern in patterns]
        effect_sizes = [head_diff[pattern]['effect_size'] for pattern in patterns]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean differences
        colors = ['green' if d > 0 else 'red' for d in differences]
        bars1 = ax1.bar(patterns, differences, color=colors, alpha=0.7)
        ax1.set_title('Head Pattern Differences\n(Correct - Incorrect)')
        ax1.set_ylabel('Mean Difference')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Effect sizes
        colors2 = ['green' if es > 0.2 else 'orange' if es > 0 else 'red' for es in effect_sizes]
        bars2 = ax2.bar(patterns, effect_sizes, color=colors2, alpha=0.7)
        ax2.set_title('Effect Sizes')
        ax2.set_ylabel('Cohen\'s d')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small Effect')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/visualizations/head_specialization_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_layer_analysis_plots(self, attention_results: Dict):
        """Create plots showing layer-wise attention patterns."""
        
        # Aggregate data by layer
        layer_data = defaultdict(lambda: {'correct': [], 'incorrect': []})
        
        for sample in attention_results['correct_predictions']:
            for layer_idx in sample['layers'].keys():
                # Average financial attention across vocabulary categories
                fin_attn = sample['layers'][layer_idx]['financial_attention']
                avg_fin_attn = np.mean([cat_data['mean_attention'] 
                                      for cat_data in fin_attn.values() 
                                      if cat_data['mean_attention'] > 0])
                layer_data[layer_idx]['correct'].append(avg_fin_attn if not np.isnan(avg_fin_attn) else 0)
        
        for sample in attention_results['incorrect_predictions']:
            for layer_idx in sample['layers'].keys():
                fin_attn = sample['layers'][layer_idx]['financial_attention']
                avg_fin_attn = np.mean([cat_data['mean_attention'] 
                                      for cat_data in fin_attn.values() 
                                      if cat_data['mean_attention'] > 0])
                layer_data[layer_idx]['incorrect'].append(avg_fin_attn if not np.isnan(avg_fin_attn) else 0)
        
        # Create plot
        layers = sorted(layer_data.keys())
        correct_means = [np.mean(layer_data[l]['correct']) if layer_data[l]['correct'] else 0 
                        for l in layers]
        incorrect_means = [np.mean(layer_data[l]['incorrect']) if layer_data[l]['incorrect'] else 0 
                          for l in layers]
        
        plt.figure(figsize=(12, 6))
        plt.plot(layers, correct_means, 'g-o', label='Correct Predictions', linewidth=2)
        plt.plot(layers, incorrect_means, 'r-o', label='Incorrect Predictions', linewidth=2)
        
        plt.xlabel('Layer Index')
        plt.ylabel('Average Financial Attention')
        plt.title('Layer-wise Financial Attention Patterns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/visualizations/layer_wise_attention.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_financial_attention_flow(self, attention_results: Dict):
        """Create visualization showing attention flow from financial terms to predictions."""
        
        # Combine all samples for analysis
        all_samples = attention_results['correct_predictions'] + attention_results['incorrect_predictions']
        
        if not all_samples:
            return
        
        # Take first sample for detailed analysis
        sample = all_samples[0]
        if not sample['layers']:
            return
        
        # Get middle layer for analysis
        layer_keys = list(sample['layers'].keys())
        mid_layer = layer_keys[len(layer_keys)//2] if layer_keys else 0
        
        if mid_layer not in sample['layers']:
            return
        
        tokens = sample['tokens']
        attention_matrix = sample['layers'][mid_layer]['attention_matrix']
        avg_attention = np.mean(attention_matrix, axis=0)  # Average across heads
        
        # Find important financial terms and prediction terms
        financial_positions = []
        prediction_positions = []
        
        important_financial_terms = ['sales', 'revenue', 'income', 'asset', 'liability', 'cash', 'debt', 'profit']
        prediction_terms = ['increase', 'decrease', 'up', 'down']
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if any(term in token_lower for term in important_financial_terms):
                financial_positions.append((i, token))
            if any(term in token_lower for term in prediction_terms):
                prediction_positions.append((i, token))
        
        if not financial_positions or not prediction_positions:
            return
        
        # Create attention flow visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Attention FROM financial terms TO all other tokens
        financial_indices = [pos for pos, _ in financial_positions[:10]]  # Limit to first 10
        attention_from_financial = avg_attention[financial_indices, :]
        
        im1 = ax1.imshow(attention_from_financial, cmap='Blues', aspect='auto')
        ax1.set_title(f'Attention FROM Financial Terms\n{sample["sample_id"]} (Layer {mid_layer})')
        ax1.set_xlabel('All Token Positions')
        ax1.set_ylabel('Financial Terms')
        
        # Add labels for financial terms
        financial_labels = [f"{pos}: {token[:10]}" for pos, token in financial_positions[:10]]
        ax1.set_yticks(range(len(financial_labels)))
        ax1.set_yticklabels(financial_labels, fontsize=8)
        
        plt.colorbar(im1, ax=ax1, label='Attention Score')
        
        # Plot 2: Attention TO prediction terms FROM all other tokens
        prediction_indices = [pos for pos, _ in prediction_positions[:5]]  # Limit to first 5
        if prediction_indices:
            attention_to_prediction = avg_attention[:, prediction_indices]
            
            im2 = ax2.imshow(attention_to_prediction.T, cmap='Reds', aspect='auto')
            ax2.set_title(f'Attention TO Prediction Terms\n{sample["sample_id"]} (Layer {mid_layer})')
            ax2.set_xlabel('All Token Positions')
            ax2.set_ylabel('Prediction Terms')
            
            # Add labels for prediction terms
            prediction_labels = [f"{pos}: {token}" for pos, token in prediction_positions[:5]]
            ax2.set_yticks(range(len(prediction_labels)))
            ax2.set_yticklabels(prediction_labels, fontsize=8)
            
            plt.colorbar(im2, ax=ax2, label='Attention Score')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/visualizations/financial_attention_flow.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary plot showing key attention connections
        self._create_attention_summary_plot(attention_results)
    
    def _create_attention_summary_plot(self, attention_results: Dict):
        """Create summary plot showing which parts of input get most attention."""
        
        # Aggregate attention across all samples
        correct_samples = attention_results['correct_predictions']
        incorrect_samples = attention_results['incorrect_predictions']
        
        # Analysis for different input regions
        regions = {
            'Instructions (0-100)': (0, 100),
            'Balance Sheet (100-800)': (100, 800), 
            'Income Statement (800-1500)': (800, 1500),
            'Analysis Steps (1500+)': (1500, 2048)
        }
        
        correct_attention_by_region = {region: [] for region in regions}
        incorrect_attention_by_region = {region: [] for region in regions}
        
        # Collect attention scores by region
        for sample in correct_samples:
            if sample['layers']:
                layer_keys = list(sample['layers'].keys())
                mid_layer = layer_keys[len(layer_keys)//2]
                
                if mid_layer in sample['layers']:
                    attention_matrix = sample['layers'][mid_layer]['attention_matrix']
                    avg_attention = np.mean(attention_matrix, axis=(0, 1))  # Average across heads and positions
                    
                    for region_name, (start, end) in regions.items():
                        end = min(end, len(avg_attention))
                        if start < len(avg_attention):
                            region_attention = np.mean(avg_attention[start:end])
                            correct_attention_by_region[region_name].append(region_attention)
        
        for sample in incorrect_samples:
            if sample['layers']:
                layer_keys = list(sample['layers'].keys())
                mid_layer = layer_keys[len(layer_keys)//2]
                
                if mid_layer in sample['layers']:
                    attention_matrix = sample['layers'][mid_layer]['attention_matrix']
                    avg_attention = np.mean(attention_matrix, axis=(0, 1))
                    
                    for region_name, (start, end) in regions.items():
                        end = min(end, len(avg_attention))
                        if start < len(avg_attention):
                            region_attention = np.mean(avg_attention[start:end])
                            incorrect_attention_by_region[region_name].append(region_attention)
        
        # Create comparison plot
        region_names = list(regions.keys())
        correct_means = [np.mean(correct_attention_by_region[region]) if correct_attention_by_region[region] else 0 
                        for region in region_names]
        incorrect_means = [np.mean(incorrect_attention_by_region[region]) if incorrect_attention_by_region[region] else 0 
                          for region in region_names]
        
        x = np.arange(len(region_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, correct_means, width, label='Correct Predictions', 
                      color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, incorrect_means, width, label='Incorrect Predictions', 
                      color='red', alpha=0.7)
        
        ax.set_xlabel('Input Regions')
        ax.set_ylabel('Average Attention Score')
        ax.set_title('Attention Distribution Across Input Regions:\nCorrect vs Incorrect Predictions')
        ax.set_xticks(x)
        ax.set_xticklabels(region_names, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/visualizations/attention_by_input_region.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_token_type_analysis(self, attention_results: Dict):
        """Create comprehensive analysis of attention by token type."""
        
        # Combine all samples for comprehensive analysis
        all_samples = attention_results['correct_predictions'] + attention_results['incorrect_predictions']
        
        if not all_samples:
            return
        
        # Analyze token types across all samples
        all_token_data = []
        
        for sample in all_samples:
            if not sample['layers']:
                continue
                
            tokens = sample['tokens']
            
            # Get middle layer attention
            layer_keys = list(sample['layers'].keys())
            mid_layer = layer_keys[len(layer_keys)//2] if layer_keys else 0
            
            if mid_layer not in sample['layers']:
                continue
            
            attention_matrix = sample['layers'][mid_layer]['attention_matrix']
            avg_attention = np.mean(attention_matrix, axis=0)  # Average across heads
            
            for i, token in enumerate(tokens):
                # Calculate attention received by this token
                attention_received = float(np.mean(avg_attention[:, i]))
                
                # Classify token type
                token_type = self._classify_token_type(token)
                
                all_token_data.append({
                    'token': token,
                    'position': i,
                    'attention': attention_received,
                    'type': token_type,
                    'sample_id': sample['sample_id'],
                    'is_correct': sample['is_correct']
                })
        
        # Create visualizations
        self._plot_attention_by_token_type(all_token_data)
        self._plot_top_tokens_by_type(all_token_data)
    
    def _classify_token_type(self, token: str) -> str:
        """Classify token into different types."""
        token_lower = token.lower().strip()
        
        # Financial terms (check against vocabulary)
        for category, vocab_terms in self.financial_vocab.items():
            if any(term in token_lower for term in vocab_terms):
                return 'FINANCIAL'
        
        # Prediction terms
        if token_lower in ['increase', 'decrease', 'up', 'down', 'rise', 'fall']:
            return 'PREDICTION'
        
        # Numeric values
        if any(char.isdigit() for char in token):
            return 'NUMERIC'
        
        # Table formatting
        if token in ['|', '-', '=', ':', '.', ',', '(', ')', '[', ']', '{', '}']:
            return 'FORMATTING'
        
        # Short tokens (likely punctuation or formatting)
        if len(token.strip()) <= 2:
            return 'PUNCTUATION'
        
        # Common stop words
        if token_lower in ['the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were']:
            return 'STOPWORD'
        
        # Analysis keywords
        if token_lower in ['analyze', 'assessment', 'step', 'solve', 'problem', 'based', 'following', 'statements']:
            return 'INSTRUCTION'
        
        # Everything else
        return 'OTHER'
    
    def _plot_attention_by_token_type(self, all_token_data: List[Dict]):
        """Plot attention distribution by token type."""
        
        # Group by token type
        type_attention = {}
        for token_data in all_token_data:
            token_type = token_data['type']
            if token_type not in type_attention:
                type_attention[token_type] = []
            type_attention[token_type].append(token_data['attention'])
        
        # Calculate statistics
        type_stats = {}
        for token_type, attentions in type_attention.items():
            type_stats[token_type] = {
                'mean': np.mean(attentions),
                'std': np.std(attentions),
                'count': len(attentions),
                'median': np.median(attentions),
                'max': np.max(attentions)
            }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Average attention by token type
        types = list(type_stats.keys())
        means = [type_stats[t]['mean'] for t in types]
        stds = [type_stats[t]['std'] for t in types]
        counts = [type_stats[t]['count'] for t in types]
        
        # Color code by attention level
        colors = ['red' if m > 0.01 else 'orange' if m > 0.005 else 'blue' for m in means]
        
        bars = ax1.bar(types, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax1.set_title('Average Attention by Token Type')
        ax1.set_ylabel('Mean Attention Score')
        ax1.set_xlabel('Token Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.annotate(f'n={count}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Token count distribution
        ax2.bar(types, counts, color='green', alpha=0.7)
        ax2.set_title('Number of Tokens by Type')
        ax2.set_ylabel('Token Count')
        ax2.set_xlabel('Token Type')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total_tokens = sum(counts)
        for i, (bar, count) in enumerate(zip(ax2.patches, counts)):
            percentage = (count / total_tokens) * 100
            height = bar.get_height()
            ax2.annotate(f'{percentage:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/visualizations/attention_by_token_type.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_tokens_by_type(self, all_token_data: List[Dict]):
        """Plot top attended tokens for each type."""
        
        # Group by token type and find top tokens
        type_top_tokens = {}
        for token_type in ['FINANCIAL', 'NUMERIC', 'PREDICTION', 'FORMATTING', 'INSTRUCTION', 'OTHER']:
            type_tokens = [t for t in all_token_data if t['type'] == token_type]
            if type_tokens:
                # Sort by attention and take top 10
                top_tokens = sorted(type_tokens, key=lambda x: x['attention'], reverse=True)[:10]
                type_top_tokens[token_type] = top_tokens
        
        # Create subplot for each type
        num_types = len(type_top_tokens)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, (token_type, top_tokens) in enumerate(type_top_tokens.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if top_tokens:
                tokens = [t['token'][:10] for t in top_tokens]  # Truncate long tokens
                attentions = [t['attention'] for t in top_tokens]
                
                bars = ax.barh(range(len(tokens)), attentions, alpha=0.7)
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens, fontsize=8)
                ax.set_xlabel('Attention Score')
                ax.set_title(f'Top {token_type} Tokens')
                ax.invert_yaxis()  # Show highest attention at top
                
                # Add attention values on bars
                for j, (bar, attention) in enumerate(zip(bars, attentions)):
                    width = bar.get_width()
                    ax.annotate(f'{attention:.4f}',
                               xy=(width, bar.get_y() + bar.get_height() / 2),
                               xytext=(3, 0),
                               textcoords="offset points",
                               va='center', fontsize=8)
        
        # Remove empty subplots
        for i in range(len(type_top_tokens), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/visualizations/top_tokens_by_type.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_results(self, attention_results: Dict, comparison_results: Dict):
        """Save comprehensive analysis results to JSON files."""
        
        # Prepare serializable data
        serializable_attention = self._make_serializable(attention_results)
        serializable_comparison = self._make_serializable(comparison_results)
        
        # Save attention data
        with open(f'{self.output_dir}/attention_analysis_results.json', 'w') as f:
            json.dump(serializable_attention, f, indent=2)
        
        # Save comparison results
        with open(f'{self.output_dir}/comparison_analysis_results.json', 'w') as f:
            json.dump(serializable_comparison, f, indent=2)
        
        # Save summary report
        self._generate_summary_report(attention_results, comparison_results)
        
        # Save detailed token analysis
        self._save_detailed_token_analysis(attention_results)
        
        # Save generated text outputs
        self._save_generated_outputs(attention_results)
        
        print(f"Analysis results saved to {self.output_dir}")
    
    def _make_serializable(self, data):
        """Convert numpy arrays and other non-serializable objects to lists."""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        else:
            return data
    
    def _save_detailed_token_analysis(self, attention_results: Dict):
        """Save detailed token-level attention analysis for manual inspection."""
        
        # Take first sample for detailed analysis
        all_samples = attention_results['correct_predictions'] + attention_results['incorrect_predictions']
        if not all_samples:
            return
        
        sample = all_samples[0]
        if not sample['layers']:
            return
        
        tokens = sample['tokens']
        
        # Create detailed token analysis
        token_analysis = {
            'sample_id': sample['sample_id'],
            'total_tokens': len(tokens),
            'token_details': []
        }
        
        # Analyze each token
        for i, token in enumerate(tokens):
            token_info = {
                'position': i,
                'token': token,
                'is_financial': False,
                'financial_categories': [],
                'attention_received': {},
                'attention_given': {}
            }
            
            # Check if token is financial
            for category, vocab_terms in self.financial_vocab.items():
                if any(term in token.lower() for term in vocab_terms):
                    token_info['is_financial'] = True
                    token_info['financial_categories'].append(category)
            
            # Get attention data from each layer
            for layer_idx, layer_data in sample['layers'].items():
                attention_matrix = layer_data['attention_matrix']
                avg_attention = np.mean(attention_matrix, axis=0)  # Average across heads
                
                # Attention received by this token
                attention_received = float(np.mean(avg_attention[:, i]))
                token_info['attention_received'][f'layer_{layer_idx}'] = attention_received
                
                # Attention given by this token
                attention_given = float(np.mean(avg_attention[i, :]))
                token_info['attention_given'][f'layer_{layer_idx}'] = attention_given
            
            token_analysis['token_details'].append(token_info)
        
        # Save detailed analysis
        with open(f'{self.output_dir}/detailed_token_analysis.json', 'w') as f:
            json.dump(token_analysis, f, indent=2)
        
        # Create readable summary of top attended tokens
        self._create_top_tokens_summary(token_analysis)
    
    def _create_top_tokens_summary(self, token_analysis: Dict):
        """Create human-readable summary of most attended tokens."""
        
        summary = []
        summary.append("TOP ATTENDED TOKENS ANALYSIS")
        summary.append("=" * 40)
        summary.append(f"Sample: {token_analysis['sample_id']}")
        summary.append(f"Total tokens: {token_analysis['total_tokens']}")
        summary.append("")
        
        # Calculate average attention received across all layers
        for token_info in token_analysis['token_details']:
            if token_info['attention_received']:
                avg_attention = np.mean(list(token_info['attention_received'].values()))
                token_info['avg_attention_received'] = avg_attention
            else:
                token_info['avg_attention_received'] = 0
        
        # Sort by attention received
        sorted_tokens = sorted(token_analysis['token_details'], 
                              key=lambda x: x['avg_attention_received'], 
                              reverse=True)
        
        # Top 50 most attended tokens (ALL tokens, regardless of type)
        summary.append("TOP 50 MOST ATTENDED TOKENS (ALL TYPES):")
        summary.append("-" * 45)
        summary.append("Rank | Position | Token                | Attention | Type")
        summary.append("-" * 65)
        
        for i, token_info in enumerate(sorted_tokens[:50]):
            if token_info['is_financial']:
                type_info = f"FINANCIAL ({', '.join(token_info['financial_categories'][:2])})"[:25]
            else:
                # Classify non-financial tokens
                token_lower = token_info['token'].lower().strip()
                if token_lower in ['increase', 'decrease', 'up', 'down', 'rise', 'fall']:
                    type_info = "PREDICTION"
                elif any(char.isdigit() for char in token_info['token']):
                    type_info = "NUMERIC"
                elif token_info['token'] in ['|', '-', '=', ':', '.', ',', '(', ')']:
                    type_info = "FORMATTING"
                elif len(token_info['token'].strip()) <= 2:
                    type_info = "PUNCTUATION"
                elif token_lower in ['the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'by']:
                    type_info = "STOPWORD"
                else:
                    type_info = "OTHER"
            
            summary.append(f"{i+1:4d} | {token_info['position']:8d} | {token_info['token'][:20]:20s} | "
                          f"{token_info['avg_attention_received']:9.4f} | {type_info}")
        
        summary.append("")
        summary.append("ATTENTION DISTRIBUTION BY TOKEN TYPE:")
        summary.append("-" * 40)
        
        # Analyze distribution by token type
        type_counts = {
            'FINANCIAL': 0, 'NUMERIC': 0, 'PREDICTION': 0, 
            'FORMATTING': 0, 'PUNCTUATION': 0, 'STOPWORD': 0, 'OTHER': 0
        }
        type_attention_sums = {key: 0.0 for key in type_counts.keys()}
        
        for token_info in sorted_tokens[:50]:
            if token_info['is_financial']:
                token_type = 'FINANCIAL'
            else:
                token_lower = token_info['token'].lower().strip()
                if token_lower in ['increase', 'decrease', 'up', 'down', 'rise', 'fall']:
                    token_type = 'PREDICTION'
                elif any(char.isdigit() for char in token_info['token']):
                    token_type = 'NUMERIC'
                elif token_info['token'] in ['|', '-', '=', ':', '.', ',', '(', ')']:
                    token_type = 'FORMATTING'
                elif len(token_info['token'].strip()) <= 2:
                    token_type = 'PUNCTUATION'
                elif token_lower in ['the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'by']:
                    token_type = 'STOPWORD'
                else:
                    token_type = 'OTHER'
            
            type_counts[token_type] += 1
            type_attention_sums[token_type] += token_info['avg_attention_received']
        
        for token_type in type_counts.keys():
            count = type_counts[token_type]
            total_attention = type_attention_sums[token_type]
            avg_attention = total_attention / count if count > 0 else 0
            percentage = (count / 50) * 100
            
            summary.append(f"{token_type:12s}: {count:2d} tokens ({percentage:5.1f}%) | "
                          f"Avg attention: {avg_attention:.4f} | Total: {total_attention:.4f}")
        
        summary.append("")
        summary.append("TOP FINANCIAL TOKENS BY ATTENTION:")
        summary.append("-" * 35)
        
        # Filter and sort financial tokens
        financial_tokens = [t for t in token_analysis['token_details'] if t['is_financial']]
        financial_sorted = sorted(financial_tokens, 
                                 key=lambda x: x['avg_attention_received'], 
                                 reverse=True)
        
        for i, token_info in enumerate(financial_sorted[:15]):
            summary.append(f"{i+1:2d}. Position {token_info['position']:4d}: '{token_info['token'][:15]:15s}' "
                          f"(attention: {token_info['avg_attention_received']:.4f}) "
                          f"[{', '.join(token_info['financial_categories'])}]")
        
        # Save summary
        with open(f'{self.output_dir}/top_tokens_summary.txt', 'w') as f:
            f.write('\n'.join(summary))
    
    def _save_generated_outputs(self, attention_results: Dict):
        """Save all generated text outputs for easy review."""
        
        outputs_summary = []
        outputs_summary.append("GENERATED TEXT OUTPUTS ANALYSIS")
        outputs_summary.append("=" * 40)
        outputs_summary.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        outputs_summary.append("")
        
        # Process correct predictions
        outputs_summary.append("CORRECT PREDICTIONS:")
        outputs_summary.append("-" * 20)
        
        for i, sample in enumerate(attention_results['correct_predictions'], 1):
            outputs_summary.append(f"\n{i}. Sample: {sample['sample_id']}")
            outputs_summary.append(f"   Expected: {sample['actual_label']}")
            outputs_summary.append(f"   Generated: \"{sample['generated_response']}\"")
            outputs_summary.append(f"   Status: ✓ CORRECT")
        
        # Process incorrect predictions
        outputs_summary.append("\n\nINCORRECT PREDICTIONS:")
        outputs_summary.append("-" * 22)
        
        for i, sample in enumerate(attention_results['incorrect_predictions'], 1):
            outputs_summary.append(f"\n{i}. Sample: {sample['sample_id']}")
            outputs_summary.append(f"   Expected: {sample['actual_label']}")
            outputs_summary.append(f"   Generated: \"{sample['generated_response']}\"")
            outputs_summary.append(f"   Status: ✗ INCORRECT")
        
        # Add statistics
        correct_count = len(attention_results['correct_predictions'])
        incorrect_count = len(attention_results['incorrect_predictions'])
        total_count = correct_count + incorrect_count
        
        outputs_summary.append(f"\n\nSTATISTICS:")
        outputs_summary.append("-" * 11)
        outputs_summary.append(f"Total samples: {total_count}")
        outputs_summary.append(f"Correct: {correct_count} ({correct_count/total_count*100:.1f}%)")
        outputs_summary.append(f"Incorrect: {incorrect_count} ({incorrect_count/total_count*100:.1f}%)")
        
        # Save to file
        with open(f'{self.output_dir}/generated_outputs_summary.txt', 'w') as f:
            f.write('\n'.join(outputs_summary))
        
        # Also save detailed outputs as JSON for programmatic access
        detailed_outputs = {
            'correct_predictions': [],
            'incorrect_predictions': [],
            'statistics': {
                'total': total_count,
                'correct': correct_count,
                'incorrect': incorrect_count,
                'accuracy': correct_count / total_count if total_count > 0 else 0
            }
        }
        
        for sample in attention_results['correct_predictions']:
            detailed_outputs['correct_predictions'].append({
                'sample_id': sample['sample_id'],
                'expected_label': sample['actual_label'],
                'predicted_label': sample.get('predicted_label', 'Unknown'),
                'generated_response': sample['generated_response'],
                'full_generated_text': sample['full_generated_text'],
                'input_prompt': sample.get('original_prompt', 'Prompt not available'),
                'tokens_count': sample.get('token_count', 0),
                'is_correct': True
            })
        
        for sample in attention_results['incorrect_predictions']:
            detailed_outputs['incorrect_predictions'].append({
                'sample_id': sample['sample_id'],
                'expected_label': sample['actual_label'],
                'predicted_label': sample.get('predicted_label', 'Unknown'),
                'generated_response': sample['generated_response'],
                'full_generated_text': sample['full_generated_text'],
                'input_prompt': sample.get('original_prompt', 'Prompt not available'),
                'tokens_count': sample.get('token_count', 0),
                'is_correct': False
            })
        
        with open(f'{self.output_dir}/generated_outputs_detailed.json', 'w') as f:
            json.dump(detailed_outputs, f, indent=2)
        
        # Save comprehensive full text outputs
        self._save_comprehensive_outputs(attention_results)
    
    def _save_comprehensive_outputs(self, attention_results: Dict):
        """Save comprehensive full text outputs with complete prompts and responses."""
        
        comprehensive_output = []
        comprehensive_output.append("COMPREHENSIVE FULL TEXT OUTPUTS")
        comprehensive_output.append("=" * 50)
        comprehensive_output.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        comprehensive_output.append("")
        
        # Process all samples
        all_samples = attention_results['correct_predictions'] + attention_results['incorrect_predictions']
        
        for i, sample in enumerate(all_samples, 1):
            status = "✓ CORRECT" if sample['is_correct'] else "✗ INCORRECT"
            
            comprehensive_output.append(f"\n{'='*80}")
            comprehensive_output.append(f"SAMPLE {i}: {sample['sample_id']} - {status}")
            comprehensive_output.append(f"{'='*80}")
            
            comprehensive_output.append(f"\nEXPECTED LABEL: {sample['actual_label']}")
            comprehensive_output.append(f"PREDICTED LABEL: {sample.get('predicted_label', 'Unknown')}")
            comprehensive_output.append(f"IS CORRECT: {sample['is_correct']}")
            
            comprehensive_output.append(f"\n{'-'*40} INPUT PROMPT {'-'*40}")
            comprehensive_output.append(sample.get('original_prompt', 'Prompt not available'))
            
            comprehensive_output.append(f"\n{'-'*35} FULL GENERATED TEXT {'-'*35}")
            comprehensive_output.append(sample['full_generated_text'])
            
            comprehensive_output.append(f"\n{'-'*35} GENERATED RESPONSE ONLY {'-'*30}")
            comprehensive_output.append(f"'{sample['generated_response']}'")
            
            comprehensive_output.append(f"\n{'-'*35} TOKEN STATISTICS {'-'*35}")
            comprehensive_output.append(f"Total tokens: {sample.get('token_count', 0)}")
            comprehensive_output.append(f"Input length: {sample.get('input_length', 0)}")
            comprehensive_output.append(f"Generated tokens: {len(sample.get('generated_tokens', []))}")
            
            comprehensive_output.append("")
        
        # Save comprehensive output
        with open(f'{self.output_dir}/comprehensive_full_outputs.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(comprehensive_output))
    
    def _reconstruct_prompt_from_tokens(self, tokens: List[str]) -> str:
        """Reconstruct the original prompt from tokenized input."""
        if not tokens:
            return "Prompt not available"
        try:
            # Convert tokens back to text
            return self.tokenizer.convert_tokens_to_string(tokens)
        except Exception as e:
            print(f"Warning: Could not reconstruct prompt from tokens: {e}")
            return "Prompt reconstruction failed"
    
    def _generate_summary_report(self, attention_results: Dict, comparison_results: Dict):
        """Generate human-readable summary report."""
        
        report = []
        report.append("MECHANISTIC INTERPRETABILITY ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Model: {self.model_name}")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Sample statistics
        correct_count = len(attention_results['correct_predictions'])
        incorrect_count = len(attention_results['incorrect_predictions'])
        total_samples = correct_count + incorrect_count
        
        report.append("SAMPLE STATISTICS")
        report.append("-" * 20)
        report.append(f"Total samples analyzed: {total_samples}")
        report.append(f"Correct predictions: {correct_count} ({correct_count/total_samples*100:.1f}%)")
        report.append(f"Incorrect predictions: {incorrect_count} ({incorrect_count/total_samples*100:.1f}%)")
        report.append("")
        
        # Financial attention differences
        if 'financial_attention_differences' in comparison_results:
            report.append("FINANCIAL VOCABULARY ATTENTION DIFFERENCES")
            report.append("-" * 45)
            
            fin_diff = comparison_results['financial_attention_differences']
            for category, data in fin_diff.items():
                diff = data['difference']
                direction = "higher" if diff > 0 else "lower"
                report.append(f"{category}: Correct predictions show {direction} attention "
                            f"(diff: {diff:.4f})")
            report.append("")
        
        # Head specialization differences
        if 'head_specialization_differences' in comparison_results:
            report.append("HEAD SPECIALIZATION PATTERN DIFFERENCES")
            report.append("-" * 40)
            
            head_diff = comparison_results['head_specialization_differences']
            for pattern, data in head_diff.items():
                diff = data['difference']
                effect_size = data['effect_size']
                direction = "higher" if diff > 0 else "lower"
                
                effect_desc = "large" if abs(effect_size) > 0.8 else \
                             "medium" if abs(effect_size) > 0.5 else \
                             "small" if abs(effect_size) > 0.2 else "negligible"
                
                report.append(f"{pattern}: Correct predictions show {direction} scores "
                            f"(effect size: {effect_size:.3f} - {effect_desc})")
            report.append("")
        
        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 12)
        
        # Analyze which patterns show strongest differences
        if 'head_specialization_differences' in comparison_results:
            head_diff = comparison_results['head_specialization_differences']
            max_effect = max(abs(data['effect_size']) for data in head_diff.values())
            max_pattern = max(head_diff.items(), key=lambda x: abs(x[1]['effect_size']))
            
            report.append(f"• Strongest difference in {max_pattern[0]} pattern "
                        f"(effect size: {max_pattern[1]['effect_size']:.3f})")
        
        if 'financial_attention_differences' in comparison_results:
            fin_diff = comparison_results['financial_attention_differences']
            max_fin_diff = max(abs(data['difference']) for data in fin_diff.values())
            max_fin_cat = max(fin_diff.items(), key=lambda x: abs(x[1]['difference']))
            
            report.append(f"• Largest financial attention difference in {max_fin_cat[0]} "
                        f"(difference: {max_fin_cat[1]['difference']:.4f})")
        
        report.append("")
        report.append("FILES GENERATED")
        report.append("-" * 15)
        report.append("• attention_analysis_results.json - Full attention data")
        report.append("• comparison_analysis_results.json - Statistical comparisons")
        report.append("• detailed_token_analysis.json - Token-level attention scores")
        report.append("• top_tokens_summary.txt - Most attended tokens summary")
        report.append("• generated_outputs_summary.txt - All model predictions and expected labels")
        report.append("• generated_outputs_detailed.json - Detailed prediction outputs in JSON format")
        report.append("• comprehensive_full_outputs.txt - Complete prompts and full generated text for all samples")
        report.append("• visualizations/attention_heatmaps.png - Financial data region heatmaps")
        report.append("• visualizations/financial_attention_flow.png - Attention flow analysis")
        report.append("• visualizations/attention_by_input_region.png - Regional attention distribution")
        report.append("• visualizations/attention_by_token_type.png - ALL token types attention analysis")
        report.append("• visualizations/top_tokens_by_type.png - Top tokens for each category")
        report.append("• visualizations/financial_attention_comparison.png - Vocabulary attention")
        report.append("• visualizations/head_specialization_comparison.png - Head pattern analysis")
        report.append("• visualizations/layer_wise_attention.png - Layer progression")
        report.append("• attention_matrices/ - Raw attention matrices")
        
        # Save report
        with open(f'{self.output_dir}/analysis_summary_report.txt', 'w') as f:
            f.write('\n'.join(report))
    
    def run_full_analysis(self, predictions_file: str, max_samples: int = 20):
        """Run complete mechanistic interpretability analysis."""
        
        print("Starting Mechanistic Interpretability Analysis")
        print("=" * 50)
        
        # Load prediction data
        print(f"Loading predictions from {predictions_file}")
        with open(predictions_file, 'r') as f:
            all_predictions = json.load(f)
        
        # Sample data for analysis
        correct_predictions = [p for p in all_predictions if p.get('is_match', False)]
        incorrect_predictions = [p for p in all_predictions if not p.get('is_match', False)]
        
        # Balance the dataset
        max_per_category = max_samples // 2
        sample_data = (
            correct_predictions[:max_per_category] + 
            incorrect_predictions[:max_per_category]
        )
        
        print(f"Analyzing {len(sample_data)} samples:")
        print(f"  - Correct: {len([s for s in sample_data if s['is_match']])}")
        print(f"  - Incorrect: {len([s for s in sample_data if not s['is_match']])}")
        print()
        
        # Extract attention patterns
        print("Step 1: Extracting attention patterns...")
        attention_results = self.extract_attention_with_predictions(sample_data)
        
        # Compare correct vs incorrect
        print("Step 2: Comparing correct vs incorrect predictions...")
        comparison_results = self.compare_correct_vs_incorrect(attention_results)
        
        # Create visualizations
        print("Step 3: Creating visualizations...")
        self.create_visualizations(attention_results, comparison_results)
        
        # Save results
        print("Step 4: Saving analysis results...")
        self.save_analysis_results(attention_results, comparison_results)
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {self.output_dir}")
        
        return attention_results, comparison_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Mechanistic Interpretability Analysis')
    parser.add_argument('--config', default='nlp/config/interpretability_config.yaml',
                       help='Path to interpretability config file')
    parser.add_argument('--predictions', default='nlp/outputs/predictions_checkpoint_1800.json',
                       help='Path to predictions JSON file')
    parser.add_argument('--max_samples', type=int, default=20,
                       help='Maximum number of samples to analyze')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FinancialAttentionAnalyzer(args.config)
    
    # Run analysis
    attention_results, comparison_results = analyzer.run_full_analysis(
        args.predictions, args.max_samples
    )
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    correct_count = len(attention_results['correct_predictions'])
    incorrect_count = len(attention_results['incorrect_predictions'])
    
    print(f"Samples analyzed: {correct_count + incorrect_count}")
    print(f"  - Correct predictions: {correct_count}")
    print(f"  - Incorrect predictions: {incorrect_count}")
    
    if comparison_results.get('head_specialization_differences'):
        print("\nTop attention pattern differences:")
        head_diff = comparison_results['head_specialization_differences']
        for pattern, data in sorted(head_diff.items(), 
                                  key=lambda x: abs(x[1]['effect_size']), 
                                  reverse=True)[:3]:
            effect_size = data['effect_size']
            direction = "higher" if data['difference'] > 0 else "lower"
            print(f"  - {pattern}: {direction} in correct predictions "
                  f"(effect size: {effect_size:.3f})")
    
    print(f"\nResults saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()