# Mechanistic Interpretability Guide for Financial Analysis

This guide will help you analyze the internal mechanisms of your Qwen2-7B model when making financial predictions.

## Quick Start

### 1. Memory Management
Due to the model size, you may need to adjust your setup:

```python
# Option 1: Use CPU (slower but more memory)
device = "cpu"

# Option 2: Use smaller model if memory is limited
model_name = "Qwen/Qwen2-1.5B-Instruct"  # Smaller alternative

# Option 3: Set memory limits for MPS
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
```

### 2. Basic Attention Analysis

```python
# Import the tools
from src.interpretability_tools import AttentionHook, FinancialAttentionAnalyzer, quick_attention_analysis

# Quick analysis of a single sample
text = "Your financial statement here..."
analyzer = quick_attention_analysis(model, tokenizer, text, layer_idx=0)

# Visualize attention patterns
analyzer.plot_attention_heatmap(layer_idx=0, head_idx=0)
analyzer.plot_financial_attention_flow(layer_idx=0)
```

### 3. Comparative Analysis

```python
from src.interpretability_tools import ComparativeAnalyzer

# Compare attention patterns between increase/decrease predictions
comparative = ComparativeAnalyzer(model, tokenizer)

increase_samples = [sample1, sample2, ...]  # Your increase prediction samples
decrease_samples = [sample3, sample4, ...]  # Your decrease prediction samples

comparison = comparative.compare_prediction_attention(increase_samples, decrease_samples)
comparative.plot_attention_comparison(comparison)
```

## Detailed Analysis Workflow

### Step 1: Understand Model Architecture

The Qwen2-7B model has:
- 32 transformer layers (layers 0-31)
- 32 attention heads per layer  
- Hidden dimension of 4096

Key layers to focus on:
- **Early layers (0-10)**: Basic pattern recognition, syntax
- **Middle layers (11-21)**: Complex reasoning, entity relationships
- **Late layers (22-31)**: Task-specific processing, final decisions

### Step 2: Identify Financial Attention Patterns

```python
# Analyze financial token attention
financial_tokens = analyzer.identify_financial_tokens()
print(f"Found {len(financial_tokens)} financial terms")

# Compute attention scores between financial terms
scores = analyzer.compute_financial_attention_scores()
```

Look for these patterns:
- **Financial clustering**: Financial terms attending to each other
- **Numerical focus**: Attention to specific numbers/ratios
- **Temporal patterns**: Attention to year indicators, time-related terms
- **Causal relationships**: Attention from effects to causes

### Step 3: Head Specialization Analysis

```python
# Analyze what different attention heads specialize in
def analyze_head_types(analyzer, layer_idx):
    layer_name = f'layer_{layer_idx}'
    attn = analyzer.attention_weights[layer_name]
    
    for head_idx in range(attn.shape[1]):
        attention_matrix = attn[0, head_idx].numpy()
        
        # Calculate specialization metrics
        diagonal_attention = np.diag(attention_matrix).mean()
        beginning_attention = attention_matrix[:, :10].mean()
        entropy = -(attention_matrix * np.log(attention_matrix + 1e-10)).sum(axis=1).mean()
        
        print(f"Head {head_idx}: Diagonal={diagonal_attention:.3f}, "
              f"Beginning={beginning_attention:.3f}, Entropy={entropy:.3f}")
```

Common head types you might find:
- **Self-attention heads**: High diagonal attention (attending to current token)
- **Beginning-focused heads**: High attention to start of sequence  
- **Causal heads**: Strong attention to previous tokens
- **Financial-specialized heads**: High attention to financial terms

### Step 4: Prediction Analysis

```python
# Analyze attention patterns for final predictions
prediction_analysis = analyzer.analyze_prediction_attention(prediction_token="increase")

# Check what the model attends to when making predictions
for layer_name, results in prediction_analysis.items():
    print(f"\n{layer_name}:")
    print(f"  Financial attention: {results['financial_attention']['mean']:.4f}")
    print(f"  Top attended tokens: {results['top_attended_tokens'][:5]}")
```

### Step 5: Intervention Experiments

```python
from src.interpretability_tools import AttentionInterventions

interventions = AttentionInterventions(model, tokenizer)

# Test what happens when you ablate specific attention heads
original_output = model.generate(input_ids, max_new_tokens=20)
ablated_output = interventions.ablate_attention_head(
    layer_idx=15, head_idx=5, input_ids=input_ids, ablation_value=0.0
)

print("Effect of ablating layer 15, head 5:")
print(f"Original: {tokenizer.decode(original_output[0])}")
print(f"Ablated: {tokenizer.decode(ablated_output.sequences[0])}")
```

## Key Research Questions

### 1. Financial Term Processing
- Do certain heads specialize in processing financial terminology?
- How does attention to different financial categories correlate with predictions?
- Which layers are most important for understanding financial relationships?

### 2. Numerical Reasoning
- How does the model process numerical values in financial statements?
- Are there heads that specifically attend to ratios and percentages?
- Does the model show different attention patterns for large vs small numbers?

### 3. Temporal Understanding
- How does the model weight recent vs historical financial data?
- Are there attention patterns that indicate time-series understanding?
- Do later layers show stronger temporal reasoning?

### 4. Decision Making Process
- What information does the model attend to when making the final prediction?
- How do attention patterns differ between confident vs uncertain predictions?
- Can we identify the "decision pathway" through the layers?

## Experimental Protocol

### Hypothesis Testing Framework

1. **Formulate hypothesis** (e.g., "Layer 20+ heads specialize in financial reasoning")
2. **Design experiment** (compare attention patterns across layers)
3. **Collect data** (run analysis on multiple samples)
4. **Statistical testing** (t-tests, correlation analysis)
5. **Visualize results** (heatmaps, bar charts, scatter plots)
6. **Interpret findings** (what does this tell us about the model?)

### Sample Analysis Pipeline

```python
def run_interpretability_experiment(data_samples, research_question):
    results = []
    
    for sample in data_samples:
        # Generate with attention capture
        with AttentionHook(model) as hook:
            output = model.generate(sample, output_attentions=True)
            
        # Analyze attention patterns
        analyzer = FinancialAttentionAnalyzer(hook.attention_weights, ...)
        
        # Extract relevant metrics
        metrics = {
            'financial_attention': analyzer.compute_financial_attention_scores(),
            'prediction_attention': analyzer.analyze_prediction_attention(),
            'head_specialization': analyze_head_specialization(analyzer)
        }
        
        results.append(metrics)
    
    # Aggregate and analyze results
    aggregated = aggregate_results(results)
    visualize_findings(aggregated)
    statistical_tests(aggregated)
    
    return aggregated
```

## Advanced Techniques

### 1. Attention Flow Analysis
Track how attention patterns change across layers:

```python
def trace_attention_flow(analyzer, target_token_pos):
    flow = {}
    for layer_name, attn in analyzer.attention_weights.items():
        # Extract attention from target token across all layers
        attention_pattern = attn[0, :, target_token_pos, :].mean(dim=0)
        flow[layer_name] = attention_pattern
    return flow
```

### 2. Attention Gradient Analysis
Understand which attention patterns are most important:

```python
def attention_gradients(model, input_ids, target_logit):
    # Enable gradients for attention weights
    for layer in model.model.layers:
        layer.self_attn.register_hook(lambda grad: grad)
    
    # Forward pass with gradient computation
    outputs = model(input_ids, output_attentions=True)
    loss = outputs.logits[0, -1, target_logit]
    loss.backward()
    
    # Extract attention gradients
    gradients = {}
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, 'attention_weights'):
            gradients[f'layer_{i}'] = layer.self_attn.attention_weights.grad
    
    return gradients
```

### 3. Causal Intervention Testing
Test causal relationships by modifying attention:

```python
def test_causal_intervention(interventions, base_text, intervention_specs):
    results = {}
    
    # Baseline prediction
    baseline = model.generate(tokenizer(base_text, return_tensors="pt").input_ids)
    
    for layer_idx, head_idx, intervention_type in intervention_specs:
        if intervention_type == "ablate":
            result = interventions.ablate_attention_head(layer_idx, head_idx, ...)
        elif intervention_type == "amplify":
            result = interventions.amplify_attention_head(layer_idx, head_idx, ...)
        
        results[(layer_idx, head_idx, intervention_type)] = result
    
    return results
```

## Tips for Success

1. **Start small**: Begin with a few samples and specific layers before scaling up
2. **Visual inspection**: Always look at attention heatmaps to sanity-check your analysis
3. **Cross-validation**: Test findings across different types of financial statements
4. **Statistical rigor**: Use proper statistical tests when making claims
5. **Reproducibility**: Save configurations and random seeds for reproducible experiments

## Common Pitfalls

1. **Memory issues**: Large models need careful memory management
2. **Attention artifacts**: Some attention patterns may be model artifacts, not meaningful
3. **Over-interpretation**: Statistical significance doesn't always mean practical importance
4. **Selection bias**: Make sure your samples are representative
5. **Layer confusion**: Different model architectures organize attention differently

## Next Steps

After completing basic analysis:

1. **Compare with other models**: How do different architectures process financial data?
2. **Fine-tuning effects**: How does fine-tuning change attention patterns?
3. **Robustness testing**: Do attention patterns hold across different input formats?
4. **Application to other domains**: Can these techniques work for other prediction tasks?

Remember: Mechanistic interpretability is an active research area. Your findings could contribute to our understanding of how language models process structured information! 