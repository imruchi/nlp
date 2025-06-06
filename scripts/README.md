# Mechanistic Interpretability Analysis Scripts

This directory contains scripts for performing mechanistic interpretability analysis on the Qwen2-7B financial prediction model.

## Files

### `mechanistic_interpretability.py`
Main analysis script that performs attention pattern analysis to understand how the model makes correct vs incorrect predictions.

**Features:**
- Extracts attention matrices from all layers and heads
- Analyzes attention to financial vocabulary terms
- Compares patterns between correct and incorrect predictions
- Detects head specialization patterns
- Creates comprehensive visualizations
- Generates statistical comparisons

**Usage:**
```bash
python mechanistic_interpretability.py \
    --config ../nlp/config/interpretability_config.yaml \
    --predictions ../nlp/outputs/predictions_checkpoint_1800.json \
    --max_samples 30
```

### `run_interpretability.slurm`
SLURM batch script for running the analysis on Northwestern Quest cluster.

**Features:**
- Optimized for A100 GPU
- Automatically sets up Python environment
- Manages CUDA and cache directories
- Comprehensive logging and error handling

**Usage:**
```bash
sbatch run_interpretability.slurm
```

## Configuration

The analysis is controlled by `nlp/config/interpretability_config.yaml`:

- **Model Configuration**: Model name, device settings, memory usage
- **Generation Settings**: Token limits, temperature, attention capture
- **Analysis Parameters**: Layers to analyze, vocabulary categories, statistical tests
- **Visualization Settings**: Color schemes, figure sizes, output formats

## Analysis Output

The script generates:

1. **Attention Matrices** (`results/interpretability/attention_matrices/`)
   - Raw attention weights for each sample and layer

2. **Visualizations** (`results/interpretability/visualizations/`)
   - Attention heatmaps comparing correct vs incorrect predictions
   - Financial vocabulary attention comparison charts
   - Head specialization pattern plots
   - Layer-wise attention progression graphs

3. **Analysis Results** (`results/interpretability/`)
   - `attention_analysis_results.json` - Complete attention data
   - `comparison_analysis_results.json` - Statistical comparisons
   - `analysis_summary_report.txt` - Human-readable summary

## Key Analysis Questions

The analysis addresses these research questions:

1. **Financial Attention Patterns**: Which financial terms does the model focus on in correct vs incorrect predictions?

2. **Head Specialization**: Do certain attention heads specialize in processing specific types of financial information?

3. **Layer-wise Processing**: How do attention patterns evolve across layers for correct vs incorrect predictions?

4. **Prediction Mechanisms**: What attention patterns precede the model's final increase/decrease prediction?

5. **Error Analysis**: What attention differences characterize failed predictions?

## Expected Insights

Based on mechanistic interpretability research, we expect to find:

- **Specialized Financial Heads**: Certain attention heads focus primarily on financial terms (revenue, profit, etc.)
- **Prediction Heads**: Heads that strongly attend to prediction-relevant tokens
- **Context Integration**: Later layers showing more complex financial reasoning patterns
- **Error Signatures**: Distinct attention patterns in incorrect predictions (e.g., over-attention to irrelevant terms)

## Performance Considerations

- **Memory Usage**: Analysis requires ~32-64GB RAM for full model attention extraction
- **GPU Requirements**: A100 GPU recommended for efficient processing
- **Sample Size**: 20-50 samples provide good statistical power while maintaining reasonable runtime
- **Cache Management**: Transformers cache configured to avoid repeated model downloads

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce `max_samples` parameter or use smaller model
2. **Missing Dependencies**: Install `requirements_interpretability.txt`
3. **Path Errors**: Ensure all paths in config files are relative to project root
4. **Permission Errors**: Check write permissions in output directories

### Debug Mode:
Add `--debug` flag to enable verbose logging and smaller sample sizes for testing.

## Next Steps

After running the analysis:

1. Review the summary report for key findings
2. Examine attention heatmaps for visual patterns
3. Analyze statistical significance of differences
4. Consider intervention experiments based on findings
5. Document insights for paper/presentation