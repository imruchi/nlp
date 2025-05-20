# NLP Project Proposal

## MSAI-337 Final Project Proposal:  
**Mechanistic Interpretability for Financial Forecasting with LLMs**  

**Team:** Ruchi B, Joanne Mathew, Narasimha Karthik J

### What task will you address, and why is it interesting?

Our project will explore the mechanistic interpretability of large language models in financial statement analysis, specifically predicting the direction of earnings changes (increase, stay the same, or decrease) from balance sheet inputs. This task is interesting because it allows us to investigate how LLMs process numerical financial data—a domain they weren't explicitly trained for—and provides insights into which components of financial statements most significantly influence predictions. By understanding and manipulating these mechanisms, we can potentially improve model accuracy and reliability in financial forecasting applications while making the reasoning process more transparent.

### How will you acquire your data?

We will use the standardized financial statement dataset described in the Kim et al. (2024) paper _"Financial Statement Analysis with Large Language Models."_ This dataset contains anonymized balance sheets and income statements from 150,678 firm-year observations (15,401 distinct firms) from Compustat spanning 1968–2021. Each financial statement is standardized following Compustat's balancing model, with company identifiers and years replaced with relative time markers (t, t-1, t-2). We will sample 5,000 company observations from this dataset, ensuring a balanced distribution across our three outcome categories (earnings increase, stay the same, decrease) to avoid bias in our analysis and evaluation.

### Which features/attributes will you use for your task?

#### 1. Input Features (Balance Sheet Items):
- Cash and Short-Term Investments  
- Receivables  
- Inventories  
- Property, Plant, and Equipment (Net)  
- Total Assets  
- Current Liabilities  
- Long-term Debt  
- Stockholders' Equity  
- Other relevant balance sheet metrics  

#### 2. Target Variable:
- Earnings direction changes (3-class classification: increase, stay the same, decrease)

#### 3. Mechanistic Interpretability Features:
- Attention weights from self-attention heads across all layers  
- Activation patterns in feed-forward networks  
- Token-level representations at different layers  
- Intermediate logit values  
- Causal intervention results on specific model components  

### What will your initial approach be?

Our approach consists of three main phases:

## Phase 1: Baseline Model Implementation and Evaluation

### 1. Data Preparation:
- Format 5,000 financial statements with standardized structure  
- Develop chain-of-thought (CoT) prompts that instruct the model to analyze financial statements  
- Create a balanced dataset with equal representation of the three earnings change categories  

### 2. Model Setup:
- Implement inference pipeline using Qwen 7B model  
- Set up instrumentation to capture internal model states and activations  
- Configure the model to generate three-class predictions (increase, stay the same, decrease)  

### 3. Baseline Evaluation:
- Run inference on the entire dataset with standard prompting  
- Calculate F1 score (macro-averaged across the three classes) as our primary quantitative metric  
- Establish this as our benchmark performance without interpretability-based interventions  

## Phase 2: Mechanistic Interpretability Analysis

### 1. Attention Analysis:
- Extract attention patterns from all layers and heads during inference  
- Identify which attention heads focus on financial metrics most relevant to earnings prediction  
- Visualize attention maps to highlight relationships between financial statement items  
- Quantify attention distribution across different balance sheet components  

### 2. Activation Extraction and Analysis:
- Extract neuron activations from each layer's feed-forward networks  
- Identify neurons that activate strongly in response to specific financial concepts  
- Create feature importance rankings based on activation magnitudes  
- Map activations to corresponding financial metrics  

### 3. Causal Interventions:
- Implement controlled perturbations of specific activations  
- Measure how changes to internal representations affect the final prediction  
- Identify critical pathways in the model that process financial information  
- Quantify the causal influence of specific model components on predictions  

## Phase 3: Model Steering and Performance Enhancement

### 1. Activation Steering:
- Based on Phase 2 findings, develop steering vectors for specific financial concepts  
- Implement direct intervention methods to modify model activations during inference  
- Test various steering intensities to find optimal intervention points  
- Measure how steering affects prediction outcomes  

### 2. Prompt Engineering Based on Interpretability Insights:
- Develop enhanced prompts that target the most influential attention patterns  
- Guide the model to focus on the financial metrics identified as most predictive  
- Implement dynamic prompting techniques informed by our mechanistic findings  

### 3. Evaluation of Enhanced Model:
- Measure F1 score of the steered model on the same dataset  
- Compare performance against the baseline (unmodified) model  
- Analyze error patterns to identify remaining limitations  
- Assess the consistency of model explanations after steering  

## Technical Implementation Details

### 1. Model Instrumentation:
- We will use the transformers library with custom hooks to access internal model states  
- Implement `forward_hooks` on attention layers and feed-forward networks to extract activations  
- Create a custom attention extraction module that captures attention weights across all heads  
- Set up causal tracing infrastructure to perform targeted interventions  

### 2. Visualization Pipeline:
- Develop heat map visualizations for attention patterns over financial statement items  
- Create network diagrams showing information flow through the model  
- Generate attribution maps highlighting which input tokens most influence predictions  
- Build interactive dashboards to explore model internals for specific financial cases  

### 3. Intervention Framework:
- Implement an activation modification module that can alter specific neuron activations  
- Create a controlled experimentation pipeline to measure effects of interventions  
- Develop a steering vector library for common financial concepts  
- Build a system to automatically identify optimal intervention points  

### 4. Evaluation Framework:
- Calculate macro-averaged F1 score across the three prediction classes  
- Implement confusion matrices to analyze error patterns  
- Create ablation studies to quantify the impact of each intervention technique  
- Develop counterfactual tests to validate causal relationships in the model  

## Novel Contribution

Our novel contribution is the development of a specialized mechanistic interpretability framework for multi-class financial prediction, which goes beyond existing work that focuses primarily on binary classification or linguistic tasks. By implementing targeted interventions informed by our interpretability analysis, we aim to not only understand how LLMs process financial data but also demonstrate how this understanding can be leveraged to enhance model performance in a controlled and transparent manner.
