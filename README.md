# Federated Learning for Optic Disc and Cup Segmentation in Glaucoma Monitoring

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation & Setup](#installation--setup)
4. [Pipeline Descriptions](#pipeline-descriptions)
5. [Running Experiments](#running-experiments)
6. [Statistical Analysis & Visualization](#statistical-analysis--visualization)
7. [Understanding Results](#understanding-results)
8. [Key Scripts Reference](#key-scripts-reference)
9. [Troubleshooting](#troubleshooting)

---

## 1. Project Overview

This repository implements **Federated Learning (FL) approaches for automated optic disc and optic cup segmentation** from color fundus photographs (CFPs), enabling **privacy-preserving glaucoma assessment and monitoring** across multiple clinical sites.

### Clinical Context

**Glaucoma** is a leading cause of irreversible blindness worldwide, affecting 3.54% of the population aged 40-80 and projected to impact 111.8 million people by 2040. A key indicator of glaucoma severity is the **vertical cup-to-disc ratio (CDR)**, with ratios ≥0.6 suggestive of glaucoma. Accurate automated segmentation of the optic disc and cup enables consistent CDR calculation for diagnosis and monitoring.

### Research Objectives

This study evaluates a **federated learning framework with site-specific fine-tuning** for optic disc and cup segmentation, aiming to:
- Match central model performance while preserving patient data privacy
- Improve cross-site generalizability compared to site-specific local models
- Compare multiple FL strategies: Global Validation, Weighted Global Validation, Onsite Validation, and Fine-Tuned Onsite Validation

### Model Architecture
- **Base Model:** Mask2Former with Swin Transformer backbone
- **Pre-training:** ADE20K dataset (semantic segmentation)
- **Task:** Multi-class segmentation (background, unlabeled, optic disc, optic cup)
- **Input:** Color fundus photographs (512×512, normalized)
- **Optimizer:** AdamW
- **Loss Function:** Multi-class cross-entropy

### Datasets (9 Public Sites)

A total of **5,550 color fundus photographs** from at least **917 patients** across **7 countries** were used:

| Dataset | Total Images | Test Images | Country | Characteristics |
|---------|-------------|-------------|---------|-----------------|
| **Chaksu** | 1,345 | 135 | India | Multi-center research dataset |
| **REFUGE** | 1,200 | 120 | China | Glaucoma challenge dataset |
| **G1020** | 1,020 | 102 | Germany | Benchmark retinal fundus dataset |
| **RIM-ONE DL** | 485 | 49 | Spain | Glaucoma assessment dataset |
| **MESSIDOR** | 460 | 46 | France | Diabetic retinopathy screening |
| **ORIGA** | 650 | 65 | Singapore | Multi-ethnic Asian population |
| **Bin Rushed** | 195 | 20 | Saudi Arabia | RIGA dataset collection |
| **DRISHTI-GS** | 101 | 11 | India | Optic nerve head segmentation |
| **Magrabi** | 94 | 10 | Saudi Arabia | RIGA dataset collection |
| **Total** | **5,550** | **558** | **7 countries** | **Multi-ethnic, varied protocols** |

**Data Split:** Each dataset was divided into training (80%), validation (10%), and testing (10%) subsets. For datasets with multiple expert annotations, the STAPLE (Simultaneous Truth and Performance Level Estimation) method was used to generate consensus segmentation labels.

---

## 2. Repository Structure

```
flglaucomasegfinal/
├── driver/                          # Main execution scripts
│   ├── centraltrain.sh             # Central model training (pooled dataset)
│   ├── persite.sh                  # Local model training (9 site-specific models)
│   ├── pipeline1.sh                # FL: Global Validation
│   ├── pipeline2.sh                # FL: Weighted Global Validation
│   ├── pipeline3.sh                # FL: Onsite Validation
│   ├── pipeline4.sh                # FL: Fine-Tuned Onsite Validation
│   └── analyze_and_plot.sh         # Statistical analysis & visualization
│
├── engine/                          # Core implementation
│   ├── train/
│   │   ├── localtraining.py        # Standard training (central/local models)
│   │   ├── pipeline1.py            # Global Validation implementation
│   │   ├── pipeline2.py            # Weighted Global Validation implementation
│   │   ├── pipeline3.py            # Onsite Validation implementation
│   │   └── pipeline4.py            # Fine-Tuned Onsite Validation implementation
│   ├── inference.py                # Model inference (multiprocess)
│   ├── evaluate.py                 # Per-sample Dice score calculation
│   ├── statistical_analysis.py     # Friedman & Wilcoxon tests
│   ├── plotting.py                 # Comprehensive visualization
│   ├── datasets.py                 # Dataset definitions
│   └── utils.py                    # Utility functions
│
├── data/                            # Raw fundus images and labels (not in repo)
│   └── {dataset}/
│       ├── images/
│       └── labels/
│
├── metadata/                        # CSV files for train/val/test splits
│   ├── combined_train.csv          # All 9 datasets merged (for central model)
│   ├── combined_val.csv
│   ├── combined_test.csv
│   ├── {dataset}_train.csv         # Per-site splits (9 datasets)
│   ├── {dataset}_val.csv
│   └── {dataset}_test.csv
│
├── models/                          # Saved model checkpoints (.pt files)
│   ├── central/                    # Central model
│   ├── persite/                    # Local models (9 site-specific)
│   │   └── {dataset}/
│   ├── pipeline1/                  # Global Validation
│   ├── pipeline2/                  # Weighted Global Validation
│   ├── pipeline3/                  # Onsite Validation
│   └── pipeline4/                  # Fine-Tuned Onsite Validation (9 models)
│       └── {dataset}/
│
├── outputs/                         # Inference predictions (colored masks)
│   └── {model_type}/
│       ├── outputs/                # PNG segmentation masks
│       └── results.csv             # Prediction metadata
│
├── scores/                          # Per-sample Dice scores for all models
│   ├── disc/
│   │   └── {dataset}.csv           # All models evaluated on each dataset
│   └── cup/
│       └── {dataset}.csv
│
├── Statistics/                      # Statistical test results
│   ├── disc/
│   │   ├── {dataset}_disc_pairwise_wilcoxon.csv
│   │   └── {dataset}_disc_friedman.csv (optional)
│   └── cup/
│       ├── {dataset}_cup_pairwise_wilcoxon.csv
│       └── {dataset}_cup_friedman.csv (optional)
│
├── plots/                           # Generated visualizations
│   ├── central_vs_local_by_dataset/
│   ├── local_vs_onsite_finetuned/
│   ├── fl_base_models_comparison/
│   ├── fl_models_vs_local/
│   ├── local_vs_central/
│   ├── onsite_finetuned_comparisons/
│   └── cross_site_performance/
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This documentation
```

**Key Directories:**
- `driver/`: Shell scripts orchestrating training, inference, evaluation, and analysis
- `engine/`: Python implementations of training algorithms and analysis tools
- `metadata/`: CSV files defining train/val/test splits for each dataset
- `models/`: Trained model checkpoints (not version-controlled due to size)
- `scores/`: Evaluation results (Dice scores) for all model-dataset combinations
- `Statistics/`: Statistical comparison results (Wilcoxon tests, p-values)
- `plots/`: Publication-ready visualizations of comparative performance

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10.2
- 80GB RAM for parallel training(Trained on NVIDIA A100-80g)

### Installation Steps

```bash
# 1. Navigate to repository
cd /path/to/flglaucomasegfinal

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. Pipeline Descriptions

### Comparative Evaluation Framework

This repository implements a comparative evaluation of **six approaches** for optic disc and cup segmentation:

| Approach | Name in Results | Script | Description |
|----------|----------------|--------|-------------|
| **Central Model** | `central` | `centraltrain.sh` | Upper bound: trained on pooled multi-site data |
| **Local Models** | `{dataset}_persite` | `persite.sh` | Lower bound: site-specific training only |
| **Global Validation** | `pipeline1` | `pipeline1.sh` | FL with round-based training & global validation |
| **Weighted Global Validation** | `pipeline2` | `pipeline2.sh` | FL with dataset-size weighted FedAvg |
| **Onsite Validation** | `pipeline3` | `pipeline3.sh` | FL with onsite validation & early stopping |
| **Fine-Tuned Onsite Validation** | `{dataset}_fl_finetuned` | `pipeline4.sh` | FL with site-specific fine-tuning |

---

### Baseline Approaches

#### Central Model (`centraltrain.sh`)
**Purpose:** Upper bound performance with maximum data aggregation

**Methodology:**
- A single model trained on all 9 datasets combined
- Uses `combined_train.csv` (5,550 images total)
- Validation on `combined_val.csv` (global validation set)

**Training Configuration:**
- Epochs: 100 (with early stopping, patience=7)
- Batch size: 8
- Learning rate: 2×10⁻⁵
- Optimizer: AdamW
- Loss: Multi-class cross-entropy

**Advantages:**
- Maximum training data → best average performance
- Captures dataset diversity → good generalization

**Disadvantages:**
- Requires data sharing → privacy concerns
- May not capture site-specific characteristics
- Violates HIPAA/GDPR requirements

---

#### Local Models (`persite.sh`)
**Purpose:** Lower bound performance with no data sharing

**Methodology:**
- 9 independent models, each trained exclusively on one site's data
- Each model uses only `{dataset}_train.csv` for its respective site
- No inter-site communication

**Training Configuration:**
- Same hyperparameters as central model
- Separate training run per dataset
- Can execute in parallel or sequential mode

**Usage:**
```bash
# Train single site:
./driver/persite.sh binrushed

# Train all sites sequentially:
./driver/persite.sh all sequential

# Train all sites in parallel (requires multiple GPUs):
./driver/persite.sh all parallel

# Skip training, use existing models:
./driver/persite.sh all --skip-training
```

**Advantages:**
- Complete data privacy (no sharing)
- Optimized for site-specific characteristics
- Simple to implement and deploy

**Disadvantages:**
- Limited training data per site
- Poor cross-site generalization
- Cannot leverage multi-site diversity

---

### Federated Learning Approaches

#### Pipeline 1: Global Validation (`pipeline1.sh`)

![Pipeline 1 Diagram](figures/1i.png)

**Purpose:** FL with round-based training and global validation for early stopping

**Methodology:**
1. **Local Training:** Each site trains for 1 epoch per round
2. **Federated Averaging:** Server aggregates model weights using unweighted FedAvg
   ```
   global_weights = (1/K) × Σ(local_weights_k)
   where K = 9 sites
   ```
3. **Global Validation:** Aggregated model validated on all sites' validation sets
4. **Early Stopping:** Training stops if global validation loss doesn't improve for 7 rounds
5. **Iterate:** Repeat for up to 100 rounds

**Algorithm Flow:**
```
for round in 1..100:
    for each site k:
        local_model_k = train_one_epoch(global_model, local_data_k)
    global_model = federated_average([local_model_1, ..., local_model_9])
    global_val_loss = validate_on_all_sites(global_model)
    if no improvement for 7 rounds:
        break
```

**Characteristics:**
- Simple implementation (1 epoch per site per round)
- No dataset size information shared
- Over-represents smaller datasets (equal weighting)
- Global validation ensures generalizability

---

#### Pipeline 2: Weighted Global Validation (`pipeline2.sh`)

**Purpose:** FL with dataset-size weighted averaging to reflect site contributions

**Methodology:**
1. **Local Training:** Each site trains for 1 epoch per round
2. **Weighted Federated Averaging:** Weights proportional to training set size
   ```
   global_weights = Σ(w_k × local_weights_k)
   where w_k = n_k / N
   n_k = number of training samples at site k
   N = total training samples across all sites
   ```
3. **Global Validation:** Same as Global Validation
4. **Early Stopping:** Same criteria as Global Validation

**Dataset Weights (Example):**
| Dataset | Training Images | Weight (w_k) |
|---------|----------------|--------------|
| Chaksu | 1,076 | 0.242 |
| REFUGE | 960 | 0.216 |
| G1020 | 816 | 0.184 |
| RIM-ONE DL | 388 | 0.087 |
| MESSIDOR | 368 | 0.083 |
| ORIGA | 520 | 0.117 |
| Bin Rushed | 156 | 0.035 |
| DRISHTI-GS | 81 | 0.018 |
| Magrabi | 75 | 0.017 |

**Characteristics:**
- Larger datasets have more influence on global model
- Requires sharing dataset size information (metadata leakage)
- Better represents true data distribution
- May under-represent smaller datasets

**Advantages:** Reflects statistical contribution of each site  
**Disadvantages:** Dominated by large datasets, some privacy trade-off

---

#### Pipeline 3: Onsite Validation (`pipeline3.sh`)

![Pipeline 3 Diagram](figures/1ii.png)

**Purpose:** FL with local early stopping to select best local models before aggregation

**Methodology:**
1. **Extended Local Training:** Each site trains for up to 20 epochs (not just 1)
2. **Onsite Validation:** Each site uses its own validation set for early stopping
3. **Local Early Stopping:** Training stops if local validation loss doesn't improve for 7 epochs
4. **Federated Averaging:** Best local models aggregated using unweighted FedAvg
5. **Global Validation:** Aggregated model validated globally
6. **Iterate:** Repeat for 10 FL rounds

**Algorithm Flow:**
```
for round in 1..10:
    for each site k:
        best_local_model_k = None
        best_val_loss_k = ∞
        for epoch in 1..20:
            local_model_k = train_one_epoch(global_model, local_data_k)
            val_loss_k = validate(local_model_k, local_val_k)
            if val_loss_k < best_val_loss_k:
                best_local_model_k = local_model_k
                best_val_loss_k = val_loss_k
            if no improvement for 7 epochs:
                break
    global_model = federated_average([best_local_model_1, ..., best_local_model_9])
    global_val_loss = validate_on_all_sites(global_model)
```

**Characteristics:**
- More local training per round (20 vs. 1 epoch)
- Local early stopping ensures high-quality local models
- Intuition: Better local models → better aggregated model
- No dataset size information shared

**Advantages:** Strong baseline FL approach  
**Disadvantages:** Longer training time, may overfit to local data

---

#### Pipeline 4: Fine-Tuned Onsite Validation (`pipeline4.sh`)

**Purpose:** Site-specific fine-tuning to personalize global FL model to local data distributions

**Methodology:**
1. **Initialize:** Start with the final Onsite Validation model (Pipeline 3)
2. **Site-Specific Fine-Tuning:** Fine-tune on each site's local data independently
   - Each site fine-tunes for up to 20 epochs
   - Local early stopping (patience=7)
3. **Output:** 9 distinct site-specific models (one per dataset)

**Algorithm Flow:**
```
global_model = trained_onsite_validation_model  # From Pipeline 3

for each site k:
    finetuned_model_k = global_model  # Initialize with global model
    best_model_k = None
    best_val_loss_k = ∞
    
    for epoch in 1..20:
        finetuned_model_k = train_one_epoch(finetuned_model_k, local_data_k)
        val_loss_k = validate(finetuned_model_k, local_val_k)
        
        if val_loss_k < best_val_loss_k:
            best_model_k = finetuned_model_k
            best_val_loss_k = val_loss_k
        
        if no improvement for 7 epochs:
            break
    
    save_model(best_model_k, f"models/pipeline4/{dataset_k}/")
```

**Rationale:**
- Global FL model captures **general features** from all sites
- Local fine-tuning adapts to **site-specific characteristics**
- Maintains robust generalizability while optimizing local performance

**Usage:**
```bash
# Fine-tune for single site:
./driver/pipeline4.sh binrushed

# Fine-tune for all sites sequentially:
./driver/pipeline4.sh all sequential

# Fine-tune for all sites in parallel:
./driver/pipeline4.sh all parallel

# Skip fine-tuning, use existing models:
./driver/pipeline4.sh all --skip-training
```

**Characteristics:**
- **Strategic two-phase training:** Global learning → Local adaptation
- **Personalized models:** Each site gets optimized model
- **Hybrid approach:** Combines FL benefits with local optimization

**Advantages:**
- Matches central model performance (within-site)
- Maintains cross-site generalizability (from FL phase)
- No data sharing required
- Personalizes to site-specific imaging protocols

**Disadvantages:**
- Results in multiple models (not a single global model)
- Requires additional fine-tuning computation
- May complicate deployment if unified model is required

---

### Model Training Details

**Shared Configuration Across All Pipelines:**
- **Model:** Mask2Former with Swin Transformer backbone
- **Pre-training:** ADE20K (Cityscapes for semantic segmentation)
- **Optimizer:** AdamW (weight decay regularization)
- **Learning Rate:** 2×10⁻⁵
- **Batch Size:** 8
- **Loss Function:** Multi-class cross-entropy (background, unlabeled, disc, cup)
- **Early Stopping Patience:** 7 epochs/rounds
- **Hardware:** CUDA-capable GPUs (cuda:0 or cuda:1)
- **Input Size:** 512×512 pixels (normalized)

---

## 5. Running Experiments

### Complete Experimental Workflow

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run all training pipelines
./driver/centraltrain.sh
./driver/persite.sh all sequential
./driver/pipeline1.sh
./driver/pipeline2.sh
./driver/pipeline3.sh
./driver/pipeline4.sh all sequential

# 3. Run statistical analysis and generate plots
./driver/analyze_and_plot.sh
```

### Skipping Training (Use Existing Models)

```bash
# If models already exist, skip training:
./driver/centraltrain.sh --skip-training
./driver/persite.sh all --skip-training
./driver/pipeline1.sh --skip-training
./driver/pipeline2.sh --skip-training
./driver/pipeline3.sh --skip-training
./driver/pipeline4.sh all --skip-training
```

### GPU Configuration

All scripts use specific CUDA devices:
- **Pipeline 1, 2, 4, Central:** `cuda:1`
- **Pipeline 3, Per-site:** `cuda:0`

Modify `--cuda_num` in driver scripts if needed.

### Expected Runtime

| Pipeline | Training Time | Inference Time | Total |
|----------|--------------|----------------|-------|
| Central | ~8-12 hours | ~30 min | ~12.5 hours |
| Per-site (all) | ~6-8 hours (parallel) | ~4.5 hours | ~12.5 hours |
| Pipeline 1 | ~10-15 hours | ~30 min | ~15.5 hours |
| Pipeline 2 | ~8-12 hours | ~30 min | ~12.5 hours |
| Pipeline 3 | ~12-16 hours | ~30 min | ~16.5 hours |
| Pipeline 4 (all) | ~6-8 hours | ~4.5 hours | ~12.5 hours |
| **Total (all)** | **~50-70 hours** | **~10 hours** | **~80 hours** |

---

## 6. Statistical Analysis & Visualization

### Statistical Analysis (`analyze_and_plot.sh`)

**Purpose:** Rigorous statistical comparison of all models using non-parametric tests

**Rationale:**
- **Non-parametric tests** are used because Dice scores may not follow normal distribution
- **Paired tests** appropriate because same test images evaluated across all models
- **Multiple testing correction** essential due to numerous pairwise comparisons

**Tests Performed:**

1. **Friedman Test** (optional, not in default output)
   - Non-parametric alternative to repeated measures ANOVA
   - Tests null hypothesis: all models perform equivalently
   - Used for overall comparison across all models per dataset

2. **Wilcoxon Signed-Rank Test** (primary analysis)
   - Non-parametric paired test for pairwise model comparisons
   - Tests whether median Dice scores differ significantly
   - **Significance threshold:** p < 0.05
   - Applied to all model pairs: 22 models → 231 pairwise comparisons per dataset

3. **Bonferroni Correction**
   - Corrects for multiple hypothesis testing
   - **Alpha level:** α = 0.05
   - Adjusted p-value threshold: p_corrected = p_raw × n_comparisons
   - Conservative but reduces Type I error (false positives)

**Winner Determination:**
A model is declared a **significant winner** if:
- Wilcoxon test p-value < 0.05 (after Bonferroni correction)
- Mean/median Dice score is higher than comparator

**Execution:**
```bash
./driver/analyze_and_plot.sh
```

This script:
1. Runs statistical analysis for disc segmentation → `Statistics/disc/`
2. Runs statistical analysis for cup segmentation → `Statistics/cup/`
3. Generates all visualizations → `plots/`

**Outputs:**
- `Statistics/disc/{dataset}_disc_pairwise_wilcoxon.csv` (9 files)
- `Statistics/cup/{dataset}_cup_pairwise_wilcoxon.csv` (9 files)

**CSV Format:**
```csv
Model_A, Model_B, Mean_A, Mean_B, Median_A, Median_B, 
Std_A, Std_B, Wilcoxon_Stat, Raw_p, Corrected_p, 
Reject_Null, Better_Model, Significant_Winner
```

**Column Definitions:**
- `Model_A`, `Model_B`: Models being compared
- `Mean_A`, `Mean_B`: Mean Dice scores
- `Median_A`, `Median_B`: Median Dice scores
- `Std_A`, `Std_B`: Standard deviations
- `Wilcoxon_Stat`: Test statistic from Wilcoxon test
- `Raw_p`: Uncorrected p-value
- `Corrected_p`: Bonferroni-corrected p-value
- `Reject_Null`: Boolean (True if p < 0.05 after correction)
- `Better_Model`: Model with higher mean Dice score
- `Significant_Winner`: Model name if significant, else `None`

---

### Visualization (`plotting.py`)

**Purpose:** Generate publication-ready figures comparing all approaches

**Visualization Strategy:**
- **Delta plots:** Show performance differences (not absolute scores)
- **Statistical annotations:** Indicate significant wins/losses
- **Color coding:** Green (improvement), Red (degradation), Gray (no significant difference)
- **Consistent labeling:** Use "local model" and "onsite finetuned" terminology (not "sk" or "fl_finetuned")

**Generated Plots:**

#### 1. Onsite Fine-tuned Comparisons (`onsite_finetuned_comparisons/`)
**Files:** 
- `onsite_finetuned_vs_all_models_disc.png`
- `onsite_finetuned_vs_all_models_cup.png`

**Description:** 6-subplot comprehensive comparison
- Fine-Tuned Onsite Validation vs. Central Model
- Fine-Tuned Onsite Validation vs. Local Models
- Fine-Tuned Onsite Validation vs. Global Validation
- Fine-Tuned Onsite Validation vs. Weighted Global Validation
- Fine-Tuned Onsite Validation vs. Onsite Validation
- Summary statistics

**Interpretation:** Shows whether site-specific fine-tuning improves upon all other approaches

---

#### 2. Central vs Local by Dataset (`central_vs_local_by_dataset/`)
**Files:**
- `central_vs_local_models_by_dataset_disc.png`
- `central_vs_local_models_by_dataset_cup.png`

**Description:** 3×3 grid (one subplot per evaluation dataset)
- X-axis: Training datasets (which local model)
- Y-axis: Delta Dice (Central - Local)
- Each subplot: Central model vs. all 9 local models on one test set

**Interpretation:** Reveals whether central model outperforms local models in cross-site evaluation

---

#### 3. Local vs Onsite Fine-tuned (`local_vs_onsite_finetuned/`)
**Files:**
- `local_models_vs_onsite_finetuned_comprehensive_disc.png`
- `local_models_vs_onsite_finetuned_comprehensive_cup.png`

**Description:** Comprehensive 3×3 grid per task
- Compares Fine-Tuned Onsite Validation models vs. Local models
- Shows delta performance for within-site and cross-site evaluations

**Interpretation:** Demonstrates cross-site generalizability improvements from FL fine-tuning

---

#### 4. FL Baseline Comparisons (`fl_base_models_comparison/`)
**Files:**
- `fl_base_models_comparison_disc.png`
- `fl_base_models_comparison_cup.png`

**Description:** Standard FL approaches (Pipelines 1, 2, 3) vs. baselines
- Compares Global Validation, Weighted Global Validation, Onsite Validation
- Against Central and Local models

**Interpretation:** Shows relative performance of different FL strategies

---

#### 5. FL vs All Local Models (`fl_models_vs_local/`)
**Files:**
- Multiple files per FL pipeline and task

**Description:** Separate plots for each FL strategy
- Each FL model vs. all 9 local models
- Evaluated on all 9 test sets

**Interpretation:** Cross-site generalization of FL models

---

#### 6. Local vs Central Summary (`local_vs_central/`)
**Files:**
- `local_models_vs_central_disc.png`
- `local_models_vs_central_cup.png`

**Description:** Single summary plot
- Central model vs. best local model per dataset

**Interpretation:** Quick overview of central vs. local performance

---

#### 7. Cross-Site Performance Heatmaps (`cross_site_performance/`)
**Files:**
- Heatmaps showing out-of-distribution performance

**Description:** Matrix visualization
- Rows: Training site
- Columns: Evaluation site
- Colors: Significant wins/losses

**Interpretation:** Identifies which training data generalizes best to external sites

---

## 7. Understanding Results

### Evaluation Metric: Sørensen–Dice Coefficient

**Definition:**
```
Dice Score = 2 × |A ∩ B| / (|A| + |B|)
```
where:
- A = Predicted segmentation mask
- B = Ground truth segmentation mask
- Range: [0, 1], where 1 = perfect overlap

**Advantages for Segmentation Evaluation:**
- Robust to class imbalance (optic disc/cup are small regions in fundus images)
- Penalizes false positives and false negatives equally
- Widely used in medical image segmentation literature

---

### Score Files (`scores/`)

**Format:** `scores/{task}/{dataset}.csv`

```csv
image_name,model_name,dice_score
cropped_br3_image22prime,central,0.9829
cropped_br3_image22prime,binrushed_persite,0.8938
cropped_br3_image22prime,pipeline1,0.9645
...
```

**Key Points:**
- **All 22 models** evaluated on same test images per dataset
- **558 total test images** across 9 datasets (varies per dataset: 10-135 images)
- **Dice scores** range [0, 1], higher is better
- Each dataset CSV contains 22 rows per image (one per model)

**Model Count Breakdown:**
- 1 Central model
- 9 Local models (one per site)
- 3 FL models (Global Validation, Weighted Global Validation, Onsite Validation)
- 9 Fine-Tuned Onsite Validation models (one per site)
- **Total:** 22 models

---

### Statistical Significance Interpretation

**p-value Thresholds (after Bonferroni correction):**
- **p < 0.05:** Statistically significant difference → Reject null hypothesis
- **p ≥ 0.05:** No significant difference → Fail to reject null hypothesis

**Significant_Winner Column in Results CSV:**
- **Model name** (e.g., `central`): That model is significantly better (p < 0.05)
- **`None`**: No significant difference between models (p ≥ 0.05)

**Interpreting Delta Plots:**
- **Positive delta:** First model outperforms second model
- **Negative delta:** Second model outperforms first model
- **Zero/near-zero delta:** Models perform equivalently

**Statistical Power Considerations:**
- Smaller test sets (e.g., Magrabi: 10, DRISHTI-GS: 11) have lower statistical power
- Larger test sets (e.g., Chaksu: 135, REFUGE: 120) enable more robust conclusions
- Bonferroni correction is conservative → reduces false positives but may increase false negatives

---

### Expected Performance Patterns

Based on the manuscript findings, typical results include:

#### 1. **Fine-Tuned Onsite Validation vs. Central Model**
- **Cup:** No significant difference across all sites (9/9 = 100%)
- **Disc:** No significant difference in most sites (7/9 ≈ 78%)
- **Interpretation:** FL fine-tuning matches central model's within-site performance

#### 2. **Fine-Tuned Onsite Validation vs. Local Models**
- **Cup:** No significant difference across all sites (9/9 = 100%)
- **Disc:** No significant difference in most sites (5/9 ≈ 56%), with FL fine-tuning winning in remaining 4/9 (≈ 44%)
- **Interpretation:** FL fine-tuning maintains local model's within-site performance while improving generalizability

#### 3. **Fine-Tuned Onsite Validation vs. Standard FL Pipelines**
- **Cup:** Significant improvements in 7/27 comparisons (≈ 26%)
  - Global Validation: 2/9 wins
  - Weighted Global Validation: 3/9 wins
  - Onsite Validation: 2/9 wins
- **Disc:** Significant improvements in 14/27 comparisons (≈ 52%)
  - Global Validation: 5/9 wins
  - Weighted Global Validation: 4/9 wins
  - Onsite Validation: 5/9 wins
- **Interpretation:** Site-specific fine-tuning notably improves disc segmentation performance

#### 4. **Cross-Site Generalizability (External Site Comparisons)**
- **Cup:** Fine-Tuned Onsite Validation shows significant wins in 25.0% (18/72) of external evaluations, with only 2.8% losses
- **Disc:** Fine-Tuned Onsite Validation shows significant wins in 54.2% (39/72) of external evaluations, with **zero losses**
- **Interpretation:** FL fine-tuning dramatically improves cross-site generalization, especially for disc segmentation

---

### Performance Characteristics by Task

#### **Optic Disc Segmentation:**
- Generally **easier task** than cup segmentation
- Higher absolute Dice scores (typically 0.90-0.98)
- More consistent across sites
- FL fine-tuning shows substantial improvements over local models
- Lower inter-rater variability in annotations

#### **Optic Cup Segmentation:**
- **More challenging task** due to:
  - Vessel density obscuring cup-rim boundary
  - Gradual intensity transitions
  - Greater inter-rater variability in annotations
  - Glaucomatous changes in cup morphology
- Lower absolute Dice scores (typically 0.75-0.92)
- More susceptible to inter-site heterogeneity
- Local models already near-optimal → FL fine-tuning shows smaller gains
- Benefits less from cross-site federated training

---

### Dataset-Specific Considerations

#### **High-Performance Datasets:**
- **Chaksu** (135 test images): Large, diverse dataset with consistent annotations
- **REFUGE** (120 test images): Challenge dataset with standardized protocols
- **G1020** (102 test images): Well-annotated benchmark dataset

#### **Challenging Datasets:**
- **DRISHTI-GS** (11 test images): Very small test set → high variance, low statistical power
- **Magrabi** (10 test images): Smallest test set → difficult to establish significance
- **Bin Rushed** (20 test images): Small sample size limits robust conclusions

#### **Local Model Performance Anomalies:**
- `drishti_persite` and `magrabi_persite` often underperform due to limited training data
- These sites benefit most from FL approaches
- Cross-site evaluation reveals severe overfitting in small-dataset local models

---

## 8. Key Scripts Reference

### Training Scripts

**`engine/train/localtraining.py`**
```bash
python3 engine/train/localtraining.py \
    --train_csv metadata/dataset_train.csv \
    --val_csv metadata/dataset_val.csv \
    --csv_img_path_col image_path \
    --csv_label_path_col label_path \
    --output_directory /tmp/output \
    --dataset_mean "0.768 0.476 0.290" \
    --dataset_std "0.220 0.198 0.166" \
    --lr 0.00002 \
    --batch_size 8 \
    --num_epochs 100 \
    --patience 7 \
    --num_workers 16
```

### Inference Script

**`engine/inference.py`**
```bash
python3 engine/inference.py \
    --model_path models/central/model_epoch50_ckpt.pt \
    --input_csv metadata/combined_test.csv \
    --csv_path_col_name image_path \
    --output_root_dir outputs/central \
    --num_processes 1 \
    --cuda_num 0
```

### Evaluation Script

**`engine/evaluate.py`**
```bash
# For disc segmentation:
python3 engine/evaluate.py \
    --prediction_folder outputs/central/outputs \
    --label_folder data/ \
    --csv_path outputs/central/results.csv \
    --eval_disc \
    --cuda_num 0 \
    --output_csv scores/central/per_sample_disc_scores.csv \
    --model_name "central" \
    --statistical_output_dir scores

# For cup segmentation:
python3 engine/evaluate.py \
    --prediction_folder outputs/central/outputs \
    --label_folder data/ \
    --csv_path outputs/central/results.csv \
    --cuda_num 0 \
    --output_csv scores/central/per_sample_cup_scores.csv \
    --model_name "central" \
    --statistical_output_dir scores
```

### Statistical Analysis

**`engine/statistical_analysis.py`**
```bash
# For disc:
python3 engine/statistical_analysis.py \
    --eval_type disc \
    --input_dir scores/disc \
    --output_dir Statistics/disc \
    --skip-summaries

# For cup:
python3 engine/statistical_analysis.py \
    --eval_type cup \
    --input_dir scores/cup \
    --output_dir Statistics/cup \
    --skip-summaries
```

### Plotting

**`engine/plotting.py`**
```bash
python3 engine/plotting.py \
    --disc_results_dir Statistics/disc \
    --cup_results_dir Statistics/cup \
    --output_dir plots
```

---

## 9. Troubleshooting

### Common Issues

#### 1. **Missing Dependencies**
```bash
# Error: ModuleNotFoundError: No module named 'statsmodels'
pip install statsmodels==0.14.4

# Error: CUDA out of memory
# Solution: Reduce batch_size in training scripts
```

#### 2. **Checkpoint Not Found**
```bash
# Error: No checkpoint found in models/pipeline1/
# Solution: Check training completed successfully or use --skip-training
ls models/pipeline1/  # Should see .pth or .pt files
```

#### 3. **Dimension Mismatch in evaluate.py**
```python
# Bug in line 42 of evaluate.py (KNOWN ISSUE):
# Wrong: class_labels.view(image.shape[1], image.shape[0])
# Fix:   class_labels.view(image.shape[0], image.shape[1])
```

#### 4. **Different Results on Re-run**
**Possible causes:**
- Random seed not set
- Different CUDA device
- Model checkpoint from different epoch
- Dataset normalization statistics changed

#### 5. **Plots Show "No Data Available"**
```bash
# Check if statistical analysis completed:
ls Statistics/disc/*_pairwise_wilcoxon.csv
ls Statistics/cup/*_pairwise_wilcoxon.csv

# If files missing, re-run:
./driver/analyze_and_plot.sh
```

### Performance Optimization

**Training Faster:**
```bash
# Use parallel per-site training:
./driver/persite.sh all parallel

# Reduce num_epochs for testing:
# Edit driver scripts: --num_epochs 10 (instead of 100)

# Use fewer workers if RAM-limited:
# Edit driver scripts: --num_workers 4 (instead of 16)
```

**Inference Faster:**
```bash
# Increase parallel processes (if GPU memory allows):
# Edit driver scripts: --num_processes 4 (instead of 1)
```

### File Size Management

**Large directories:**
- `/tmp/flglaucomaseg_train/` - Training outputs (can be deleted after training)
- `outputs/` - Prediction images (650MB per model)
- `models/` - Model checkpoints (1GB per model)

**Space-saving options:**
```bash
# Remove temporary training outputs:
rm -rf /tmp/flglaucomaseg_train/

# Remove prediction images (keep CSVs):
find outputs -name "*.png" -delete

# Keep only best checkpoint per model:
# (Manually remove model_epoch*_ckpt.pt files except best)
```

### Debug Mode

**Enable verbose output:**
```bash
# Add to any Python script:
import logging
logging.basicConfig(level=logging.DEBUG)

# Or run with Python debug flag:
python3 -u engine/train/localtraining.py [args...]
```

---

## Key Findings & Conclusions

Based on the comparative evaluation of 22 models across 5,550 fundus photographs from 9 datasets:

### Main Results

1. **Fine-Tuned Onsite Validation Matches Central Model Performance**
   - Achieved performance comparable to the central model (upper bound) for cup segmentation across all sites (9/9)
   - Matched central model performance for disc segmentation in most sites (7/9)
   - **No data sharing required**, preserving patient privacy and HIPAA/GDPR compliance

2. **Improved Cross-Site Generalizability Over Local Models**
   - **Optic Disc:** Significant wins in 54.2% (39/72) of external site evaluations with **zero losses**
   - **Optic Cup:** Significant wins in 25.0% (18/72) of external site evaluations with minimal losses (2.8%)
   - Maintained equivalent within-site performance to local models while dramatically improving generalization

3. **Outperformed Standard FL Approaches**
   - **Optic Disc:** Significant improvements in 52% of comparisons against Global Validation, Weighted Global Validation, and Onsite Validation
   - **Optic Cup:** Significant improvements in 26% of comparisons
   - Site-specific fine-tuning effectively personalizes generalized FL models to local data distributions

### Clinical Implications

✅ **Privacy-Preserving AI Deployment:** Enables multi-institutional collaboration without data sharing  
✅ **Robust Performance:** Achieves central-level accuracy with improved cross-site generalizability  
✅ **Adaptable to Local Protocols:** Site-specific fine-tuning accommodates heterogeneous imaging conditions  
✅ **Scalable Solution:** Supports standardized glaucoma assessment across diverse clinical settings  
✅ **Reproducible Methodology:** Complete open-source pipeline for FL in medical imaging

### Significance

This work demonstrates that **federated learning with site-specific fine-tuning** can effectively personalize generalized models to local data distributions, combining the strengths of:
- **Central training** (maximum data, best generalization)
- **Local training** (privacy preservation, site-specific optimization)
- **Federated learning** (collaborative learning without data sharing)

The approach enables **privacy-preserving, high-performance AI deployment** across heterogeneous clinical environments, supporting reproducible and generalizable glaucoma detection and monitoring at scale.

---

## Summary

This repository provides a **complete federated learning pipeline** for optic disc and cup segmentation with:

✅ **6 training approaches** (central, local, 4 FL strategies)  
✅ **Multi-site evaluation** (5,550 images from 9 datasets across 7 countries)  
✅ **Rigorous statistical comparison** (Wilcoxon signed-rank tests + Bonferroni correction)  
✅ **Comprehensive visualization** (7 types of publication-ready plots)  
✅ **Cross-site generalizability testing** (all model-dataset combinations)  
✅ **Privacy-preserving methodology** (no data sharing, HIPAA/GDPR compliant)  
✅ **Reproducible workflow** (automated shell scripts + detailed documentation)

### For Typical Use:

**1. Setup (one-time):**
```bash
cd /path/to/flglaucomasegfinal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Run All Training Pipelines (60-80 hours GPU time):**
```bash
./driver/centraltrain.sh
./driver/persite.sh all sequential
./driver/pipeline1.sh
./driver/pipeline2.sh
./driver/pipeline3.sh
./driver/pipeline4.sh all sequential
```

**3. Perform Statistical Analysis & Generate Visualizations:**
```bash
./driver/analyze_and_plot.sh
```

**4. Review Results:**
- **Plots:** `plots/` directory (7 subdirectories with PNG figures)
- **Statistical Tests:** `Statistics/disc/` and `Statistics/cup/` (CSV files)
- **Raw Scores:** `scores/disc/` and `scores/cup/` (per-sample Dice scores)

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{shrivastava2025federated,
  title={A Federated Learning-based Optic Disc and Cup Segmentation Model for Glaucoma Monitoring in Color Fundus Photographs},
  author={Shrivastava, Sudhanshu and Thakuria, Upasana and Kinder, Scott and Nebbia, Giacomo and Zebardast, Nazlee and Baxter, Sally L. and Xu, Benjamin and Alryalat, Saif Aldeen and Kahook, Malik and Kalpathy-Cramer, Jayashree and Singh, Praveer},
  year={2025},
  institution={University of Colorado Anschutz Medical Campus}
}
```

**Corresponding Author:**  
Praveer Singh, PhD  
Department of Ophthalmology  
University of Colorado Anschutz Medical Campus  
1675 Aurora Ct, Aurora, CO 80045  
Email: Praveer.Singh@cuanschutz.edu

---

## Acknowledgments

This research was conducted at the University of Colorado Anschutz Medical Campus, Department of Ophthalmology, in collaboration with:
- Massachusetts Eye and Ear Infirmary, Harvard Medical School
- Viterbi Family Department of Ophthalmology and Shiley Eye Institute, UC San Diego
- Division of Biomedical Informatics, UC San Diego
- Roski Eye Institute, Keck School of Medicine, USC

**Public Datasets Used:**
- Bin Rushed, Magrabi, MESSIDOR (RIGA collection)
- Chákṣu database (Manipal Academy of Higher Education)
- DRISHTI-GS (Indian Institute of Technology)
- G1020 (German benchmark dataset)
- ORIGA (Singapore Eye Research Institute)
- REFUGE (glaucoma challenge dataset)
- RIM-ONE DL (Spanish multi-center study)

We acknowledge the contributors and institutions that made these datasets publicly available for research.

---

## License

This code is provided for research purposes. Please refer to individual dataset licenses for data usage restrictions.

---

*Documentation Version: 2.0 | Last Updated: 2025-11-14*

