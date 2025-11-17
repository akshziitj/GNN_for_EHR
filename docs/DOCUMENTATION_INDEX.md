# EHR Dataset Documentation Index

## Overview
This folder now contains comprehensive documentation explaining the EHR dataset used in the GNN_for_EHR project. The dataset contains 1,000 patient records from MIMIC-III with 133,279 medical codes, used to train a Graph Neural Network to predict patient outcomes.

---

## Documentation Files Created

### 1. **QUICK_REFERENCE.md** ⭐ START HERE
**Length:** 10 minutes | **Complexity:** Beginner-friendly
- The simplest explanation of the entire dataset
- Key numbers to remember
- Common confusions cleared up
- One complete example
- Recommended for anyone new to the project

**Read this first if you:**
- Are new to this project
- Want a quick overview
- Get confused easily
- Need a summary

---

### 2. **EHR_DATASET_EXPLANATION.md**
**Length:** 30 minutes | **Complexity:** Intermediate
- Comprehensive explanation of every concept
- 9 detailed sections covering:
  1. What is EHR?
  2. How is EHR represented?
  3. What does each value mean?
  4. What are medical codes?
  5. Complete example with real interpretation
  6. Why is data sparse?
  7. Class imbalance explanation
  8. Real-world hospital scenario
  9. Summary and key insights

**Read this if you:**
- Want detailed explanations
- Need to understand medical terms
- Want real-world context
- Are writing a report/paper

---

### 3. **DATASET_VISUAL_GUIDE.md**
**Length:** 20 minutes | **Complexity:** Visual learner
- 8 detailed ASCII diagrams
- Visual representations of:
  1. From real hospital data to binary codes
  2. Dataset structure (rows & columns)
  3. What each value means
  4. Three types of medical codes
  5. Why 99.95% zeros
  6. Class imbalance visualization
  7. Complete data flow
  8. Summary table

**Read this if you:**
- Learn better with diagrams
- Want to see examples
- Need to explain to others
- Visual learner

---

### 4. **FAQ_DATASET_EXPLAINED.md**
**Length:** 15 minutes | **Complexity:** Q&A format
- 20 common questions and answers:
  - Q1-5: Basic data interpretation
  - Q6-10: Understanding codes and outcomes
  - Q11-15: Statistical concepts
  - Q16-20: Advanced questions

**Read this if you:**
- Have specific confusions
- Like Q&A format
- Want to skip non-relevant sections
- Prefer quick answers

---

### 5. **sample_dataset.csv**
**Type:** Data file | **Size:** 1.4 KB
- Sample of actual dataset in CSV format
- 20 patient records
- 25 sample medical code columns
- Shows real data structure
- Headers: Patient_ID, Outcome_Label, Code_0 through Code_133277

**Use this to:**
- See actual data values
- Load in Excel/spreadsheet
- Understand row/column structure
- Reference when reading documentation

---

## How to Use These Files

### If you have 5 minutes:
→ Read **QUICK_REFERENCE.md** (high-level overview)

### If you have 15 minutes:
→ Read **QUICK_REFERENCE.md** + **sample_dataset.csv**

### If you have 30 minutes:
→ Read **QUICK_REFERENCE.md** + **DATASET_VISUAL_GUIDE.md**

### If you have 1 hour:
→ Read all of:
1. QUICK_REFERENCE.md
2. EHR_DATASET_EXPLANATION.md
3. DATASET_VISUAL_GUIDE.md

### If you have a specific confusion:
→ Search **FAQ_DATASET_EXPLAINED.md** for your question

### If you're teaching someone:
→ Use **DATASET_VISUAL_GUIDE.md** (with diagrams)

---

## Key Topics Covered

### Data Fundamentals
- What is EHR (Electronic Health Record)?
- Why binary representation?
- What does 1 and 0 mean?
- Why 133,279 codes?

### Medical Codes (3 types)
- Diagnoses (ICD-9): Medical conditions
- Procedures (ICD-9): Medical interventions
- Lab Tests (Quantized): Abnormal values

### Data Characteristics
- 1,000 patients
- 67 codes per patient (average)
- 99.95% sparsity (mostly zeros)
- 1:124 class imbalance
- Real MIMIC-III data

### Outcome Labels
- POSITIVE: High-risk (patient had adverse outcome)
- NEGATIVE: Low-risk (patient survived/no complications)

### Key Concepts
- Sparsity: Why most values are zero
- Class Imbalance: Why only 8 positive cases
- Quantization: How lab values become codes
- Preprocessing: From raw EHR to binary codes

---

## Common Questions Answered

### "What does '1' in the data mean?"
Code is PRESENT for that patient. ✓ Has this diagnosis/procedure/lab abnormality

### "What does '0' in the data mean?"
Code is ABSENT for that patient. ✗ Does not have this diagnosis/procedure/lab value

### "What does POSITIVE outcome mean?"
Patient had an adverse event (died or serious complications).

### "What does NEGATIVE outcome mean?"
Patient did well (survived/no serious complications).

### "Why is 99.95% of data zero?"
Each patient only has ~67 codes out of 133,279 possible. Most are irrelevant to each patient.

### "Why is there class imbalance (1:124)?"
Most patients survive. Only 8 out of 1,000 died (0.8% mortality rate).

### "Is this real data or synthetic?"
REAL data from MIMIC-III hospital database, anonymized and preprocessed.

### "Can I use this for actual patient care?"
NO. Research use only. Would need FDA approval and clinical validation.

---

## File Sizes & Reading Times

| File | Size | Time | Difficulty |
|------|------|------|------------|
| QUICK_REFERENCE.md | 10 KB | 10 min | ⭐ Easy |
| EHR_DATASET_EXPLANATION.md | 12 KB | 30 min | ⭐⭐ Medium |
| DATASET_VISUAL_GUIDE.md | 21 KB | 20 min | ⭐⭐ Medium |
| FAQ_DATASET_EXPLAINED.md | 16 KB | 15 min | ⭐⭐ Medium |
| sample_dataset.csv | 1.4 KB | 5 min | ⭐ Easy |
| **TOTAL** | **60 KB** | **70 min** | **⭐ to ⭐⭐** |

---

## Related Files in Repository

### Original Source Code
- `model.py` - VariationalGNN model architecture
- `dataloader.py` - Data loading utilities
- `train.py` - Training script
- `utils.py` - Training and evaluation functions
- `gnn_ehr.ipynb` - Interactive Jupyter notebook demo

### Original Data
- `data/preprocess_x.pkl` - Feature matrix (1000 × 133,279)
- `data/y_bin.pkl` - Binary labels (outcomes)
- `data/train_idx.pkl` - Training indices
- `data/val_idx.pkl` - Validation indices
- `data/test_idx.pkl` - Test indices

### Preprocessing Scripts
- `data/mimic/preprocess_mimic.py` - MIMIC-III preprocessing
- `data/eicu/preprocess_eicu.py` - eICU preprocessing

---

## Key Statistics at a Glance

```
DATASET DIMENSIONS:
├─ Patients: 1,000
├─ Medical codes: 133,279
├─ Non-zero values: 67,131
├─ Codes per patient (avg): 67.1
└─ Sparsity: 99.95%

OUTCOME DISTRIBUTION:
├─ POSITIVE (high-risk): 8 patients (0.8%)
├─ NEGATIVE (low-risk): 992 patients (99.2%)
└─ Class ratio: 1:124

SPLITS:
├─ Training: 693 samples (69.3%)
├─ Validation: 102 samples (10.2%)
└─ Test: 205 samples (20.5%)

MOST COMMON CODES:
├─ Code 133277: 334 patients (33.4%)
├─ Code 133276: 333 patients (33.3%)
└─ Code 133273: 333 patients (33.3%)
```

---

## How This Dataset Is Used

### In the Project
1. **Data Loading** → `dataloader.py` loads preprocessed data
2. **Graph Construction** → Medical code co-occurrence patterns
3. **Model Training** → VariationalGNN learns from code patterns
4. **Outcome Prediction** → Model predicts POSITIVE/NEGATIVE
5. **Evaluation** → AUPRC metric measures prediction quality

### Why GNN for this data?
- Captures code relationships (which codes co-occur)
- Graph attention learns importance weights
- Variational regularization improves representations
- Better than treating codes independently

---

## Tips for Understanding

1. **Start simple:** Read QUICK_REFERENCE.md first
2. **Use analogies:** Think of codes like symptoms in a disease
3. **Visual learning:** Use DATASET_VISUAL_GUIDE.md diagrams
4. **Real examples:** Study the hospital scenario in EHR_DATASET_EXPLANATION.md
5. **Ask questions:** Check FAQ_DATASET_EXPLAINED.md for your specific confusion
6. **See real data:** Open sample_dataset.csv in spreadsheet program

---

## Learning Path Recommendation

### Level 1: Absolute Beginner (30 minutes)
1. QUICK_REFERENCE.md
2. Open sample_dataset.csv in Excel
3. Re-read first 2 sections

### Level 2: Intermediate (1 hour)
1. QUICK_REFERENCE.md
2. DATASET_VISUAL_GUIDE.md (focus on diagrams)
3. EHR_DATASET_EXPLANATION.md sections 1-5

### Level 3: Advanced (2 hours)
1. Read all 4 documentation files
2. Open Python and explore `data/preprocess_x.pkl`
3. Run the Jupyter notebook `gnn_ehr.ipynb`

### Level 4: Expert (4+ hours)
1. Read all documentation
2. Study preprocessing scripts
3. Run training script with modifications
4. Analyze model outputs and attention weights

---

## Version Information

- **Created:** November 17, 2025
- **Dataset Source:** MIMIC-III (publicly available research dataset)
- **Project:** GNN_for_EHR
- **Documentation Status:** Complete
- **Latest Update:** November 17, 2025

---

## Quick Navigation

**Need help with:**
- **"What is this data?"** → QUICK_REFERENCE.md
- **"What do the numbers mean?"** → EHR_DATASET_EXPLANATION.md sections 3-4
- **"Why is most data zero?"** → DATASET_VISUAL_GUIDE.md diagram 5
- **"How do I interpret outcomes?"** → FAQ_DATASET_EXPLAINED.md Q3-4
- **"Show me an example"** → sample_dataset.csv + EHR_DATASET_EXPLANATION.md section 8
- **"I'm confused about X"** → Search FAQ_DATASET_EXPLAINED.md

---

## Contact / Questions

If anything is still unclear after reading these documents:
1. Check FAQ_DATASET_EXPLAINED.md for similar questions
2. Re-read relevant section in EHR_DATASET_EXPLANATION.md
3. Study diagram in DATASET_VISUAL_GUIDE.md
4. Reference sample_dataset.csv for actual data examples

---

**You are now equipped with comprehensive knowledge of the EHR dataset!** 📚
