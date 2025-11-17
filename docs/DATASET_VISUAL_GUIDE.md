# Visual Guide to Understanding EHR Dataset

## DIAGRAM 1: From Real Hospital Data to Binary Codes

```
REAL HOSPITAL (Text-based EHR):
================================
Patient #0
├─ Admitted: ICU
├─ Diagnoses:
│  ├─ Acute kidney injury
│  ├─ Sepsis
│  └─ Anemia
├─ Procedures:
│  ├─ Mechanical ventilation
│  ├─ Central line placement
│  └─ Blood transfusion
├─ Lab Results:
│  ├─ WBC: 18,000 (HIGH)
│  ├─ Hemoglobin: 7.5 (LOW)
│  ├─ Creatinine: 3.2 (HIGH)
│  └─ O2 saturation: 88% (LOW)
└─ Outcome: Survived (NEGATIVE)


                        PREPROCESSING
                              ↓


STANDARDIZED CODES (Using ICD-9):
===================================
Code_584_9 → Acute kidney injury
Code_995_92 → Sepsis
Code_285_9 → Anemia
Code_9390 → Mechanical ventilation
Code_3893 → Central line
Code_9904 → Blood transfusion
Code_Lab_WBC_HIGH → High WBC
Code_Lab_Hgb_LOW → Low hemoglobin
Code_Lab_Creat_HIGH → High creatinine
Code_Lab_O2_LOW → Low oxygen


                    BINARY REPRESENTATION
                             ↓


PATIENT VECTOR (Sparse):
===================================
Position:   0   1   2  ...  584_9  ...  9390  ...  Lab_WBC_HIGH  ...  999999
Value:      0   0   0  ...    1    ...    1    ...       1       ...    0
            ^   ^   ^         ^           ^             ^                ^
         absent absent absent present    present       present         absent
                    
Final Format in Dataset:
[0, 0, 0, ..., 1, ..., 1, ..., 1, ..., 1, ..., 1, 0, ..., 0]
                ^                           ^     ^
             AKI present              O2 low present

Outcome Label: NEGATIVE (patient survived)
```

---

## DIAGRAM 2: Dataset Structure (Rows & Columns)

```
           COLUMNS (Medical Codes: 0 to 133,278)
           ↓                                      ↓
         Code_0  Code_1  Code_2  ... Code_133278
ROWS    ─────────────────────────────────────────────
        ┌───────┬───────┬───────┬─────┬──────────┐
Pat_0   │  0    │  0    │  1    │ ... │    1     │  Outcome: NEGATIVE
        ├───────┼───────┼───────┼─────┼──────────┤
Pat_1   │  0    │  1    │  0    │ ... │    1     │  Outcome: NEGATIVE
        ├───────┼───────┼───────┼─────┼──────────┤
Pat_2   │  1    │  0    │  1    │ ... │    0     │  Outcome: NEGATIVE
        ├───────┼───────┼───────┼─────┼──────────┤
Pat_3   │  0    │  0    │  0    │ ... │    1     │  Outcome: NEGATIVE
        ├───────┼───────┼───────┼─────┼──────────┤
        │  ...  │  ...  │  ...  │ ... │   ...    │
Pat_12  │  1    │  1    │  0    │ ... │    1     │  Outcome: POSITIVE ⚠️
        ├───────┼───────┼───────┼─────┼──────────┤
        │  ...  │  ...  │  ...  │ ... │   ...    │
Pat_999 │  0    │  0    │  1    │ ... │    0     │  Outcome: NEGATIVE
        └───────┴───────┴───────┴─────┴──────────┘

Total: 1,000 rows × 133,279 columns = 133,279,000 cells
Only 67,131 cells have value "1" (99.95% are "0")
```

---

## DIAGRAM 3: What Each Value Means

```
CELL VALUE INTERPRETATION:
===========================

     CODE PRESENT (1)
            ↓
    ┌──────────────┐
    │      1       │  → Patient HAS this medical code
    └──────────────┘     (Diagnosed with it, underwent it, or had abnormal lab)
            
     CODE ABSENT (0)
            ↓
    ┌──────────────┐
    │      0       │  → Patient DOES NOT HAVE this medical code
    └──────────────┘     (Not diagnosed, didn't have procedure, lab normal)


EXAMPLE INTERPRETATION:
========================

Patient 0 Row: [0, 0, 1, 0, 1, 0, ..., 0, 1, ..., 0]
               ^  ^  ^  ^  ^            ^  ^     ^
               |  |  |  |  |            |  |     |
             Code0 Code2 Code4 ...   Code_N-1 Code_N
             (NO) (YES) (NO) (YES)     (NO)   (YES)

What this patient HAS:
  ✓ Code 2: YES (Patient has this condition/procedure/lab abnormality)
  ✓ Code 4: YES (Patient has this condition/procedure/lab abnormality)
  ✓ Code N-1: YES (Patient has this condition/procedure/lab abnormality)

What this patient DOES NOT HAVE:
  ✗ Code 0: NO
  ✗ Code 1: NO
  ✗ Code 3: NO
  ✗ ... (133,276 more codes absent)
```

---

## DIAGRAM 4: Three Types of Medical Codes

```
THE 133,279 MEDICAL CODES REPRESENT:
=====================================

┌─────────────────────────────────────────────────────────────┐
│ 1. DIAGNOSES (ICD-9 Codes)                                │
│    ────────────────────────────────────                    │
│    Medical conditions the patient was diagnosed with        │
│                                                             │
│    Examples:                                                │
│    ├─ Diabetes (Code: 250.xx)                              │
│    ├─ Heart failure (Code: 428.xx)                         │
│    ├─ Sepsis (Code: 995.92)                                │
│    └─ Pneumonia (Code: 486)                                │
│                                                             │
│    Meaning: If Code=1 → Patient WAS diagnosed with this    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 2. PROCEDURES (ICD-9 Procedure Codes)                      │
│    ──────────────────────────────────────                  │
│    Medical interventions performed on the patient           │
│                                                             │
│    Examples:                                                │
│    ├─ Mechanical ventilation (Code: 93.90)                 │
│    ├─ Central line placement (Code: 38.93)                 │
│    ├─ Intubation (Code: 96.04)                             │
│    └─ Blood transfusion (Code: 99.04)                      │
│                                                             │
│    Meaning: If Code=1 → Patient UNDERWENT this procedure   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 3. LAB TESTS & VITAL SIGNS (Quantized/Binned)             │
│    ──────────────────────────────────────────────          │
│    Laboratory values binned into abnormality categories     │
│                                                             │
│    Examples:                                                │
│    ├─ High WBC count >15,000 (abnormal)                    │
│    ├─ Low hemoglobin <7.5 (anemia)                         │
│    ├─ High creatinine >3.0 (kidney dysfunction)            │
│    └─ Low oxygen saturation <90% (hypoxemia)               │
│                                                             │
│    Conversion:                                              │
│    Raw Value      →  Quantized Code                        │
│    ───────────────────────────────────                     │
│    WBC < 4,500    →  Code_Lab_WBC_LOW                      │
│    WBC 4,500-11k  →  Code_Lab_WBC_NORMAL                   │
│    WBC > 11,000   →  Code_Lab_WBC_HIGH                     │
│                                                             │
│    Meaning: If Code=1 → Patient HAD this abnormal value    │
└─────────────────────────────────────────────────────────────┘

DISTRIBUTION ACROSS 133,279 CODES:
═════════════════════════════════════════════════════════════
Diagnoses: ~10,000 codes
Procedures: ~3,000 codes
Lab Tests & Vitals: ~120,279 codes (binned measurements)
─────────────────────────────
Total: 133,279 codes
```

---

## DIAGRAM 5: Why 99.95% Zeros (Sparsity)

```
WHY IS THE DATA SO SPARSE?
===========================

MATRIX: 1,000 patients × 133,279 codes
Total cells: 133,279,000

Visual representation (showing actual vs. possible):

ACTUAL DATA DISTRIBUTION:
┌─────────────────────────────────────────────────────────┐
│ Patient 0:  [0, 0, 1, 0, 1, 0, ..., 0, 1, ..., 0]     │  63 codes present
│ Patient 1:  [0, 1, 0, 1, 0, 0, ..., 1, 0, ..., 1]     │  76 codes present
│ Patient 2:  [1, 0, 0, 1, 0, 1, ..., 0, 0, ..., 0]     │  65 codes present
│ Patient 3:  [0, 0, 0, 0, 1, 0, ..., 0, 1, ..., 1]     │  66 codes present
│ ...         [., ., ., ., ., ., ..., ., ., ..., .]     │
│ Patient 999:[0, 0, 1, 0, 0, 0, ..., 1, 0, ..., 0]     │  70 codes present
└─────────────────────────────────────────────────────────┘

Total "1"s: 67,131
Total "0"s: 133,211,869
Sparsity: 99.95%


REASON 1: PATIENT HETEROGENEITY
═════════════════════════════════════
Each patient has unique medical history

Patient A's codes:     [1, 0, 1, 0, 0, 1, 0, ...]
Patient B's codes:     [0, 1, 0, 1, 1, 0, 1, ...]
Patient C's codes:     [1, 1, 0, 0, 1, 0, 0, ...]

Each has ~67 codes out of 133,279 possible
Most codes will be "0" for most patients


REASON 2: CODE RARITY
═══════════════════════════════════
Most medical codes are RARE

Code Frequency Distribution:
┌─────────────────────────────────────┐
│ 3 codes: appear in 33% of patients  │  COMMON
│ 100 codes: appear in 1-5% of patients│  MODERATE
│ 133,000+ codes: appear in <1% each  │  RARE
└─────────────────────────────────────┘

So most patients DON'T have most codes = lots of zeros


REASON 3: HIGH-DIMENSIONAL MEDICAL VOCAB
════════════════════════════════════════════════════
Medical system has ENORMOUS vocabulary

Complete ICD-9 Code Universe:
├─ Diagnoses: 10,000+ possible codes
├─ Procedures: 3,000+ possible codes
└─ Lab binned values: 120,000+ possible codes

One patient can only have subset of this universe


REASON 4: STORAGE EFFICIENCY
═══════════════════════════════════════════════════════
Sparse format saves massive storage

Full matrix:
  1,000 patients × 133,279 codes × 1 bit = ~16 GB

Sparse CSR format:
  Only store: 67,131 non-zero entries + indices = ~5 MB

Compression: 99.95%+ savings!
```

---

## DIAGRAM 6: Class Imbalance

```
OUTCOME DISTRIBUTION:
======================

Total: 1,000 patients

Visualization using dots:
NEGATIVE (survived):     ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
                         ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
                         ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
                         ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
                         ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
                         ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
                         ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
POSITIVE (high-risk):    ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️

992 vs 8  (124:1 ratio)


THE PROBLEM (CLASS IMBALANCE):
═══════════════════════════════════════════════════════

If model just predicts "NEGATIVE" for everyone:
┌─────────────────────────────┐
│ Accuracy: 99.2%             │  (Always right for negatives!)
│ BUT...                       │
│ Fails to identify 8 POSITIVE │  (Completely misses high-risk!)
│ patients at risk             │
└─────────────────────────────┘

This is NOT a good model, even though accuracy looks high!


THE SOLUTION (CLASS WEIGHTING):
═════════════════════════════════════════════════════════

During training, weight each positive case 124x higher:

Loss = BCE_loss + weight * KL_loss

Where:
  weight = (# negatives) / (# positives)
         = 992 / 8
         = 124

Effect: Model learns to pay ATTENTION to rare positive cases
       instead of just predicting negative for everything
```

---

## DIAGRAM 7: From Real Hospital to Model Input

```
REAL WORLD HOSPITAL SCENARIO:
═══════════════════════════════════════════════════════════

Patient admitted with fever, high heart rate, low oxygen

Doctor's Notes:
┌─────────────────────────────────────────┐
│ Patient ID: 0                           │
│ Admission: Septic, needs ICU            │
│                                          │
│ Diagnoses:                              │
│ - Sepsis (blood infection)              │
│ - Acute kidney injury (kidney failure)  │
│ - Pneumonia (lung infection)            │
│                                          │
│ Procedures done:                        │
│ - Mechanical ventilation (oxygen help)  │
│ - Central venous catheter (IV in neck)  │
│ - Hemodialysis (kidney support)         │
│                                          │
│ Lab results:                            │
│ - WBC: 25,000 (very high - infected)    │
│ - Creatinine: 4.2 (kidney failure)      │
│ - O2 saturation: 85% (low oxygen)       │
│                                          │
│ Question: Will patient survive? OUTCOME?│
└─────────────────────────────────────────┘

                        PREPROCESSING
                              ↓

STANDARDIZED ICD-9 CODES:
Code_995_92    = Sepsis
Code_584_9     = Acute kidney injury
Code_486       = Pneumonia
Code_9390      = Mechanical ventilation
Code_3893      = Central line
Code_3995      = Hemodialysis
Code_Lab_WBC_HIGH     = High WBC (>20k)
Code_Lab_Creat_CRIT   = High creatinine (>3.5)
Code_Lab_O2_LOW       = Low O2 (<90%)
Code_...       = (other codes absent)

                    BINARY ENCODING
                              ↓

PATIENT VECTOR (DATASET FORMAT):
[0, 0, 0, ..., 1, ..., 1, ..., 1, ..., 1, ..., 1, ..., 1, ..., 1, ..., 0, ...]
                 ^           ^      ^      ^      ^      ^      ^
            sepsis       AKI  pneum vent  cline  hdial  wbc   creat  o2

With outcome label: POSITIVE (high-risk)

                    GNN MODEL INPUT
                              ↓

Model sees:
  - 9-10 codes present
  - 133,270 codes absent
  - Outcome: POSITIVE

Model learns:
  - This COMBINATION of codes predicts HIGH RISK
  - (Sepsis + kidney injury + low O2) = dangerous pattern
  - Interventions (vent, dialysis) indicate severity
```

---

## DIAGRAM 8: Complete Data Flow

```
ORIGINAL MIMIC-III DATASET
(Real hospital records)
         ↓
┌─────────────────────────────────────┐
│ Extract & Standardize:              │
│ - Patient records                   │
│ - Diagnoses (ICD-9)                 │
│ - Procedures (ICD-9)                │
│ - Lab values (quantize)             │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Create Binary Vectors:              │
│ - 1 if code present                 │
│ - 0 if code absent                  │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Store in Sparse Format:             │
│ - preprocess_x.pkl (CSR matrix)     │
│ - y_bin.pkl (labels)                │
│ - train/val/test indices            │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ This Dataset You Have:              │
│ - 1,000 patients                    │
│ - 133,279 codes                     │
│ - 67,131 present values             │
│ - 8 positive outcomes               │
│ - 992 negative outcomes             │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ GNN Model:                          │
│ - Learns relationships between codes│
│ - Predicts outcome (POS/NEG)        │
│ - Uses graph attention              │
│ - Outputs risk scores               │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Output:                             │
│ Risk prediction for patient         │
│ High/Low likelihood of adverse event│
└─────────────────────────────────────┘
```

---

## Summary Table

| Concept | Meaning | Example |
|---------|---------|---------|
| **Patient Row** | One patient's complete medical record | Patient 0: 63 codes present |
| **Code Column** | One unique medical code | Code_995_92 = Sepsis |
| **Value = 1** | Code is PRESENT in patient record | Patient had sepsis → 1 |
| **Value = 0** | Code is ABSENT from patient record | Patient didn't have heart failure → 0 |
| **Outcome Label** | Whether patient had adverse outcome | POSITIVE (died) or NEGATIVE (survived) |
| **Sparsity** | Percentage of zeros | 99.95% (mostly empty) |
| **Class Imbalance** | Ratio of positive to negative | 1:124 (rare positive cases) |
