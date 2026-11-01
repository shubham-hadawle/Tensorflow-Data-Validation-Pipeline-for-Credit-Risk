# MLOps Lab 1: TFDV Data Validation & Automated Gatekeeper

An end-to-end machine learning data validation project building upon **TensorFlow Data Validation (TFDV)** principles. This project includes a real-world financial risk dataset, an predictive machine learning model, an automated pre-inference validation gatekeeper, and an interactive real-time web dashboard.

---

## Table of Contents

1. [Key Modifications from the Original Lab](#key-modifications-from-the-original-lab)
2. [Project Architecture](#project-architecture)
3. [Dependencies & Environment Setup](#dependencies--environment-setup)
4. [Dataset & Split Design](#dataset--split-design)
5. [Step-by-Step TFDV Engineering Walkthrough](#step-by-step-tfdv-engineering-walkthrough)
   - [Step 1: Baseline Statistics & Schema Inference](#step-1-baseline-statistics--schema-inference)
   - [Step 2: Comparative Statistics & Facets Overview](#step-2-comparative-statistics--facets-overview)
   - [Step 3: Anomaly Detection & Constraint Relaxation](#step-3-anomaly-detection--constraint-relaxation)
   - [Step 4: Multi-Environment Schema Masking](#step-4-multi-environment-schema-masking)
   - [Step 5: Chebyshev L-infinity Drift Detection](#step-5-chebyshev-l-infinity-drift-detection)
   - [Step 6: Downstream Model & Automated Gatekeeper](#step-6-downstream-model--automated-gatekeeper)
6. [Interactive Web Dashboard](#interactive-web-dashboard)
7. [Automated CI/CD & Unit Testing](#automated-cicd--unit-testing)
8. [Quickstart & Reproduction Commands](#quickstart--reproduction-commands)

---

## Key Modifications from the Original Lab

| :--- | :--- |
| **Dataset Domain** | **FinTech Credit Risk & Loan Default Assessment** (Multi-variate continuous financial metrics, credit ratings, loan intent categories) |
| **Data Splits** | **Train (10,000 clean baseline), Eval (2,500 with injected anomalies), Serving (2,500 unlabelled), Drifted Batch (3,000 macroeconomic downturn)** |
| **Injected Anomalies** | **Novel categorical values (`CRYPTO_MINING`, `TAX_SETTLEMENT`), severe missingness spikes (28% nulls violating presence constraints), and negative DTI range violations** |
| **Environment Handling** | **Multi-environment masking for `is_default` (`not_in_environment: SERVING`) avoiding false-alarm dropped column failures** |
| **Drift Monitoring** | **Chebyshev distance ($L_\infty = \max \|P(x)-Q(x)\|$) drift detection + relative numerical mean shift tracking** |
| **Machine Learning Model** | **L2-regularized Credit Risk Classifier (ROC-AUC ~0.74, Accuracy 81.2%) trained on baseline and evaluated on held-out splits** |
| **Automated Gatekeeper** | **Programmatic `DataValidationGate` guarding model inference—blocks corrupt batches and warns on distribution drift** |
| **Interactive Dashboard** | **Full-stack Web Dashboard (FastAPI backend + responsive standalone HTML/CSS/JS) with interactive Facets distribution plots and live applicant scoring** |
| **CI/CD & Testing** | **Automated GitHub Actions workflow (`.github/workflows/ci.yml`) + 12 unit tests verifying schema invariants** |

---

## Project Architecture
---

## Dependencies & Environment Setup

This project is built to run reliably on modern Python versions (3.10 and 3.11). It includes zero-dependency fallback algorithms for lightweight environments (e.g. CI runners or local laptops) while remaining 100% compatible with native TensorFlow Extended (TFX) and TensorFlow Data Validation (TFDV).

### Required Core Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `python` | `>=3.10, <3.12` | Runtime environment |
| `numpy` | `>=1.24.0` | Numerical linear algebra, array operations, vectorization |
| `pandas` | `>=2.0.0` | Dataframe manipulation, CSV parsing, data slicing |
| `scipy` | `>=1.10.0` | Statistical distributions and metric calculations |
| `fastapi` | `>=0.95.0` | REST API backend for live validation gate scoring |
| `uvicorn` | `>=0.22.0` | ASGI server for dashboard application |
| `pydantic` | `>=1.10.0` | Schema payload validation for API requests |
| `matplotlib` | `>=3.7.0` | Visual figure generation and distribution plotting |
| `seaborn` | `>=0.12.0` | Statistical visualization enhancements |
| `pytest` / `unittest` | `Built-in / >=7.3.0` | Unit test suite execution |

### Optional Native TFDV Dependencies

To run with the native Google TFX engine:
```bash
pip install tensorflow-data-validation>=1.14.0 tensorflow>=2.14.0
