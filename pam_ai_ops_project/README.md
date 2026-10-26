# PAMSentinel: Multimodal Clinical Decision Support & Explainable AI for Early Detection of Primary Amoebic Meningoencephalitis (*Naegleria fowleri*)

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2014-black?style=flat&logo=next.js)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20Late%20Fusion-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![XAI](https://img.shields.io/badge/Explainability-Grad--CAM%20%26%20SHAP-blueviolet?style=flat)]()
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat&logo=python)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=flat&logo=typescript)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Executive Summary & Clinical Background

**Primary Amoebic Meningoencephalitis (PAM)**, caused by the free-living thermophilic amoeba *Naegleria fowleri*, is an ultra-rare, fulminant central nervous system infection with a **case fatality rate exceeding 97%**. Death typically occurs within **1 to 12 days** (median 5 days) due to acute hemorrhagic necrotizing meningoencephalitis and subsequent uncal cerebral herniation.

### The Diagnostic Dilemma
Initial clinical signs (excruciating bifrontal headache, high pyrexia, nuchal rigidity, projective vomiting) closely mimic acute bacterial meningitis. Standard antibiotic therapies provide zero efficacy against amoebic trophozoites, leading to catastrophic diagnostic delays. Aggressive intervention (IV/intrathecal **Miltefosine**, Liposomal Amphotericin B, and targeted hypothermia) must begin within the critical **48-hour therapeutic window** to ensure patient survival.

### Technical Solution
**PAMSentinel** is an end-to-end clinical decision support system combining:
1. **Multimodal Dual-Branch Late Fusion**:
   - **Vision Branch**: Modified `ResNet-18` (1-channel 224×224 grayscale axial brain MRI) extracting a 64-dimensional latent visual embedding.
   - **Tabular Branch**: 3-layer Multilayer Perceptron (MLP) mapping 9 bedside clinical parameters (optimized via Recursive Feature Elimination and Chi-Square significance) to a 32-dimensional embedding.
   - **Fusion Head**: Concatenates both embeddings (96-D fused tensor) through dense projection and sigmoid activation to output calibrated PAM risk probabilities.
   - **Classical ML Baseline**: 59-dimensional fused Random Forest classifier achieving **95.73% accuracy**, **96.34% sensitivity (recall)**, and **0.9832 ROC-AUC**.
2. **Dual Explainable AI (XAI)**:
   - **Grad-CAM (Vision)**: Gradient-weighted Class Activation Mapping isolating neuropathological foci in the basal meningeal cisterns and cerebral edema regions.
   - **SHAP (Tabular)**: KernelExplainer calculating Shapley feature attribution weights to clearly delineate risk-accelerating symptoms vs. protective clinical indicators.
3. **Clinical Web Portal**:
   - High-performance **FastAPI** asynchronous REST backend serving inference in **< 120ms**.
   - Modern **Next.js 14** web application with Tailwind CSS, Recharts dynamic visualizations, and an interactive **Explainable AI Studio**.

---

## 2. Directory Architecture

```text
pam_ai_ops_project/
├── .gitignore                         # Ignore rules (node_modules, venv, caches, builds)
├── README.md                          # This documentation & execution guide
├── requirements.txt                   # Pinned Python dependencies (PyTorch, FastAPI, SHAP, etc.)
├── run_server.py                      # Main backend startup script (port 8000)
├── best_multimodal_model.pth          # Pre-trained PyTorch Late Fusion weights (44.9 MB)
│
├── api/                               # Phase 6 PyTorch Late Fusion & XAI Endpoints
│   └── main.py                        # POST /api/predict_multimodal (ResNet-18 + Grad-CAM + SHAP)
│
├── app/                               # Production FastAPI Server & Predictor Engine
│   ├── main.py                        # Root FastAPI app, CORS, legacy & XAI routing
│   ├── predictor.py                   # PAMPredictor inference engine & OpenCV feature extraction
│   └── schemas.py                     # Pydantic clinical input/output validation models
│
├── data/                              # Dataset repository
│   ├── raw/                           # Synthetic & clinical cohort datasets (1,000 cases)
│   └── processed/                     # Train/test splits & Power BI Star Schema export
│
├── frontend/                          # Enterprise Next.js 14 Web Interface
│   ├── components/                    # Modular UI components (ClinicalDashboard, XAI Studio)
│   ├── src/app/                       # App Router pages, layout, and global styling
│   ├── package.json                   # Node.js dependencies (Lucide, Recharts, Framer Motion)
│   └── tailwind.config.ts             # Design system & color tokens
│
├── models/                            # Persisted Classical ML Artifacts
│   └── artifacts/                     # best_model.pkl, multimodal_model.pkl, scaler.pkl
│
├── notebooks/                         # Jupyter Research & Evaluation Walkthroughs
│   ├── 01_phase1_data_engineering.ipynb
│   ├── 02_phase2_model_training.ipynb
│   ├── 03_model_evaluation_presentation.ipynb
│   └── PAM_Clinical_Decision_Support_Master_Walkthrough.ipynb
│
├── src/                               # Deep Learning & XAI Source Modules
│   ├── clinical_shap.py               # KernelExplainer for 9-parameter bedside vector
│   ├── multimodal_model.py            # PAMLateFusionNet PyTorch dual-backbone architecture
│   └── vision_gradcam.py              # MRIExplainer Grad-CAM visualization engine
│
├── run_phase1.py                      # Standalone data engineering & RFE execution
├── run_phase2.py                      # Standalone tabular model training & evaluation
└── run_phase5_xai.py                  # Standalone Grad-CAM & SHAP validation script
```

---

## 3. Step-by-Step Setup Guide on a New Device

Follow these instructions to set up and execute the entire project from scratch on any new device (Windows, macOS, or Linux).

### Prerequisites
Make sure your system has the following installed:
- **Python**: Version `3.10`, `3.11`, or `3.12` ([Download Python](https://www.python.org/downloads/))
- **Node.js**: Version `18.0.0+` or `20.0.0+` with `npm` ([Download Node.js](https://nodejs.org/))
- **Git**: For version control ([Download Git](https://git-scm.com/))

---

### Step 1: Open Terminal & Navigate to Project Directory

```bash
cd pam_ai_ops_project
```

---

### Step 2: Set Up & Launch Python Backend (Terminal 1)

#### 2.1 Create & Activate a Virtual Environment
- **Windows (Command Prompt / PowerShell)**:
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 2.2 Install Python Dependencies
Upgrade `pip` and install all required scientific, machine learning, and server packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch**: `requirements.txt` installs standard PyTorch. If you have an NVIDIA GPU and want CUDA acceleration, install the appropriate CUDA PyTorch build following [PyTorch Get Started](https://pytorch.org/get-started/locally/). If no GPU is found, the system automatically falls back to CPU inference.

#### 2.3 Start the Backend Server
With your virtual environment active, run:
```bash
python run_server.py
```

The server will start and output:
```text
Launching PAM Early Prediction API Server...
Documentation available at: http://localhost:8000/docs
ReDoc available at:         http://localhost:8000/redoc
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **Alternative ReDoc UI**: `http://127.0.0.1:8000/redoc`
- **System Health Check**: `http://127.0.0.1:8000/health`

---

### Step 3: Set Up & Launch Frontend (Terminal 2)

Open a **new, separate terminal window** (keep the backend server running in the first terminal).

Navigate to the frontend directory:
```bash
cd pam_ai_ops_project/frontend
```

#### 3.1 Install Node.js Dependencies
```bash
npm install
```

#### 3.2 Start the Next.js Development Server
```bash
npm run dev
```

The frontend will compile and launch:
```text
  ▲ Next.js 14.2.35
  - Local:        http://localhost:3000

 ✓ Ready in 3.6s
```

Open your browser and navigate to:
👉 **`http://localhost:3000`**

---

## 4. How to Use the Web Application

### A. Patient Intake Form & Triage Portal
1. Scroll to the **Clinical Decision Support & XAI Portal** section.
2. Select a quick clinical vignette:
   - Click **"High-Risk PAM Case"** to load typical acute amoebic meningoencephalitis signs (freshwater exposure, bifrontal headache, pyrexia, nuchal rigidity, high CSF pressure).
   - Click **"Control Non-Amoebic Case"** to load a non-amoebic presentation.
3. (Optional) Upload an axial brain MRI scan in PNG/JPEG format (or leave blank to auto-synthesize a baseline scan).
4. Click **"Execute Multimodal Risk Stratification"**.
5. Inspect the dynamic diagnostic card:
   - **Triage Level**: `CRITICAL EMERGENCY ALERT: POSITIVE PAM SCREEN` vs `LOW RISK`.
   - **Fused Risk Probability**: Calibrated confidence gauge.
   - **Clinical Triage Pathway**: 4 emergency intervention steps (Miltefosine, Liposomal Amphotericin B, Hypothermia, ICP Decompression).
   - **Explainable AI (XAI) Attribution**:
     - **Panel A (Grad-CAM)**: Heatmap overlay highlighting basal cistern activation.
     - **Panel B (SHAP)**: Horizontal bar chart highlighting positive risk drivers (coral red `#f43f5e`) vs. protective indicators (sky blue `#38bdf8`).

### B. Interactive Explainable AI (XAI) Studio
1. Click **`XAI Studio`** in the top navigation bar or switch to the **"Explainable AI (XAI) Studio"** tab in the portal.
2. Interactively adjust continuous normalized sliders:
   - Patient Age
   - Symptom Onset Progression
   - CSF Opening Pressure
3. Toggle individual acute symptoms and observe real-time dual-backbone predictions, Grad-CAM neuroimaging localization, and SHAP Shapley weights.

---

## 5. Offline Research Scripts & Notebooks

You can also run standalone analytical pipelines independently of the web server:

| Script | Purpose |
|---|---|
| `python run_phase1.py` | Runs clinical feature engineering, Chi-Square significance testing, Pearson collinearity pruning, and Decision Tree RFE. |
| `python run_phase2.py` | Trains and benchmarks Random Forest, Gradient Boosting, XGBoost, and Logistic Regression models; exports validation metrics. |
| `python run_phase5_xai.py` | Runs standalone Grad-CAM and SHAP explainability pipelines, saving output heatmaps and attribution charts. |
| `notebooks/` | Interactive Jupyter Notebooks walking through each phase step-by-step with exploratory data analysis and ROC curves. |

---

## 6. Troubleshooting & FAQs

### Q1: Port 8000 or 3000 is already in use
- **Windows**:
  ```powershell
  # Check process using port 8000
  netstat -ano | findstr :8000
  # Terminate process by PID
  taskkill /PID <PID> /F
  ```
- **macOS / Linux**:
  ```bash
  lsof -ti :8000 | xargs kill -9
  lsof -ti :3000 | xargs kill -9
  ```

### Q2: PyTorch / torchvision warning on CPU devices
The code detects GPU capability via `torch.cuda.is_available()` and gracefully falls back to CPU mode. Pre-trained weights in `best_multimodal_model.pth` are loaded with `map_location=device` ensuring 100% compatibility across all machines.

### Q3: Next.js build errors or module resolution issues
Ensure dependencies are cleanly installed:
```bash
cd pam_ai_ops_project/frontend
rm -rf node_modules .next package-lock.json
npm install
npm run build
```

---

## 7. Git Push Instructions (When Ready)

When you are ready to push this project to your GitHub repository, run the following commands from your terminal (make sure to replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository's URL):

```bash
# 1. Check status (verify no node_modules or cache files are staged)
git status

# 2. Stage all project files
git add .

# 3. Commit changes
git commit -m "feat: complete multimodal clinical decision support system with Grad-CAM and SHAP XAI"

# 4. Set remote repository (if not already configured)
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git

# 5. Push to GitHub
git branch -M main
git push -u origin main
```

---

## 8. Clinical Research Disclaimer
*This software is an investigational Clinical Decision Support prototype developed for academic research, algorithm validation, and portfolio demonstration. It does not replace certified laboratory diagnostic procedures (such as CDC real-time PCR or CSF wet mount microscopy) or licensed physician judgment.*
