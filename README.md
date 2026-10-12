# 🌾 SmartFarm AI — Hybrid Crop Disease Diagnostic Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.35+-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Gemini_Vision-2.5_Flash-blue?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/Groq-LLaMA_3.3_70B-purple?style=for-the-badge" />
</p>

> **SmartFarm AI** is an intelligent, multi-modal agricultural diagnostic system that identifies crop diseases from leaf images and generates actionable, farmer-friendly treatment plans using a **hybrid CNN + Vision LLM pipeline**.

---

## 📸 Demo

| 🏠 Home | 🩺 Diagnosis Tool | 📊 Farm History |
|--------|-----------------|----------------|
| Overview & quick start | Upload leaf photo → AI diagnosis | Past diagnoses & trends |

---

## 🌟 Key Features

- 🧠 **Custom CNN Model** — Trained on the PlantVillage dataset to classify **38 crop-disease combinations** across **14 plant species**
- 🔍 **Gemini Vision AI** — Google's `gemini-2.5-flash` multimodal model triggered when CNN confidence is low (< 75%), performing deep visual analysis for crop type, disease stage, and severity
- 💬 **Groq LLM (LLaMA 3.3 70B)** — Converts raw AI output into clear, farmer-friendly advice including organic and chemical treatment plans
- ⚡ **Smart Caching** — ChromaDB vector similarity cache avoids redundant API calls for previously analyzed images (>98% similarity threshold)
- 🗂️ **Long-Term Farm Memory** — ChromaDB-backed persistent storage tracks diagnosis history across sessions
- 🌐 **Multi-Page Streamlit UI** — Clean dashboard with Home, Diagnosis Tool, and Farm History pages

---

## 🏗️ Architecture

```
Farmer uploads leaf image
        │
        ▼
┌──────────────────────────────────────┐
│        Hybrid Predictor Pipeline     │
│                                      │
│  1. Check Vector Cache (ChromaDB)    │
│         ↓ (cache miss)               │
│  2. CNN Model → disease label + conf │
│         ↓                            │
│  3. If conf < 75%:                   │
│     Gemini Vision 2.5 Flash          │
│     (crop + disease + severity)      │
│         ↓                            │
│  4. Groq LLaMA 3.3 70B              │
│     (farmer-friendly explanation)    │
│         ↓                            │
│  5. Store result in ChromaDB         │
└──────────────────────────────────────┘
        │
        ▼
 Diagnosis Report + Treatment Plan
```

---

## 🌿 Supported Crops & Diseases (38 Classes)

| Crop | Diseases Detected |
|------|-------------------|
| 🍎 Apple | Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌽 Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca, Leaf Blight, Healthy |
| 🍊 Orange | Haunglongbing (Citrus Greening) |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Bell Pepper | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🌱 Soybean | Healthy |
| 🎃 Squash | Powdery Mildew |
| 🍅 Tomato | Bacterial Spot, Early/Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit (multi-page) |
| **CNN Model** | TensorFlow / Keras — Custom trained on PlantVillage |
| **Vision AI** | Google Gemini 2.5 Flash (Multimodal) |
| **Language AI** | Groq API — LLaMA 3.3 70B Versatile |
| **Vector Memory** | ChromaDB (image & diagnosis caching) |
| **Image Processing** | OpenCV, Pillow (PIL) |
| **Data** | NumPy, Pandas |
| **Environment** | python-dotenv |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- API Keys for:
  - [Google AI Studio](https://aistudio.google.com/) → `GOOGLE_API_KEY`
  - [Groq Console](https://console.groq.com/) → `GROK_API_KEY`

### 1. Clone the repository
```bash
git clone https://github.com/gauravbharambe12/Smart-Farm-AI.git
cd Smart-Farm-AI
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
GROK_API_KEY=your_groq_api_key_here
```

### 4. Add the trained CNN model
Download the model weights and place them at:
```
model/smartfarm_cnn_best_model.keras
```
> 📥 **Model Download:** *(Add your Hugging Face or Google Drive link here)*

### 5. Run the application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Smart-Farm-AI/
│
├── app.py                          # Main Streamlit router
├── 🏠_Home.py                      # Home page
│
├── pages/
│   ├── 1_🩺_Diagnosis_Tool.py      # Upload & diagnose page
│   └── 2_📊_Farm_History.py        # History & trends page
│
├── core/
│   ├── predict.py                  # CNN model loader & inference
│   └── feature_extractor.py        # Image feature extraction
│
├── services/
│   ├── hybrid_predictor.py         # Main AI orchestration pipeline
│   └── memory_service.py           # ChromaDB vector memory
│
├── ai_modules/
│   ├── llm_client.py               # Gemini Vision + Groq LLM clients
│   └── prompt_templates.py         # Structured AI prompt templates
│
├── model/
│   └── class_indices.json          # 38-class disease label mapping
│
├── data/
│   └── uploads/                    # Uploaded leaf images (runtime)
│
├── .env                            # API keys (NOT committed)
├── .gitignore
└── requirements.txt
```

---

## 🔬 How the Hybrid Pipeline Works

1. **Image Upload** — Farmer uploads a leaf photo via the Streamlit UI
2. **Cache Check** — ChromaDB checks if a near-identical image was analyzed recently (similarity > 98%)
3. **CNN Inference** — Custom Keras CNN classifies the disease from 38 possible classes
4. **Confidence Gate** — If CNN confidence ≥ 75% → proceed to Groq. If < 75% → trigger Gemini Vision
5. **Gemini Vision** (conditional) — Google's multimodal model analyzes the image for crop type, disease identity, severity stage, and confidence
6. **Groq LLM** — LLaMA 3.3 70B converts technical AI output into farmer-friendly language with treatment plans
7. **Memory Storage** — Diagnosis saved to ChromaDB for future cache hits and farm history tracking

---

## 📊 Model Information

| Property | Details |
|----------|---------|
| **Architecture** | Custom CNN (Keras) |
| **Dataset** | PlantVillage (54,305 images) |
| **Classes** | 38 (14 crops × disease variants) |
| **Input Shape** | 128×128×3 RGB |
| **CNN Confidence Threshold** | 75% (triggers Gemini below this) |
| **Training Notebook** | `Final_project_model_training (2).ipynb` |

---

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini Vision API key | Required |
| `GROK_API_KEY` | Groq LLM API key | Required |
| `CNN_CONF_THRESHOLD` | Minimum CNN confidence before Vision AI triggers | `0.75` |
| `VISION_CONF_THRESHOLD` | Minimum Gemini confidence to store in cache | `0.50` |

---

## 🤝 Author

**Gaurav Bharambe**
- 🎓 B.Tech — Artificial Intelligence & Data Science, Dr. Babasaheb Ambedkar Technological University, Lonere
- 📧 gauravbharambe485@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/gaurav-bharambe)
- 🐙 [GitHub](https://github.com/gauravbharambe12)

---

## 📄 License

This project is for academic and educational purposes.

---

<p align="center">Made with ❤️ for farmers and agriculture 🌱</p>
