# Hardware Trojan Detection API

A FastAPI-based backend for detecting structural anomalies and Trojans in gate-level netlists using Graph Attention Networks (GAT).

## 🚀 Features
- **Semi-Supervised Detection**: Uses GAT + GraphNorm with calibrated Top-5% aggregation.
- **RESTful API**: Clean modular structure with auto-generated Swagger docs.
- **Async Processing**: Fast file uploads and background cleanup.
- **Static Artifacts**: Directly serves generated node-level probability CSVs.

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   Copy `.env.example` to `.env` and adjust paths if necessary.
   ```bash
   cp .env.example .env
   ```

3. **Run the API**:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`.

## 📡 API Endpoints

### 🩺 Health Check
`GET /api/v1/health`
Checks if the model and preprocessor are loaded correctly.

### 🧠 Prediction
`POST /api/v1/predict`
Accepts `multipart/form-data`:
- `features_file`: `features.npy`
- `edges_file`: `edge_index.npy`
- `meta_file`: `meta.json` (optional)

**Returns**: Structured JSON with probabilities, suspicious nodes, and artifact paths.

### 📁 Static Files
`GET /static/results/{circuit_name}_inference_results.csv`
Download node-level detail CSVs.

## 🏗️ Project Structure
- `main.py`: Entry point and app configuration.
- `app/api/v1/endpoints/`: Route handlers.
- `app/services/`: Business logic and model interaction.
- `app/core/`: Configuration and settings.
- `static/`: Generated artifacts and temp uploads.
