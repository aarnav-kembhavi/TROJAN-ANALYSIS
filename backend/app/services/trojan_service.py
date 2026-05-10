import sys
import os
import pandas as pd
from app.core.config import settings
from app.services.inference_engine import TrojanInferenceEngine

class TrojanInferenceService:
    def __init__(self, model_path: str):
        self.engine = TrojanInferenceEngine(model_path)
        self.results_dir = settings.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

    def predict_circuit(self, data_dir: str):
        result = self.engine.predict_circuit(data_dir)
        
        if result["status"] == "success":
            # Save node scores to CSV artifact
            session_id = os.path.basename(data_dir)
            csv_filename = f"{session_id}_node_scores.csv"
            csv_path = os.path.join(self.results_dir, csv_filename)
            
            node_probs = result.pop("node_probabilities", [])
            df = pd.DataFrame({
                "node_id": range(len(node_probs)),
                "anomaly_score": node_probs
            })
            df.to_csv(csv_path, index=False)
            
            result["artifacts"]["node_scores_csv"] = csv_path
            
        return result

# Initialize the service instance (Singleton pattern for FastAPI)
# Path is relative to the backend root where the server runs
trojan_service = TrojanInferenceService(
    model_path=settings.MODEL_PATH
)
