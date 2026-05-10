import sys
import os
from app.core.config import settings

# Add TROJAN-ANALYSIS/Dataset to path to import TrojanInferenceService
# Note: In a real production environment, this would be a proper package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../TROJAN-ANALYSIS/Dataset")))

try:
    from backend_service import TrojanInferenceService
except ImportError:
    # Fallback if path appending failed or structure changed
    from TROJAN_ANALYSIS.Dataset.backend_service import TrojanInferenceService

# Initialize the service instance (Singleton pattern for FastAPI)
trojan_service = TrojanInferenceService(
    model_path=settings.MODEL_PATH,
    preprocessor_path=settings.PREPROCESSOR_PATH
)
# Update results directory to API static results
trojan_service.results_dir = os.path.abspath(settings.RESULTS_DIR)
