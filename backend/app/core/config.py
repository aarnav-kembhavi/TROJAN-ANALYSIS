from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    APP_NAME: str = "Hardware Trojan Detector"
    API_V1_STR: str = "/api/v1"
    
    # Model Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../TROJAN-ANALYSIS/Dataset/best_model.pkl")
    PREPROCESSOR_PATH: str = os.getenv("PREPROCESSOR_PATH", "../TROJAN-ANALYSIS/Dataset/preprocessor_params.pkl")
    
    # Storage
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", "./static/results")
    UPLOADS_DIR: str = os.getenv("UPLOADS_DIR", "./static/uploads")
    
    # Security
    CORS_ORIGINS: List[str] = ["*"]
    
    DEBUG: bool = True

    class Config:
        env_file = ".env"

settings = Settings()
