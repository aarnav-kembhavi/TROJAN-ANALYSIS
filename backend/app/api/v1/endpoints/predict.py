from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
import os
import uuid
import shutil
import aiofiles
from app.services.trojan_service import trojan_service
from app.core.config import settings

router = APIRouter()

async def cleanup_temp_files(path: str):
    """Background task to remove temp upload directory."""
    if os.path.exists(path):
        shutil.rmtree(path)

@router.post("/predict")
async def predict_circuit(
    background_tasks: BackgroundTasks,
    features_file: UploadFile = File(...),
    edges_file: UploadFile = File(...),
    meta_file: UploadFile = File(None)
):
    # 1. Create a unique session directory
    session_id = str(uuid.uuid4())
    temp_dir = os.path.join(settings.UPLOADS_DIR, session_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 2. Save uploaded files
        for upload_file, name in [
            (features_file, "features.npy"),
            (edges_file, "edge_index.npy"),
            (meta_file, "meta.json")
        ]:
            if upload_file:
                file_path = os.path.join(temp_dir, name)
                async with aiofiles.open(file_path, 'wb') as out_file:
                    content = await upload_file.read()
                    await out_file.write(content)

        # 3. Run Inference via the service
        # TrojanInferenceService.predict_circuit expects a path to a directory
        result = trojan_service.predict_circuit(temp_dir)
        
        if result.get("status") == "failed":
            raise HTTPException(status_code=400, detail=result.get("error"))

        # 4. Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, temp_dir)
        
        return result

    except Exception as e:
        # Cleanup on failure
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "model_loaded": trojan_service is not None
    }
