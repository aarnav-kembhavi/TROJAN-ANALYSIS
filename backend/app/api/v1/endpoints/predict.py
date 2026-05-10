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

import subprocess

@router.post("/predict")
async def predict_circuit(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    # 1. Create a unique session directory
    session_id = str(uuid.uuid4())
    temp_dir = os.path.join(settings.UPLOADS_DIR, session_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 2. Save all uploaded files
        file_map = {}
        for f in files:
            # Handle possible nested paths if uploaded as folder
            filename = os.path.basename(f.filename)
            save_path = os.path.join(temp_dir, filename)
            async with aiofiles.open(save_path, 'wb') as out_f:
                content = await f.read()
                await out_f.write(content)
            file_map[filename] = save_path

        # 3. Decision Logic: Verilog vs. GNN Data
        has_gnn_data = "features.npy" in file_map and "edge_index.npy" in file_map
        
        if not has_gnn_data:
            # Look for a .v file
            v_files = [n for n in file_map.keys() if n.endswith(".v")]
            if not v_files:
                raise HTTPException(status_code=400, detail="No valid circuit data (GNN files or .v) found.")
            
            v_filename = v_files[0]
            v_path = file_map[v_filename]
            
            # Step 1: Convert Verilog to JSON (via WSL Yosys)
            json_path = os.path.join(temp_dir, "netlist.json")
            v_path_wsl = "/mnt/c" + os.path.abspath(v_path).replace("C:", "").replace("\\", "/")
            json_path_wsl = "/mnt/c" + os.path.abspath(json_path).replace("C:", "").replace("\\", "/")
            
            yosys_cmd = f'wsl yosys -p "read_verilog {v_path_wsl}; prep; write_json {json_path_wsl}"'
            subprocess.run(yosys_cmd, shell=True, check=True, capture_output=True)

            # Step 2: Convert JSON to GNN
            conv_cmd = ["python", "convert_verilog_to_gnn.py", json_path, temp_dir]
            subprocess.run(conv_cmd, check=True, capture_output=True)
            
            circuit_display_name = v_filename
        else:
            circuit_display_name = "Uploaded Graph"

        # 4. Run Inference via the service
        result = trojan_service.predict_circuit(temp_dir)
        
        if result.get("status") == "failed":
            raise HTTPException(status_code=400, detail=result.get("error"))

        result["metadata"]["circuit_name"] = result["metadata"].get("circuit_name") or circuit_display_name

        # 5. Schedule cleanup
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
