from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.services.data_processing import DataProcessingService
from app.services.storage import StorageService
from typing import Dict, Any

router = APIRouter()

@router.post("/upload", response_model=Dict[str, Any])
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Uploads a file, cleans it, and returns metadata + basic statistics.
    Indexing happens in the background.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    print(f"DEBUG: Starting upload for {file.filename}")
    
    # 1. Read File
    df = await DataProcessingService.read_file(file)
    print(f"DEBUG: Read file done. Rows: {len(df)}")
    
    # 2. Clean Data
    df_clean = DataProcessingService.clean_data(df)
    print("DEBUG: Clean data done")
    
    # 3. Generate Metadata & Stats
    metadata = DataProcessingService.get_dataset_metadata(df_clean)
    stats = DataProcessingService.get_basic_stats(df_clean)
    print("DEBUG: Stats done")
    
    # 4. Save to Storage
    dataset_id = StorageService.save_dataset(df_clean, file.filename)
    print(f"DEBUG: Saved dataset {dataset_id}")
    
    # 5. Trigger Background Indexing
    background_tasks.add_task(StorageService.index_dataset, dataset_id, df_clean)
    
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "metadata": metadata,
        "analysis": stats,
        "message": "File processed successfully"
    }
