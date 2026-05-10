from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from app.api.v1.endpoints import predict
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)

# Mount static files for generated results/artifacts
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include Routers
app.include_router(predict.router, prefix=settings.API_V1_STR, tags=["Trojan Detection"])

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME} API. Visit /docs for documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
