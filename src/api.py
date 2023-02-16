from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

import ee
from google.auth import compute_engine

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.data.constantes import SERVICE_ACCOUNT, SERVICE_KEY

from src.routers import classificationRouter

app = FastAPI()

if os.environ.get("_ENVIRONMENT") == "PRODUCTION":
    credentials = compute_engine.Credentials(scopes=['https://www.googleapis.com/auth/earthengine'])
else:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, SERVICE_KEY)

ee.Initialize(credentials)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "*"
]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
)

app.include_router(classificationRouter.router)

@app.get("/")
async def read_root() -> dict:
    image = ee.Image("projects/agritech-local/assets/vhr_raw");
    example_image = image.getThumbURL(params={
        "dimensions": 256
    })
    return {
        "message": "Hello ArkeUp",
        "example_image": example_image
    }

