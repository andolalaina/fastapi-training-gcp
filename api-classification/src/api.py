from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import ee
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.data.constantes import SERVICE_ACCOUNT, SERVICE_KEY

from src.routers import classificationRouter

app = FastAPI()

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, SERVICE_KEY)
ee.Initialize(credentials)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
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
    
    return {"message": "Hello ArkeUp"}

