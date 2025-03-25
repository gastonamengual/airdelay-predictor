from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import model_router, router


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(model_router)

    return app
