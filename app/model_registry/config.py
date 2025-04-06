from pydantic import BaseModel


class ModelRegistryConfig(BaseModel):
    model_name: str
    model_version: str | None = None
    job_id: str | None = None
    artifact_path: str | None = None
    model_uri: str | None = None
    run_id: str | None = None
