from sqlalchemy import Column, Integer, String

from .abstract_table import AbstractTable


class ModelsTable(AbstractTable):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String)
    model_version = Column(String)
    job_id = Column(String)
    artifact_path = Column(String)
    model_uri = Column(String)
    run_id = Column(String)
