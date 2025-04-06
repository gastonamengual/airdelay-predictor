import ray
from app.database.database_connector import DatabaseConnector
from app.experiment_tracking import AbstractExperimentTracker
from app.model_registry import MLFlowModelRegistry
from app.model_registry.config import ModelRegistryConfig
from app.pipeline import ModelPipeline, PreprocessingPipeline
from app.settings import Settings

ray.init(ignore_reinit_error=True)  # type: ignore


@ray.remote  # type: ignore
def async_train(
    model_name: str,
    experiment_tracker: AbstractExperimentTracker,
    preprocessing_pipeline: PreprocessingPipeline,
    model_pipeline: ModelPipeline,
    model_registry: MLFlowModelRegistry,
) -> ModelRegistryConfig:
    experiment_tracker.set_experiment()
    with experiment_tracker.start_run() as run:
        run_id = run.__dict__["_info"].__dict__["_run_id"]

        database_connector = DatabaseConnector(
            engine_url=Settings.FLIGHTS_DB, table_name=Settings.FLIGHTS_DB_TABLE
        )
        flihgts = database_connector.load()

        train_dataset, valid_dataset, _ = preprocessing_pipeline.run(flihgts)
        training_output = model_pipeline.train(train_dataset, valid_dataset, run_id)
        model = model_pipeline.get_model_from_checkpoint(training_output.checkpoint)

        model_registry_config = model_registry.create_model(
            model, ModelRegistryConfig(model_name=model_name)
        )
        context = ray.get_runtime_context()  # type: ignore
        model_registry_config = model_registry_config.model_copy(
            update={"job_id": context.job_id}
        )

        database_connector = DatabaseConnector(
            engine_url=Settings.MODELS_DB, table_name=Settings.MODELS_DB_TABLE
        )
        database_connector.save([model_registry_config.model_dump()])

        return ModelRegistryConfig(model_name=model_name)
