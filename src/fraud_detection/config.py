from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    transaction_path: str
    identity_path: str
    train_cutoff_days: int
    target_column: str


class FeaturesConfig(BaseModel):
    numeric_fill_value: float
    cat_fill_value: str


class LGBMConfig(BaseModel):
    n_estimators: int
    learning_rate: float
    num_leaves: int
    scale_pos_weight: float


class TrainingConfig(BaseModel):
    random_seed: int
    test_size: float = Field(gt=0.0, lt=1.0)
    lgbm: LGBMConfig


class EvaluationConfig(BaseModel):
    fp_cost: float
    fn_cost: float


class MLflowConfig(BaseModel):
    experiment_name: str
    model_name: str
    model_stage: str = "latest"


class ServingConfig(BaseModel):
    host: str
    port: int
    decision_threshold: float


class Config(BaseModel):
    data: DataConfig
    features: FeaturesConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    serving: ServingConfig


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
