"""Environment-backed application configuration."""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; every value may be overridden through the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: str = "dev"
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 60
    postgres_url: str = (
        "postgresql+psycopg2://driftguard:driftguard@localhost:5432/driftguard"
    )
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db: str = "driftguard"
    artifact_dir: Path = Path("artifacts")
    random_seed: int = 42
    max_upload_mb: int = 50
    max_batch_rows: int = 50000
    psi_bins: int = 10
    null_bootstrap_b: int = 200
    null_quantile: float = 0.95
    norm_cap_psi: float = 0.5
    norm_cap_ks: float = 0.5
    norm_cap_jsd: float = 0.3
    d_aggregation: str = "max"
    severity_moderate: float = 0.15
    severity_high: float = 0.5
    batch_dis_moderate: float = 0.05
    batch_dis_high: float = 0.25
    perf_meaningful_drop: float = 0.02
    perf_severe_drop: float = 0.05
    perf_bootstrap_b: int = 200
    sensitivity_repeats: int = 5
    shap_sample_size: int = 2000
    min_labels: int = 200
    conventional_psi: float = 0.25
    conventional_ks_p: float = 0.01
    conventional_jsd: float = 0.1
    # Champion-challenger retraining: minimum AUC gain on the independent
    # test split before an administrator may deploy a challenger.
    retrain_min_gain_auc: float = 0.01
    retrain_min_production_labels: int = 200
    # ADWIN (River) sensitivity for the prediction / error streams.
    adwin_delta: float = 0.002

    @model_validator(mode="after")
    def validate_production_secret(self) -> "Settings":
        """Prevent the documented development key from being deployed."""
        if self.app_env != "dev" and (
            self.secret_key == "change-me" or len(self.secret_key) < 32
        ):
            raise ValueError(
                "SECRET_KEY must be at least 32 characters outside development"
            )
        return self


settings = Settings()
