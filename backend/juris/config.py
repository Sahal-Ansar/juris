"""Typed, layered configuration.

Two layers:

- ``Settings``: machine and secret settings from the environment and ``.env``
  (database, provider keys, paths, cache mode) plus default models and budget.
- ``PipelineProfile``: a named YAML file in ``configs/pipeline/`` describing one
  system configuration (IDEA_final §5): which stages run and with what parameters.
  It may override the default models and budget.

``resolve_run_config`` merges both into a ``RunConfig`` that holds no secrets and
is what goes into the run manifest (PLAN 0.5).
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import quote

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIGS_DIR = REPO_ROOT / "configs" / "pipeline"


class Role(StrEnum):
    """Every LLM-backed role; each can get its own model."""

    ISSUE_FRAMER = "issue_framer"
    RESEARCHER = "researcher"
    COUNSEL = "counsel"
    PRECEDENT_ANALYST = "precedent_analyst"
    VERIFIER = "verifier"
    CROSS_EXAMINER = "cross_examiner"
    JUROR = "juror"
    JUDGE = "judge"
    BASELINE = "baseline"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(_Strict):
    provider: str = "anthropic"
    name: str = "claude-opus-5-5"


class ModelsConfig(_Strict):
    """A default model plus optional per-role overrides."""

    default: ModelSpec = ModelSpec()
    roles: dict[Role, ModelSpec] = Field(default_factory=dict)

    def for_role(self, role: Role) -> ModelSpec:
        return self.roles.get(role, self.default)


class BudgetConfig(_Strict):
    """Per-case limits; ``None`` means unlimited."""

    max_tokens_per_case: int | None = Field(default=None, gt=0)
    max_usd_per_case: float | None = Field(default=None, gt=0)


CacheMode = Literal["read_write", "read_only", "off"]
StageMode = Literal["on", "off", "passthrough"]


class StagesConfig(_Strict):
    """Pipeline stages S0-S10 (IDEA_final §6). Used by ``kind: juris`` profiles."""

    s0_issue_framing: StageMode = "on"
    s1_research: StageMode = "on"
    s2_opening: StageMode = "on"
    s3_precedent: StageMode = "on"
    s4_verify: StageMode = "on"
    s5_rebuttal: StageMode = "on"
    s6_cross_exam: StageMode = "on"
    s7_verify: StageMode = "on"
    s8_jury: StageMode = "on"
    s9_judge: StageMode = "on"
    s10_final_verify: StageMode = "on"


class DeliberationParams(_Strict):
    """Defaults from IDEA_final §6.2."""

    max_issues: int = Field(default=4, ge=1)
    positions_per_issue: int = Field(default=2, ge=2, le=3)
    evidence_per_issue_min: int = Field(default=10, ge=1)
    evidence_per_issue_max: int = Field(default=25, ge=1)
    rebuttal_rounds: int = Field(default=1, ge=0, le=2)
    extra_searches_per_counsel: int = Field(default=3, ge=0)
    jurors: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _evidence_range(self) -> Self:
        if self.evidence_per_issue_min > self.evidence_per_issue_max:
            raise ValueError("evidence_per_issue_min must be <= evidence_per_issue_max")
        return self


class PipelineProfile(_Strict):
    """One named system configuration, loaded from ``configs/pipeline/<name>.yaml``."""

    name: str
    description: str = ""
    kind: Literal["juris", "b0", "b1", "b2"]
    seed: int = 0
    models: ModelsConfig | None = None
    budget: BudgetConfig | None = None
    stages: StagesConfig = StagesConfig()
    params: DeliberationParams = DeliberationParams()


class Settings(BaseSettings):
    """Environment settings. Secrets come only from env or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="JURIS_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    # Database: shares its variables with docker-compose.yml.
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_user: str = Field(default="juris", validation_alias="POSTGRES_USER")
    postgres_password: SecretStr = Field(
        default=SecretStr("juris"), validation_alias="POSTGRES_PASSWORD"
    )
    postgres_db: str = Field(default="juris", validation_alias="POSTGRES_DB")

    # Provider keys.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    # Defaults that profiles may override.
    models: ModelsConfig = ModelsConfig()
    budget: BudgetConfig = BudgetConfig()

    # LLM gateway flags.
    llm_cache: CacheMode = "read_write"
    llm_max_concurrency: int = Field(default=4, ge=1)

    # Paths.
    data_dir: Path = REPO_ROOT / "data"
    runs_dir: Path = REPO_ROOT / "data" / "runs"
    pipeline_configs_dir: Path = PIPELINE_CONFIGS_DIR

    def database_url(self, driver: str | None = "psycopg") -> str:
        """SQLAlchemy URL (``postgresql+psycopg://``), or a libpq URI if ``driver`` is None."""
        scheme = f"postgresql+{driver}" if driver else "postgresql"
        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"{scheme}://{user}:{password}@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_profile(
    name_or_path: str | Path, configs_dir: Path = PIPELINE_CONFIGS_DIR
) -> PipelineProfile:
    """Load a profile by name (``juris_full``) or by path to a YAML file."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = configs_dir / f"{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PipelineProfile.model_validate(data)


class RunConfig(_Strict):
    """Fully resolved, secret-free configuration for one run."""

    profile: PipelineProfile
    models: ModelsConfig
    budget: BudgetConfig
    llm_cache: CacheMode
    llm_max_concurrency: int

    def model_for(self, role: Role) -> ModelSpec:
        return self.models.for_role(role)


def resolve_run_config(settings: Settings, profile: PipelineProfile) -> RunConfig:
    """Profile values win over settings defaults; ``models`` and ``budget`` replace wholesale."""
    return RunConfig(
        profile=profile,
        models=profile.models or settings.models,
        budget=profile.budget or settings.budget,
        llm_cache=settings.llm_cache,
        llm_max_concurrency=settings.llm_max_concurrency,
    )
