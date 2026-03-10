"""Runtime configuration model for Crucible.

This module owns all environment variable parsing and validation.
Other modules consume a typed config object instead of raw env reads.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from core.constants import DEFAULT_DATA_ROOT
from core.errors import CrucibleConfigError


@dataclass(frozen=True)
class CrucibleConfig:
    """Validated runtime configuration.

    Attributes:
        data_root: Local root directory for metadata and snapshots.
        s3_region: Optional default AWS region for S3 operations.
        s3_profile: Optional AWS profile for boto3 session initialization.
        random_seed: Seed used for deterministic shuffling.
    """

    data_root: Path
    s3_region: str | None
    s3_profile: str | None
    random_seed: int

    @classmethod
    def from_env(cls) -> "CrucibleConfig":
        """Build config from process environment variables.

        Returns:
            A validated config object.

        Raises:
            CrucibleConfigError: If environment values are invalid.
        """
        data_root_value = os.getenv("CRUCIBLE_DATA_ROOT", str(DEFAULT_DATA_ROOT))
        s3_region = os.getenv("CRUCIBLE_S3_REGION")
        s3_profile = os.getenv("CRUCIBLE_S3_PROFILE")
        random_seed_value = os.getenv("CRUCIBLE_RANDOM_SEED", "42")
        random_seed = _parse_random_seed(random_seed_value)
        return cls(
            data_root=Path(data_root_value).expanduser().resolve(),
            s3_region=s3_region,
            s3_profile=s3_profile,
            random_seed=random_seed,
        )


def _parse_random_seed(raw_value: str) -> int:
    """Parse the random seed environment value.

    Args:
        raw_value: Raw string from environment.

    Returns:
        Parsed integer seed.

    Raises:
        CrucibleConfigError: If value cannot be parsed into int.
    """
    try:
        return int(raw_value)
    except ValueError as error:
        raise CrucibleConfigError(
            "Invalid CRUCIBLE_RANDOM_SEED value: "
            f"expected integer, got '{raw_value}'. "
            "Set CRUCIBLE_RANDOM_SEED to a numeric value."
        ) from error
