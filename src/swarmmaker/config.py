"""Configuration management with singleton Settings instance."""
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class Settings:
    """Singleton settings instance that loads environment variables from .env file."""

    _instance: Optional["Settings"] = None
    _initialized: bool = False

    def __new__(cls) -> "Settings":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize settings by loading .env file if not already initialized."""
        if not Settings._initialized:
            self._load_env()
            Settings._initialized = True

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        if load_dotenv is None:
            return

        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"

        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable by key."""
        return os.environ.get(key, default)


# Singleton instance
settings = Settings()

