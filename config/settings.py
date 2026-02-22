"""
config/settings.py — Environment configuration
================================================
Reads all API keys and tuning parameters from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Auto-load .env from project root
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    # API Keys
    fred_api_key: str = ""
    fmp_api_key: str = ""
    newsapi_api_key: str = ""
    alphavantage_api_key: str = ""
    twelvedata_api_key: str = ""
    sec_user_agent: str = ""

    # HTTP
    http_timeout_sec: int = 20

    # Cache
    cache_dir: str = ".cache"
    cache_ttl_sec: int = 3600

    # Rate limiting
    rate_limit_qps: float = 2.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            fred_api_key=os.environ.get("FRED_API_KEY", ""),
            fmp_api_key=os.environ.get("FMP_API_KEY", ""),
            newsapi_api_key=os.environ.get("NEWSAPI_API_KEY", ""),
            alphavantage_api_key=os.environ.get("ALPHAVANTAGE_API_KEY", ""),
            twelvedata_api_key=os.environ.get("TWELVEDATA_API_KEY", ""),
            sec_user_agent=os.environ.get("SEC_USER_AGENT", ""),
            http_timeout_sec=int(os.environ.get("HTTP_TIMEOUT_SEC", "20")),
            cache_dir=os.environ.get("CACHE_DIR", ".cache"),
            cache_ttl_sec=int(os.environ.get("CACHE_TTL_SEC", "3600")),
            rate_limit_qps=float(os.environ.get("RATE_LIMIT_QPS", "2.0")),
        )


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
