"""The frozen bar: a versioned, hashed spec that every track is measured against.

The JSON file under ``bar/`` is the single source of truth. Its sha256 is printed
in every evaluation report and stored in every ledger row, so a result can only
be compared with another result measured against the same bar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BAR_DIR = REPO_ROOT / "bar"
DEFAULT_BAR = BAR_DIR / "bar_v1.json"


@dataclass(frozen=True)
class Bar:
  raw: dict
  sha256: str
  path: Path

  # convenience accessors -------------------------------------------------------
  @property
  def version(self) -> str:
    return self.raw["version"]

  @property
  def frozen(self) -> bool:
    return bool(self.raw.get("frozen", False))

  @property
  def tracking(self) -> dict:
    return self.raw["tracking"]

  @property
  def push(self) -> dict:
    return self.raw["push"]

  @property
  def sim2sim(self) -> dict:
    return self.raw["sim2sim"]

  @property
  def fall(self) -> dict:
    return self.raw["fall"]

  @property
  def thresholds(self) -> dict:
    return self.raw["thresholds"]

  @property
  def short_hash(self) -> str:
    return self.sha256[:12]


def file_sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bar(path: str | Path | None = None) -> Bar:
  p = Path(path) if path is not None else DEFAULT_BAR
  raw = json.loads(p.read_text())
  return Bar(raw=raw, sha256=file_sha256(p), path=p)
