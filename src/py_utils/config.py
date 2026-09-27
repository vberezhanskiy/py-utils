"""Runtime configuration for an py-utils instance.

A single `Config` object pins the active project root and the storage
directory; everything else (graph, retriever, memory) takes a `Config`
or a `project_root` and derives its own paths from there.

Optional per-project overrides live in ``<project_root>/.py-utils.toml``:

    [ignore]
    dirs = ["my_generated_dir"]      # extra dirs to skip during graph_build
    [extensions]
    extra = ["*.foo", "*.bar"]       # extra source-file extensions to index
    [graph]
    max_file_bytes = 10485760        # override the default 5 MB skip threshold

Loaded once at Config init; subsystems read it via ``config.project``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

DEFAULT_STORAGE = Path.home() / ".py-utils"
PROJECT_CONFIG_FILENAME = ".py-utils.toml"

# Потолок max_file_bytes из .py-utils.toml: больше гигабайта в граф не читается
# никогда, а опечатка в числе иначе пускала бы в память дампы целиком.
_MAX_FILE_BYTES_LIMIT = 1024 * 1024 * 1024


def project_slug_for(project_root: "str | Path") -> str:
    """Каталог проекта в хранилище — байт в байт как у Node-воркера.

    Граф строит Node (``legacyProjectSlug`` в agent-worker/src/tools/_store.ts:
    имя папки с заменой всего, кроме ``[a-zA-Z0-9_-]``, плюс md5 от
    ``path.resolve(root)``), а читает этот сайдкар. Здесь имя бралось как есть,
    а md5 — от ``Path.resolve()``, который на Windows меняет регистр на
    настоящий и раскрывает junction. У папки «Мой проект» или «site.ru»
    каталоги расходились, и сайдкар не видел построенного воркером графа.

    ``os.path.abspath`` повторяет ``path.resolve`` из Node: нормализует
    разделители и ``..``, но не трогает регистр и ссылки.
    """
    identity = os.path.abspath(str(project_root))
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.basename(identity))
    digest = hashlib.md5(identity.encode("utf-8")).hexdigest()[:8]
    return f"{name}_{digest}"


def _legacy_project_slug(resolved_root: Path) -> str:
    """Слаг до выравнивания с Node — только чтобы найти прежний кэш."""
    h = hashlib.md5(str(resolved_root).encode()).hexdigest()[:8]
    return f"{resolved_root.name or '_global'}_{h}"


def _string_list(table: dict, key: str, where: str) -> list[str]:
    value = table.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        # Строка вместо списка раньше перебиралась посимвольно:
        # dirs = "build" игнорировал каталоги b, u, i, l и d.
        logger.warning("%s: %s must be a list of strings, ignored", where, key)
        return []
    items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(items) != len(value):
        logger.warning("%s: non-string entries in %s ignored", where, key)
    return items


def _table(data: dict, key: str, where: str) -> dict:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        logger.warning("%s: [%s] must be a table, ignored", where, key)
        return {}
    return value


@dataclass
class ProjectConfig:
    """Per-project overrides loaded from ``<project_root>/.py-utils.toml``.

    Empty when the file is absent or unparseable — every subsystem falls
    back to its own defaults so the file is purely additive.
    """

    extra_ignore_dirs: list[str] = field(default_factory=list)
    extra_extensions: list[str] = field(default_factory=list)
    max_file_bytes: Optional[int] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, project_root: Path) -> "ProjectConfig":
        toml_path = project_root / PROJECT_CONFIG_FILENAME
        if not toml_path.is_file():
            return cls()
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", toml_path, exc)
            return cls()

        # Каждое поле проверяется отдельно: неверный тип одного из них раньше
        # ронял Config целиком (AttributeError, ValueError), и вместе с ним
        # поиск и граф проекта — из-за одной опечатки в необязательном файле.
        where = str(toml_path)
        ignore = _table(data, "ignore", where)
        extensions = _table(data, "extensions", where)
        graph = _table(data, "graph", where)

        max_file_bytes = graph.get("max_file_bytes")
        if max_file_bytes is not None and (
            isinstance(max_file_bytes, bool)
            or not isinstance(max_file_bytes, int)
            or not 0 < max_file_bytes <= _MAX_FILE_BYTES_LIMIT
        ):
            logger.warning(
                "%s: graph.max_file_bytes must be an integer from 1 to %d, ignored",
                where, _MAX_FILE_BYTES_LIMIT,
            )
            max_file_bytes = None

        return cls(
            extra_ignore_dirs=_string_list(ignore, "dirs", where),
            extra_extensions=_string_list(extensions, "extra", where),
            max_file_bytes=max_file_bytes,
            raw=data,
        )


@dataclass
class Config:
    project_root: Path
    storage_root: Path = field(default_factory=lambda: DEFAULT_STORAGE)
    project: ProjectConfig = field(default_factory=ProjectConfig)

    def __post_init__(self) -> None:
        # Слаг считается от пути как его назвали, до resolve(): так же, как
        # у Node-воркера, иначе регистр и junction разводят каталоги.
        self._slug = project_slug_for(self.project_root)
        self._legacy_checked = False
        self.project_root = Path(self.project_root).resolve()
        self.storage_root = Path(self.storage_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        # Re-load the project config now that the path is resolved. Callers
        # who passed an explicit ``project=`` win — we only auto-load when
        # the default factory left it empty.
        if self.project == ProjectConfig():
            self.project = ProjectConfig.load(self.project_root)

    @property
    def project_slug(self) -> str:
        return self._slug

    @property
    def project_dir(self) -> Path:
        d = self.storage_root / "projects" / self.project_slug
        if not self._legacy_checked:
            # Один раз на Config: каталог нового слага Node обычно уже завёл
            # (граф, правила), а плотного индекса в нём ещё нет.
            self._legacy_checked = True
            if not (d / "retriever").exists():
                self._adopt_legacy_retriever_cache(d)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _adopt_legacy_retriever_cache(self, target: Path) -> None:
        """Забрать плотный индекс из каталога со старым слагом.

        Его пересборка на CPU идёт часами, а граф и остальное Node всегда
        писал уже в новый каталог — переносить больше нечего.
        """
        legacy = self.storage_root / "projects" / _legacy_project_slug(self.project_root)
        source = legacy / "retriever"
        if legacy == target or not source.is_dir():
            return
        try:
            target.mkdir(parents=True, exist_ok=True)
            source.rename(target / "retriever")
            logger.info("Moved retriever cache %s -> %s", source, target)
        except OSError as exc:
            logger.warning("Could not move retriever cache from %s: %s", legacy, exc)

    @property
    def graph_dir(self) -> Path:
        d = self.project_dir / "graph"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def retriever_cache_dir(self) -> Path:
        d = self.project_dir / "retriever"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def memory_dir(self) -> Path:
        d = self.project_dir / "memory"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def models_dir(self) -> Path:
        d = self.storage_root / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d
