"""Недоверенный ввод не роняет индексацию: ответ модели, файлы проекта, гонки с диском."""

import asyncio
import sys
import types

import pytest

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _fake = types.ModuleType("sentence_transformers")
    _fake.SentenceTransformer = object
    _fake.CrossEncoder = object
    sys.modules["sentence_transformers"] = _fake

from py_utils.config import ProjectConfig
from py_utils.llm.extractor import OpenAICompatExtractor

graph_module = pytest.importorskip("py_utils.core.graph")


@pytest.mark.parametrize("reply", ["[]", "null", '"ok"', "42", '{"entities": 5, "relations": "x"}'])
def test_extract_returns_empty_graph_for_non_graph_json(reply, monkeypatch):
    # Модель вернула валидный JSON, но не объект графа. Раньше extract()
    # падал на result.get(...) с AttributeError вместо пустого результата.
    extractor = OpenAICompatExtractor(base_url="http://127.0.0.1:9", api_key="test")

    async def fake_complete(prompt, max_tokens=400, temperature=0.0):
        return reply

    monkeypatch.setattr(extractor, "complete", fake_complete)
    result = asyncio.run(extractor.extract("app.py", "def handler():\n    return 1\n"))

    assert result == {"entities": [], "relations": []}
    assert extractor.last_diagnostic


def test_project_config_with_utf8_bom_is_read(tmp_path):
    # PowerShell 5.1 (Out-File -Encoding utf8) и старый Блокнот пишут BOM —
    # tomllib на нём падает, и настройки проекта молча пропадали.
    (tmp_path / ".py-utils.toml").write_bytes(
        "﻿[ignore]\ndirs = [\"сгенерировано\"]\n".encode("utf-8")
    )
    config = ProjectConfig.load(tmp_path)
    assert config.extra_ignore_dirs == ["сгенерировано"]


def test_pending_files_tolerates_file_deleted_mid_scan(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    kept = project / "kept.py"
    kept.write_text("def kept():\n    return 1\n", encoding="utf-8")
    gone = project / "gone.py"
    gone.write_text("def gone():\n    return 1\n", encoding="utf-8")
    graph = graph_module.CodeGraph(project, tmp_path / "graph")
    for path in (kept, gone):
        graph._mark_file_seen(path.name, path.stat().st_mtime)
    scanned = graph._get_files()
    gone.unlink()  # файл удалили между обходом и проверкой свежести
    monkeypatch.setattr(graph, "_get_files", lambda: scanned)

    pending = graph.get_pending_files()

    assert pending["stale"] == ["gone.py"]
    assert pending["unindexed"] == []


def test_visualizer_html_escapes_untrusted_names():
    # Имена и пути в данных визуализации — из чужого кода и ответа модели:
    # «</script>» в имени закрывал встроенный JSON, и дальше шла разметка.
    import json
    import re

    from py_utils.core.visualizer import render_html

    hostile = "</script><script>alert(1)</script>&"
    out = render_html({"name": hostile}, "<b>проект</b>")

    assert "<script>alert(1)" not in out
    assert "<b>проект</b>" not in out
    embedded = re.search(r"const DATA = (.*);", out).group(1)
    assert json.loads(embedded) == {"name": hostile}
