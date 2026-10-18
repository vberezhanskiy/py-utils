"""Содержимое секретных файлов не уходит наружу через поиск по графу.

Писатели графа (Node-воркер и write_batch здесь) для .env и ключей хранят
только имя и координаты, сниппет пустой. Но читатели досказывали недостающее
сами, прямо с диска: search_entity при пустом сниппете перечитывал файл и
отдавал строки «KEY=значение», а запасной поиск по сырым вхождениям обходил
все файлы проекта, включая .env и credentials.json. Оба пути ведут в
/search_code, то есть в ответ модели.
"""

import os
import sys
import types

import numpy as np
import pytest

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _fake = types.ModuleType("sentence_transformers")
    _fake.SentenceTransformer = object
    _fake.CrossEncoder = object
    sys.modules["sentence_transformers"] = _fake

graph_module = pytest.importorskip("py_utils.core.graph")

SECRET = "hunter2-very-secret"


def _vectors(texts, **_kwargs):
    rows = [[float(len(t) % 7 + 1), float(sum(map(ord, t)) % 11 + 1), 1.0] for t in texts]
    array = np.asarray(rows, dtype="float32")
    return array / np.linalg.norm(array, axis=1, keepdims=True)


@pytest.fixture()
def make_graph(monkeypatch, tmp_path):
    monkeypatch.setattr(graph_module, "encode_documents", _vectors)
    monkeypatch.setattr(graph_module, "encode_query", _vectors)

    def make(project):
        return graph_module.CodeGraph(project, tmp_path / "graph")

    return make


def _insert_entity(graph, file, name, snippet="", line_start=None):
    with graph_module._connect(graph.db_path) as con:
        con.execute(
            "INSERT INTO entities(file, name, type, description, line_start, line_end, snippet) "
            "VALUES(?,?,?,?,?,?,?)",
            (file, name, "variable", "", line_start, line_start, snippet),
        )


def test_search_entity_does_not_reread_secret_file(make_graph, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(f"DB_HOST=localhost\nDB_PASSWORD={SECRET}\n", encoding="utf-8")
    graph = make_graph(project)
    # Так пишет Node-воркер: имя и координаты есть, сниппет вычищен.
    _insert_entity(graph, ".env", "DB_PASSWORD", snippet="", line_start=2)

    results = graph.search_entity("DB_PASSWORD")

    assert [r["name"] for r in results] == ["DB_PASSWORD"]
    assert results[0]["line_start"] == 2
    assert SECRET not in results[0]["snippet"]
    rendered = "\n".join(graph.format_entity_result(results[0]))
    assert SECRET not in rendered


def test_legacy_secret_contents_are_purged_on_open(make_graph, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    graph = make_graph(project)
    # База, собранная до чистки на записи: значение лежит прямо в строке.
    with graph_module._connect(graph.db_path) as con:
        con.execute(
            "INSERT INTO entities(file, name, type, description, line_start, snippet) VALUES(?,?,?,?,?,?)",
            ("config/.env.production", "DB_PASSWORD", "variable", f"password is {SECRET}", 3,
             f"DB_PASSWORD={SECRET}"),
        )
        con.execute(
            "INSERT INTO entities(file, name, type, description, line_start, snippet) VALUES(?,?,?,?,?,?)",
            ("app.py", "handler", "function", "entry point", 1, "def handler():"),
        )

    reopened = make_graph(project)

    with graph_module._connect(reopened.db_path) as con:
        rows = dict(
            ((r[0], r[1]), (r[2], r[3], r[4]))
            for r in con.execute("SELECT file, name, description, snippet, line_start FROM entities")
        )
    assert rows[("config/.env.production", "DB_PASSWORD")] == ("", "", 3)
    assert rows[("app.py", "handler")] == ("entry point", "def handler():", 1)


def test_raw_occurrence_fallback_skips_secret_files(make_graph, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(f"API_TOKEN={SECRET}\n", encoding="utf-8")
    (project / "credentials.json").write_text(f'{{"API_TOKEN": "{SECRET}"}}\n', encoding="utf-8")
    (project / "app.py").write_text("import os\nTOKEN = os.environ['API_TOKEN']\n", encoding="utf-8")
    graph = make_graph(project)

    results = graph.search_entity("API_TOKEN")

    assert {r["file"] for r in results} == {"app.py"}
    assert all(SECRET not in r["snippet"] for r in results)


def test_search_entity_survives_file_replaced_by_outside_symlink(make_graph, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def handler():\n    return 1\n", encoding="utf-8")
    try:
        os.symlink(outside, project / "handler.py")
    except (OSError, NotImplementedError):
        pytest.skip("создание симлинков в этой системе недоступно")
    graph = make_graph(project)
    _insert_entity(graph, "handler.py", "handler")

    # Раньше _find_entity_snippet отдавал {} и вызов падал с KeyError.
    results = graph.search_entity("handler")

    assert [r["name"] for r in results] == ["handler"]
    assert results[0]["snippet"] == ""


def test_structural_search_rejects_path_filter_outside_project(make_graph, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    graph = make_graph(project)
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda name: "ast-grep")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("ast-grep не должен запускаться по пути вне проекта")

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = graph.structural_search("fetch($URL)", path_filter="..")
    assert result["matches"] == []
    assert "outside" in result["warning"]
    result = graph.structural_search("fetch($URL)", path_filter=str(tmp_path))
    assert result["matches"] == []


def test_structural_search_drops_secret_and_outside_matches(make_graph, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    graph = make_graph(project)
    import json
    import shutil
    import subprocess

    records = [
        {"file": str(project / "app.py"), "range": {"start": {"line": 0}, "end": {"line": 0}}, "text": "fetch(url)"},
        {"file": str(project / "credentials.json"), "range": {"start": {"line": 0}, "end": {"line": 0}}, "text": SECRET},
        {"file": str(tmp_path / "elsewhere.py"), "range": {"start": {"line": 0}, "end": {"line": 0}}, "text": "fetch(x)"},
    ]
    monkeypatch.setattr(shutil, "which", lambda name: "ast-grep")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="\n".join(json.dumps(r) for r in records), stderr="",
        ),
    )

    result = graph.structural_search("fetch($URL)")

    assert [m["file"] for m in result["matches"]] == ["app.py"]
