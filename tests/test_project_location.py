"""Где лежит сам проект, не влияет на то, что в нём индексируется.

Чёрный список каталогов (build, temp, out, dist, logs, …) сверялся со всеми
частями АБСОЛЮТНОГО пути. Проект в D:\\temp\\app или ~/work/build/site целиком
выпадал из графа и ретривера: обход находил ноль файлов, а search_entity
отфильтровывал все сущности, уже лежащие в базе.
"""

import importlib
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

PARENTS = ["build", "temp", "dist", "logs", "out"]


def _vectors(texts, **_kwargs):
    rows = [[float(len(t) % 7 + 1), float(sum(map(ord, t)) % 11 + 1), 1.0] for t in texts]
    array = np.asarray(rows, dtype="float32")
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def _project(tmp_path, parent):
    project = tmp_path / parent / "site"
    (project / "src").mkdir(parents=True)
    (project / "src" / "app.py").write_text("def handler():\n    return 1\n" * 5, encoding="utf-8")
    (project / "build").mkdir()
    (project / "build" / "bundle.py").write_text("def generated():\n    return 2\n" * 5, encoding="utf-8")
    return project


@pytest.mark.parametrize("parent", PARENTS)
def test_graph_indexes_project_under_listed_folder_name(parent, tmp_path, monkeypatch):
    monkeypatch.setattr(graph_module, "encode_documents", _vectors)
    monkeypatch.setattr(graph_module, "encode_query", _vectors)
    project = _project(tmp_path, parent)
    graph = graph_module.CodeGraph(project, tmp_path / "graph")

    files = [p.relative_to(project).as_posix() for p in graph._get_files()]
    # Внутри проекта список по-прежнему действует.
    assert files == ["src/app.py"]

    with graph_module._connect(graph.db_path) as con:
        con.execute(
            "INSERT INTO entities(file, name, type, description, line_start, snippet) VALUES(?,?,?,?,?,?)",
            ("src/app.py", "handler", "function", "", 1, "def handler():"),
        )
    assert [e["name"] for e in graph.search_entity("handler")] == ["handler"]
    assert graph.get_file_entities("src/app.py")


@pytest.fixture()
def retriever_module(monkeypatch):
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = object
    fake.CrossEncoder = object
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    for name in ("py_utils.core.retriever", "py_utils.core.embedder"):
        sys.modules.pop(name, None)
    module = importlib.import_module("py_utils.core.retriever")
    monkeypatch.setattr(module, "encode_documents", _vectors)
    monkeypatch.setattr(module, "encode_query", _vectors)
    monkeypatch.setattr(module, "rerank_enabled", lambda: False)
    yield module
    for name in ("py_utils.core.retriever", "py_utils.core.embedder"):
        sys.modules.pop(name, None)


@pytest.mark.parametrize("parent", PARENTS + ["Build"])
def test_retriever_indexes_project_under_listed_folder_name(parent, tmp_path, retriever_module):
    project = _project(tmp_path, parent)
    retriever = retriever_module.MultiLangCodeRetriever(project, tmp_path / "cache")
    try:
        files = [p.relative_to(project).as_posix() for p in retriever._iter_indexable_files()]
        assert files == ["src/app.py"]
        assert retriever.chunks, "индекс пустой"
        assert "src/app.py" in retriever.search("handler")
    finally:
        retriever.close()
