"""Плотный индекс ретривера и правки файлов после его загрузки.

Загруженный индекс правок не замечал: поиск отдавал куски удалённых файлов,
пока процесс не перезапустят. И конструктор из поиска уходил в полную сборку
(часы на CPU), если файлы поменялись между проверкой кэша и вызовом.
Эмбеддер подменяется детерминированным: проверяется ключ состояния, а не
качество векторов.
"""

import importlib
import sys
import types

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")
pytest.importorskip("diskcache")
pytest.importorskip("gitignore_parser")


def _vectors(texts, **_kwargs):
    rows = [[float(len(text) % 7 + 1), float(sum(map(ord, text)) % 11 + 1), 1.0] for text in texts]
    array = np.asarray(rows, dtype="float32")
    return array / np.linalg.norm(array, axis=1, keepdims=True)


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


def _write(path, body):
    path.write_text(body * 5, encoding="utf-8")


def test_правка_после_загрузки_видна_как_устаревший_индекс(retriever_module, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _write(project / "app.py", "def handler(request):\n    return request.user\n")
    retriever = retriever_module.MultiLangCodeRetriever(project, tmp_path / "cache")
    assert retriever.cache_key is not None
    assert retriever.is_current()

    _write(project / "app.py", "def other():\n    return 42\n")
    assert not retriever.is_current(), "правка после загрузки не замечена"


def test_без_готового_индекса_конструктор_не_собирает(retriever_module, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _write(project / "app.py", "def handler(request):\n    return request.user\n")
    calls = []
    original = retriever_module.MultiLangCodeRetriever._build_index
    retriever_module.MultiLangCodeRetriever._build_index = lambda self, key=None: calls.append(key)
    try:
        retriever = retriever_module.MultiLangCodeRetriever(project, tmp_path / "cache", build_if_missing=False)
    finally:
        retriever_module.MultiLangCodeRetriever._build_index = original
    assert calls == []
    assert retriever.cache_key is None
    assert not retriever.is_current()


def test_готовый_индекс_загружается_с_ключом(retriever_module, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _write(project / "app.py", "def handler(request):\n    return request.user\n")
    built = retriever_module.MultiLangCodeRetriever(project, tmp_path / "cache")
    key = built.cache_key
    built.cache.close()

    loaded = retriever_module.MultiLangCodeRetriever(project, tmp_path / "cache", build_if_missing=False)
    assert loaded.cache_key == key
    assert loaded.chunks, "индекс из кэша пустой"


def test_ignore_change_invalidates_cache_and_filters_loaded_chunks(retriever_module, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _write(project / "private_config.py", "def private_config():\n    return 123\n")
    r = retriever_module.MultiLangCodeRetriever(project, tmp_path / "cache")
    assert r.is_current()
    (project / ".gitignore").write_text("private_config.py\n", encoding="utf-8")
    assert not r.is_current()
    assert not r._iter_indexable_files()
    assert "private_config.py" not in r.search("private_config")
    r.cache.close()
