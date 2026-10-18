"""Пересборка FTS и FAISS не оставляет читателей с пустым или устаревшим индексом.

FTS: принудительная пересборка сначала стирала chunks_fts и фиксировала это,
а потом читала все файлы проекта. Всё это время таблица была пустой, и
Node-воркер, который читает её напрямую, видел «ничего не найдено» — а его
проверка живости кода по пустой выдаче объявляла мёртвыми всех кандидатов.
Два одновременных построения на пустой таблице оба вставляли куски, и
результаты regex-поиска удваивались.

FAISS: две пересборки шли параллельно, и та, что начала раньше со старыми
строками, публиковала индекс позже — свежий индекс молча затирался старым, а
флаг «грязный» был уже снят.
"""

import sys
import threading
import time
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


def _vectors(texts, **_kwargs):
    rows = [[float(len(t) % 7 + 1), float(sum(map(ord, t)) % 11 + 1), 1.0] for t in texts]
    array = np.asarray(rows, dtype="float32")
    return array / np.linalg.norm(array, axis=1, keepdims=True)


@pytest.fixture()
def project_graph(monkeypatch, tmp_path):
    monkeypatch.setattr(graph_module, "encode_documents", _vectors)
    monkeypatch.setattr(graph_module, "encode_query", _vectors)
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("def handler():\n    return 'needle'\n", encoding="utf-8")
    (project / "util.py").write_text("def helper():\n    return 'needle'\n", encoding="utf-8")
    return graph_module.CodeGraph(project, tmp_path / "graph")


def _fts_rows(graph):
    with graph_module._connect(graph.db_path) as con:
        return con.execute("SELECT file, line_start FROM chunks_fts ORDER BY file").fetchall()


def test_forced_fts_rebuild_keeps_old_chunks_visible_until_swap(project_graph, monkeypatch):
    graph = project_graph
    assert graph._ensure_text_index(force=True) == 2
    seen_during_rebuild = []
    original = graph._get_files

    def files_with_probe():
        # Конкурирующий читатель смотрит в базу посреди пересборки.
        seen_during_rebuild.append(len(_fts_rows(graph)))
        return original()

    monkeypatch.setattr(graph, "_get_files", files_with_probe)
    assert graph._ensure_text_index(force=True) == 2

    assert seen_during_rebuild == [2]
    assert len(_fts_rows(graph)) == 2


def test_concurrent_fts_builds_do_not_duplicate_chunks(project_graph, monkeypatch):
    graph = project_graph
    original = graph._get_files
    nested = []

    def files_with_competitor():
        # Пока первое построение читает файлы, второе успевает пройти целиком.
        if not nested:
            nested.append(True)
            graph._ensure_text_index()
        return original()

    monkeypatch.setattr(graph, "_get_files", files_with_competitor)
    graph._ensure_text_index()

    assert _fts_rows(graph) == [("app.py", 1), ("util.py", 1)]
    found = graph.search_regex("needle")
    assert len(found["matches"]) == 2


def _insert_entity(graph, file, name):
    with graph_module._connect(graph.db_path) as con:
        con.execute(
            "INSERT INTO entities(file, name, type, description, snippet) VALUES(?,?,?,?,?)",
            (file, name, "function", "", f"def {name}(): pass"),
        )


def test_older_faiss_rebuild_cannot_overwrite_newer_one(project_graph, monkeypatch):
    graph = project_graph
    _insert_entity(graph, "app.py", "handler")
    first_started = threading.Event()
    release_first = threading.Event()

    def slow_first_encode(texts, **kwargs):
        if threading.current_thread().name == "first":
            first_started.set()
            assert release_first.wait(10)
        return _vectors(texts, **kwargs)

    monkeypatch.setattr(graph_module, "encode_documents", slow_first_encode)

    first = threading.Thread(target=graph._rebuild_faiss, name="first")
    first.start()
    assert first_started.wait(10)
    # Пока первая пересборка кодирует старый набор, в графе появилась сущность.
    _insert_entity(graph, "util.py", "helper")
    second = threading.Thread(target=graph._rebuild_faiss, name="second")
    second.start()
    time.sleep(0.3)
    release_first.set()
    first.join(10)
    second.join(10)

    _, names = graph.faiss_pair
    assert sorted(names) == ["handler", "helper"]


def test_concurrent_ensure_faiss_builds_once(project_graph, monkeypatch):
    graph = project_graph
    _insert_entity(graph, "app.py", "handler")
    calls = []
    gate = threading.Event()

    def counting_encode(texts, **kwargs):
        calls.append(len(texts))
        gate.wait(0.3)
        return _vectors(texts, **kwargs)

    monkeypatch.setattr(graph_module, "encode_documents", counting_encode)
    threads = [threading.Thread(target=graph.ensure_faiss) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert calls == [1]
    assert graph.faiss_names == ["handler"]
