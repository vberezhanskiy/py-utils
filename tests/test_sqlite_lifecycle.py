"""Соединение с графом закрывается на выходе из ``with``, а не сборщиком мусора."""

import sqlite3
import sys
import types

import pytest

# Графу модели не нужны, а модуль эмбеддера импортирует sentence-transformers
# при загрузке — без него в тестовом окружении подставляем пустышку.
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _fake = types.ModuleType("sentence_transformers")
    _fake.SentenceTransformer = object
    _fake.CrossEncoder = object
    sys.modules["sentence_transformers"] = _fake

graph = pytest.importorskip("py_utils.core.graph")


def test_with_фиксирует_и_закрывает(tmp_path):
    db = tmp_path / "graph.db"
    with graph._connect(db) as con:
        con.execute("CREATE TABLE t (x INTEGER)")
        con.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")
    with graph._connect(db) as again:
        assert again.execute("SELECT x FROM t").fetchall() == [(1,)]


def test_ошибка_откатывает_и_тоже_закрывает(tmp_path):
    db = tmp_path / "graph.db"
    with graph._connect(db) as con:
        con.execute("CREATE TABLE t (x INTEGER)")
    with pytest.raises(RuntimeError):
        with graph._connect(db) as con:
            con.execute("INSERT INTO t VALUES (2)")
            raise RuntimeError("stop")
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")
    with graph._connect(db) as again:
        assert again.execute("SELECT count(*) FROM t").fetchone() == (0,)
