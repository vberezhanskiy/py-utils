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


def test_ожидание_блокировки_и_wal_выставлены(tmp_path):
    with graph._connect(tmp_path / "graph.db") as con:
        assert con.execute("PRAGMA busy_timeout").fetchone() == (30_000,)
        assert con.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_все_модули_открывают_базу_только_через_общий_connect():
    # Миксины поиска и анализа и визуализатор открывали graph.db голым
    # sqlite3.connect: без WAL, с пятисекундным ожиданием и без закрытия.
    # Пока сборка держала запись, их чтения падали с «database is locked».
    from pathlib import Path

    import py_utils

    package = Path(py_utils.__file__).parent
    offenders = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if path.name != "db.py" and "sqlite3.connect(" in path.read_text(encoding="utf-8")
    )
    assert offenders == []
