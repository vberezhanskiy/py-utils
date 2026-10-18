"""Соединение с базой графа — одно на всех читателей и писателей.

Отдельный модуль, а не функция в graph.py: миксины поиска и анализа и
визуализатор импортируются самим graph.py, и взять соединение оттуда они не
могут. Они открывали базу голым ``sqlite3.connect`` — без WAL, с пятисекундным
ожиданием и без закрытия, то есть ровно с теми дефектами, от которых граф
уже лечили.
"""

import sqlite3


_SQLITE_BUSY_TIMEOUT_MS = 30_000


class _ClosingConnection(sqlite3.Connection):
    """``with`` завершает транзакцию И закрывает соединение.

    У обычного sqlite3.Connection выход из ``with`` только фиксирует или
    откатывает транзакцию, а соединение остаётся открытым до сборщика мусора.
    Сайдкар живёт всё время работы приложения, и каждый запрос к графу
    оставлял за собой открытый файл базы и WAL.
    """

    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def _connect(db_path) -> sqlite3.Connection:
    """Соединение с графом; ``with _connect(...)`` закрывает его на выходе.

    WAL и busy_timeout не выставлялись нигде, поэтому действовал
    rollback-journal и пятисекундное ожидание: полная пересборка делает VACUUM
    под эксклюзивной блокировкой всей базы, и параллельные чтения падали с
    «database is locked» — интерфейс получал голый 500, а записи билдера
    возвращали модели ошибку, и файл молча оставался неиндексированным.
    """
    con = sqlite3.connect(db_path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000, factory=_ClosingConnection)
    con.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
    try:
        con.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        # Сетевые тома WAL не поддерживают — остаёмся на журнале по умолчанию.
        pass
    return con
