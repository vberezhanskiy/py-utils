"""Папки сборки отсекаются и при обходе, и в фильтре выдачи [07 №33].

В списке ретривера были ``CMakeFiles``, ``Debug`` и ``Release``, а фильтр
сравнивал с ним части пути в нижнем регистре: эти каталоги отсекались только
при обходе, а в фильтре выдачи — никогда.
"""

from pathlib import Path

import pytest

pytest.importorskip("gitignore_parser")

from py_utils.core.retriever import MultiLangCodeRetriever  # noqa: E402


def _retriever(root: Path) -> MultiLangCodeRetriever:
    # Без конструктора: он грузит индекс и эмбеддер, а проверяется только фильтр.
    probe = object.__new__(MultiLangCodeRetriever)
    probe.root_dir = root.resolve()
    probe._gitignore_parser = None
    return probe


def test_build_dirs_ignored_in_walk_and_filter(tmp_path):
    retriever = _retriever(tmp_path)
    for folder in ("CMakeFiles", "Debug", "Release", "node_modules", "Node_Modules"):
        assert retriever._should_ignore(Path(folder) / "a.cpp"), folder
        assert retriever._should_ignore_dir(folder), folder
    assert not retriever._should_ignore(Path("src") / "Layout.tsx")
    assert not retriever._should_ignore_dir("src")
