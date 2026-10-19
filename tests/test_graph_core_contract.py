"""Контракт с ядром (agent-worker): граф пишет Node, свежесть считает Python.

Ядро кладёт в ``file_meta.mtime`` значение ``fs.statSync().mtimeMs / 1000``,
Python читает ``stat().st_mtime`` — округления разные, и побитовое сравнение
считало «устаревшим» каждый пятый файл навсегда: сборщик переписывал одни и
те же файлы за деньги до лимита шагов. Ворота входящих рёбер — те же, что
повторяет ядро (``agent-worker/src/tools/graph-contract.ts``): векторы ниже
меняются только вместе с ним.
"""

import shutil
import subprocess
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

graph_module = pytest.importorskip("py_utils.core.graph")

# Те же расширения — в agent-worker/src/tools/graph-contract.ts (INCOMING_GATE_EXTENSIONS).
GATE_VECTORS = sorted([
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".kt",
    ".scala", ".cs", ".cpp", ".cc", ".c", ".h", ".hpp", ".rb", ".php", ".swift", ".html", ".vue",
    ".svelte", ".css", ".scss", ".sass", ".less",
])


def _node_mtime_seconds(path) -> float:
    """Время файла так, как его пишет ядро; без Node — та же формула в Python."""
    node = shutil.which("node")
    if node:
        out = subprocess.run(
            [node, "-e", "process.stdout.write(String(require('fs').statSync(process.argv[1]).mtimeMs / 1000))", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout)
    stat = path.stat()
    return (stat.st_mtime_ns / 1e6) / 1000


def test_incoming_gate_extensions_match_core_vectors():
    assert sorted(graph_module.INCOMING_GATE_EXTENSIONS) == GATE_VECTORS


def test_file_written_by_core_is_not_stale(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    files = []
    for index in range(60):
        path = project / f"m{index}.py"
        path.write_text(f"def f{index}():\n    return {index}\n", encoding="utf-8")
        files.append(path)
    graph = graph_module.CodeGraph(project, tmp_path / "graph")
    with graph_module._connect(graph.db_path) as con:
        for path in files:
            con.execute(
                "INSERT OR REPLACE INTO file_meta(file, mtime, indexed, incoming_done) VALUES(?,?,1,1)",
                (path.name, _node_mtime_seconds(path)),
            )
    stale = [path.name for path in files if graph._file_needs_update(path)]
    assert stale == []
    # Правка файла по-прежнему видна.
    files[0].write_text("def changed():\n    return 0\n", encoding="utf-8")
    import os
    os.utime(files[0], ns=(files[0].stat().st_atime_ns, files[0].stat().st_mtime_ns + 2_000_000_000))
    assert graph._file_needs_update(files[0])
