"""Загрузка эмбеддера и реранкера: одна копия на процесс, кэш без полуфайлов.

Первые вызовы приходят из пула потоков сайдкара разом, и каждый грузил свою
копию модели, сохраняя её в тот же каталог. Прерванное сохранение оставляло
каталог, из которого модель потом не грузилась никогда. sentence-transformers
подменяется: проверяется порядок загрузки, а не сами модели.
"""

import importlib
import sys
import threading
import time
import types
from pathlib import Path

import pytest


class _FakeModel:
    loads: list = []
    lock = threading.Lock()

    def __init__(self, source, device=None, model_kwargs=None, cache_folder=None):
        if Path(str(source)).is_dir() and (Path(str(source)) / "broken").exists():
            raise OSError("config.json is missing")
        with _FakeModel.lock:
            _FakeModel.loads.append(str(source))
        time.sleep(0.05)  # окно, в котором второй поток успел бы начать свою загрузку
        self.max_seq_length = 512
        self.max_length = 512

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.bin").write_text("weights", encoding="utf-8")

    save_pretrained = save


@pytest.fixture()
def embedder(monkeypatch, tmp_path):
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _FakeModel
    fake.CrossEncoder = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setenv("MCP_RAG_DEVICE", "cpu")
    sys.modules.pop("py_utils.core.embedder", None)
    module = importlib.import_module("py_utils.core.embedder")
    module.configure(tmp_path / "models")
    _FakeModel.loads = []
    yield module
    sys.modules.pop("py_utils.core.embedder", None)


def test_параллельные_первые_вызовы_грузят_модель_один_раз(embedder):
    results = []
    threads = [threading.Thread(target=lambda: results.append(embedder.get_embedder())) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(_FakeModel.loads) == 1
    assert len({id(model) for model in results}) == 1


def test_скачанная_модель_кладётся_в_кэш_целиком(embedder, tmp_path):
    embedder.get_reranker()
    cached = tmp_path / "models" / embedder._model_dir_name(embedder._reranker_model_id())
    assert (cached / "model.bin").is_file()
    assert not list((tmp_path / "models").glob("*.partial-*")), "временный каталог остался"


def test_битый_кэш_скачивается_заново(embedder, tmp_path):
    cached = tmp_path / "models" / embedder._model_dir_name(embedder._embed_model_id())
    cached.mkdir(parents=True)
    (cached / "broken").write_text("", encoding="utf-8")

    embedder.get_embedder()
    assert _FakeModel.loads == [embedder._embed_model_id()], "должна была пойти загрузка по имени модели"
    assert (cached / "model.bin").is_file()
    assert not (cached / "broken").exists()
