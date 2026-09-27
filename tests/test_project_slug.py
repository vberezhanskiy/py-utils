"""Каталог проекта в хранилище — общий с Node-воркером, и проверка .py-utils.toml.

Граф строит Node-воркер (``legacyProjectSlug`` в agent-worker), а читает
сайдкар. Слаги расходились для папок с кириллицей, пробелами и точками, и
сайдкар не видел построенного графа. Те же векторы проверяет
agent-worker/src/tools/project-slug.test.ts — меняются только вместе.
"""

import hashlib
import logging
import ntpath
import posixpath
import types

import pytest

from py_utils import config as config_module
from py_utils.config import Config, ProjectConfig, project_slug_for

VECTORS = {
    "win32": [
        ("C:\\Users\\eddy\\paragonProjects\\agent-python", "agent-python_f7b65a99"),
        ("C:/Users/eddy/paragonProjects/Мой проект/", "___________1181cd90"),
        ("D:\\work\\site.ru", "site_ru_c27acdc5"),
        ("C:\\Users\\eddy\\x\\..\\y", "y_b79d74a5"),
        ("C:\\", "_2f158c47"),
    ],
    "posix": [
        ("/home/eddy/paragonProjects/agent-python", "agent-python_71e68cf5"),
        ("/home/eddy/paragonProjects/Мой проект/", "___________c85da437"),
        ("/srv/work/site.ru", "site_ru_81b35c96"),
        ("/home/eddy/x/../y", "y_befa1c6f"),
        ("/", "_6666cd76"),
    ],
}


@pytest.mark.parametrize("flavour, pathmod", [("win32", ntpath), ("posix", posixpath)])
def test_слаг_совпадает_с_node(monkeypatch, flavour, pathmod):
    # os.path подменяется, чтобы обе таблицы проверялись на любой ОС.
    monkeypatch.setattr(config_module, "os", types.SimpleNamespace(path=pathmod))
    for root, expected in VECTORS[flavour]:
        assert project_slug_for(root) == expected, root


def test_config_берёт_слаг_от_пути_как_его_назвали(tmp_path):
    project = tmp_path / "Мой проект"
    project.mkdir()
    cfg = Config(project_root=project, storage_root=tmp_path / "store")
    assert cfg.project_slug == project_slug_for(project)
    assert cfg.graph_dir.parent.name == project_slug_for(project)


def test_плотный_индекс_переезжает_из_каталога_со_старым_слагом(tmp_path):
    project = tmp_path / "site.ru"
    project.mkdir()
    store = tmp_path / "store"
    resolved = project.resolve()
    legacy = store / "projects" / f"{resolved.name}_{hashlib.md5(str(resolved).encode()).hexdigest()[:8]}"
    (legacy / "retriever").mkdir(parents=True)
    (legacy / "retriever" / "cache.db").write_text("index", encoding="utf-8")

    cfg = Config(project_root=project, storage_root=store)
    assert (cfg.retriever_cache_dir / "cache.db").read_text(encoding="utf-8") == "index"
    assert not (legacy / "retriever").exists()


def _load(tmp_path, text):
    (tmp_path / ".py-utils.toml").write_text(text, encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def test_верный_toml_читается(tmp_path):
    loaded = _load(tmp_path, '[ignore]\ndirs = ["gen"]\n[extensions]\nextra = ["*.foo"]\n[graph]\nmax_file_bytes = 1048576\n')
    assert loaded.extra_ignore_dirs == ["gen"]
    assert loaded.extra_extensions == ["*.foo"]
    assert loaded.max_file_bytes == 1048576


@pytest.mark.parametrize("text", [
    'ignore = "build"\n',                       # таблица строкой
    '[ignore]\ndirs = "build"\n',               # список строкой: раньше b, u, i, l, d
    '[graph]\nmax_file_bytes = "10MB"\n',       # ValueError ронял Config
    '[graph]\nmax_file_bytes = -1\n',           # отрицательный — пропуск всех файлов
    '[graph]\nmax_file_bytes = true\n',
    '[extensions]\nextra = [1, 2]\n',
])
def test_неверный_toml_не_роняет_config(tmp_path, caplog, text):
    with caplog.at_level(logging.WARNING):
        loaded = _load(tmp_path, text)
    assert loaded.extra_ignore_dirs == []
    assert loaded.extra_extensions == []
    assert loaded.max_file_bytes is None
    assert caplog.records, "ошибку в файле надо показать в логе, а не проглотить"


def test_явный_projectconfig_только_с_потолком_не_перезаписывается(tmp_path):
    (tmp_path / ".py-utils.toml").write_text("[graph]\nmax_file_bytes = 5\n", encoding="utf-8")
    cfg = Config(project_root=tmp_path, storage_root=tmp_path / "store", project=ProjectConfig(max_file_bytes=1024))
    assert cfg.project.max_file_bytes == 1024
