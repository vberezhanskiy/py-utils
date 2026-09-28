from py_utils.ignore import ProjectIgnore


def test_nested_rules_and_changes_are_live(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    file = nested / "private.py"
    file.write_text("test = 1")
    policy = ProjectIgnore(tmp_path)
    assert not policy(file)
    ignore = nested / ".gitignore"
    ignore.write_text("private.py\n")
    assert policy(file)
    ignore.unlink()
    assert not policy(file)


def test_nested_negation_and_excluded_parent(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    file = nested / "keep.py"
    file.write_text("test = 1")
    (tmp_path / ".gitignore").write_text("*.py\n")
    (nested / ".gitignore").write_text("!keep.py\n")
    assert not ProjectIgnore(tmp_path)(file)
    (tmp_path / ".gitignore").write_text("nested/\n")
    assert ProjectIgnore(tmp_path)(file)
