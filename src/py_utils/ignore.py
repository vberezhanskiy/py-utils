"""Live, nested gitignore rules shared by lexical, dense and graph readers."""
from pathlib import Path
import os

from gitignore_parser import rule_from_pattern


class ProjectIgnore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._cache = {}

    def _rules(self, directory: Path):
        file = directory / ".gitignore"
        try:
            # Never follow an ignore file outside the project.
            file.resolve().relative_to(self.root)
            info = file.stat()
            stamp = (info.st_mtime_ns, info.st_size)
            old = self._cache.get(file)
            if old and old[0] == stamp:
                return old[1]
            lines = file.read_text(encoding="utf-8-sig").splitlines()
        except FileNotFoundError:
            self._cache.pop(file, None)
            return []
        except (OSError, ValueError, UnicodeError):
            # An unreadable policy must not silently permit the subtree.
            return [rule_from_pattern("*", base_path=str(directory))]
        try:
            rules = [rule for line in lines if (rule := rule_from_pattern(line, base_path=str(directory)))]
        except Exception:
            rules = [rule_from_pattern("*", base_path=str(directory))]
        self._cache[file] = (stamp, rules)
        return rules

    def __call__(self, value):
        candidate = Path(os.path.abspath(value))
        try:
            relative = candidate.relative_to(self.root)
        except ValueError:
            return True
        rules = []
        parent = self.root
        for index, part in enumerate(relative.parts):
            rules.extend(self._rules(parent))
            parent = parent / part
            is_directory = index < len(relative.parts) - 1 or parent.is_dir()
            target = str(parent) + (os.sep if is_directory else "")
            ignored = False
            for rule in rules:
                if rule.match(target):
                    ignored = not rule.negation
            # Git cannot re-include a child of an excluded directory.
            if ignored:
                return True
        return False
