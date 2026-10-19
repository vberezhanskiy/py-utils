"""Секретный ли файл по имени — тот же точный набор, что в ядре.

Векторы общие: agent-worker/src/secret-file-cases.json. Ядро проверяет
себя по нему же (tools/secret-files.test.ts), поэтому две половины продукта
не разойдутся в том, какой файл беречь. Когда ядро рядом не лежит (py-utils
отдельно), проверяются только встроенные случаи.
"""

import json
import unittest
from pathlib import Path

from py_utils.paths import is_secret_file

CASES = Path(__file__).resolve().parents[2] / "agent-worker" / "src" / "secret-file-cases.json"


class SecretFileNamesTest(unittest.TestCase):
    def test_exact_names_not_substrings(self):
        for name in (".env", ".env.production", "prod.env", "server.pem", "id_rsa", "credentials.json"):
            self.assertTrue(is_secret_file(f"a/{name}"), name)
        # Подстрока больше не делает файл секретным.
        for name in (".env.example", "credentials.ts", "secrets-panel.tsx", "config.keymap.json", "id_rsa.pub"):
            self.assertFalse(is_secret_file(f"a/{name}"), name)

    @unittest.skipUnless(CASES.is_file(), "ядро не лежит рядом")
    def test_shared_vectors_with_core(self):
        cases = json.loads(CASES.read_text(encoding="utf-8"))
        for name in cases["secret"]:
            self.assertTrue(is_secret_file(f"src/{name}"), name)
        for name in cases["notSecret"]:
            self.assertFalse(is_secret_file(f"src/{name}"), name)


if __name__ == "__main__":
    unittest.main()
