import unittest

from docagent.graph.extract import extract_entities, norm_entity


class TestGraphExtract(unittest.TestCase):
    def test_extract_entities_finds_proper_nouns(self):
        text = "Carl Jung wrote about Analytical Psychology in New York."
        ents = extract_entities(text)
        names = {v[0] for v in ents.values()}
        self.assertIn("Carl Jung", names)
        self.assertIn("New York", names)

    def test_norm_entity(self):
        self.assertEqual(norm_entity("  New   York "), "new york")


if __name__ == "__main__":
    unittest.main()

