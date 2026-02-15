import sqlite3
import unittest

from docagent.index import sqlite_store
from docagent.index.retriever import _fts_query


class TestFtsQuery(unittest.TestCase):
    def test_fts_query_sanitizes_punctuation(self):
        q = _fts_query("What is attachment theory? (define) secure base!")
        self.assertNotIn("?", q)
        self.assertNotIn("(", q)
        self.assertTrue(q)

    def test_fts_search_does_not_crash_on_user_text(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        sqlite_store.init_db(conn)

        doc_id, changed = sqlite_store.upsert_document(
            conn,
            source_type="md",
            path="notes.md",
            title="notes",
            sha256="deadbeef",
            metadata={},
        )
        self.assertTrue(changed)

        sqlite_store.insert_chunks(
            conn,
            [
                {
                    "doc_id": doc_id,
                    "source_ref": "md:notes.md#L1",
                    "heading": "Test",
                    "page_start": None,
                    "page_end": None,
                    "text": "Attachment theory proposes early caregiver relationships shape later relationships.",
                    "metadata": {},
                }
            ],
        )
        conn.commit()
        sqlite_store.rebuild_fts(conn)

        hits = sqlite_store.fts_search(conn, _fts_query("What is attachment theory?"), limit=5)
        self.assertTrue(len(hits) >= 1)


if __name__ == "__main__":
    unittest.main()

