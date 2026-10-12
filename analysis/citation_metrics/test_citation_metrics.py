"""Offline tests: metric math plus an end-to-end run against a stubbed OpenAlex.

All author data below is synthetic and exists only to exercise code paths.
Run: python -m unittest test_citation_metrics -v
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import citation_metrics as cm


class MetricTests(unittest.TestCase):
    def test_h_index(self):
        self.assertEqual(cm.h_index([10, 8, 5, 4, 3]), 4)
        self.assertEqual(cm.h_index([25, 8, 5, 3, 3]), 3)
        self.assertEqual(cm.h_index([0, 0]), 0)
        self.assertEqual(cm.h_index([]), 0)

    def test_first_pub_year_skips_stray_early_paper(self):
        counts = {"1985": 1, "2009": 1, "2010": 2, "2011": 3}
        self.assertEqual(cm.first_pub_year(counts), 2009)

    def test_composite_h_equals_h_at_typical_ratio(self):
        # Hirsch: C ~ a*h^2 with a ~ 3-5; at a = 4, h_c reduces to h.
        self.assertAlmostEqual(cm.composite_h(30, 4 * 30 ** 2), 30.0)
        self.assertGreater(cm.composite_h(30, 10 * 30 ** 2), 30.0)

    def test_cni(self):
        m = cm.metrics(20, 1600, 10)
        self.assertAlmostEqual(m["CNI"], 2.0)
        self.assertAlmostEqual(m["m_quotient"], 2.0)

    def test_percentile_rank(self):
        self.assertEqual(cm.percentile_rank(5, [1, 2, 3, 4]), 100.0)
        self.assertEqual(cm.percentile_rank(2, [1, 2, 3, 4]), 37.5)

    def test_field_share(self):
        topics = [{"count": 6, "field": {"display_name": "Neuroscience"}},
                  {"count": 4, "field": {"display_name": "Medicine"}}]
        self.assertAlmostEqual(cm.field_share(topics, ["Neuroscience", "Psychology"]), 0.6)


def _author(aid, name, h, c, works, field="Neuroscience"):
    return {"id": f"https://openalex.org/{aid}", "display_name": name, "works_count": works,
            "cited_by_count": c, "summary_stats": {"h_index": h},
            "topics": [{"count": 10, "field": {"display_name": field}}],
            "last_known_institutions": []}


class StubOpenAlex(cm.OpenAlex):
    """Serves synthetic responses for the endpoints the pipeline calls."""

    INST = {"Vanderbilt University": "I1", "Peer U": "I2", "Top U": "I3"}
    AUTHORS = {
        "I1": [_author("A10", "Colleague One", 40, 8000, 120),
               _author("A11", "Colleague Two", 15, 900, 40),
               _author("A12", "Off Field", 50, 20000, 200, field="Chemistry")],
        "I2": [_author("A20", "Peer One", 25, 3000, 60)],
        "I3": [_author("A30", "Top One", 60, 25000, 150), _author("A31", "Top Two", 35, 6000, 80)],
    }
    FIRST = {"A10": 1998, "A11": 2012, "A12": 1990, "A20": 2006, "A30": 2000, "A31": 2008, "A1": 2008}

    def __init__(self):
        self.extra = {}

    def get(self, path, **p):
        if path == "/institutions":
            return {"results": [{"id": f"https://openalex.org/{self.INST[p['search']]}",
                                 "display_name": p["search"]}]}
        if path == "/authors" and "search" in p:
            return {"results": [_author("A1", "Andre M. Bastos", 0, 0, 100)]}
        if path == "/authors":
            iid = p["filter"].split(",")[0].split(":")[1]
            return {"results": self.AUTHORS[iid], "meta": {"next_cursor": None}}
        if path == "/works" and "group_by" in p:
            aid = p["filter"].split(":")[1]
            y = self.FIRST[aid]
            return {"group_by": [{"key": str(y + i), "count": 3} for i in range(10)]}
        if path == "/works":
            works = [{"id": f"W{i}", "publication_year": 2008 + i % 15,
                      "cited_by_count": 200 - 2 * i, "title": f"Paper {i}"} for i in range(100)]
            return {"results": works, "meta": {"next_cursor": None}}
        raise AssertionError(path)


class EndToEndTest(unittest.TestCase):
    def test_pipeline_writes_report(self):
        cfg = json.loads((Path(__file__).parent / "config.json").read_text())
        cfg["groups"] = {"Vanderbilt": [{"name": "Vanderbilt University"}],
                         "Peer institutions": [{"name": "Peer U"}],
                         "Top institutions": [{"name": "Top U"}]}
        cfg["focal_author"]["scholar_manual"].update(citations=12000, h_index=40)
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "cfg.json"
            cfg_path.write_text(json.dumps(cfg))
            argv = ["x", "--config", str(cfg_path), "--out", str(Path(d) / "out")]
            with mock.patch.object(cm, "OpenAlex", lambda *_: StubOpenAlex()), \
                 mock.patch.object(sys, "argv", argv):
                cm.main()
            report = (Path(d) / "out" / "report.md").read_text()
            rows = (Path(d) / "out" / "cohort_authors.csv").read_text()
            focal = json.loads((Path(d) / "out" / "focal_author.json").read_text())
        self.assertIn("Percentile of the focal author", report)
        self.assertNotIn("Off Field", rows)          # field filter
        self.assertEqual(focal["openalex"]["h_index"], cm.h_index([200 - 2 * i for i in range(100)]))
        self.assertEqual(focal["openalex"]["first_pub_year"], 2008)
        self.assertTrue(math.isfinite(focal["scholar_manual"]["CNI"]))


if __name__ == "__main__":
    unittest.main()
