#!/usr/bin/env python3
"""Career-normalized citation benchmarking against institutional cohorts.

Pulls citation counts and h-indices from OpenAlex (Google Scholar has no API and
blocks automated access), builds Neuroscience/Psychology cohorts at Vanderbilt,
peer institutions and top institutions, and places a focal author within each.

Metric (see README.md):
    h_c = sqrt( h * sqrt(C) / 2 )      composite h: geometric mean of h and sqrt(C)/2
    CNI = h_c / A                      career-normalized impact, A = career age (years)

Usage:
    python citation_metrics.py --config config.json --out results/
    python citation_metrics.py --resolve-only     # just show focal-author candidates

Env:
    OPENALEX_API_KEY   optional API key (higher rate limits)
    OPENALEX_MAILTO    optional contact email for OpenAlex's polite pool
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.openalex.org"
AUTHOR_FIELDS = ",".join([
    "id", "display_name", "works_count", "cited_by_count", "summary_stats",
    "topics", "last_known_institutions",
])
THIS_YEAR = dt.date.today().year


# --------------------------------------------------------------------------- metrics

def h_index(citation_counts):
    """Hirsch h-index from a list of per-paper citation counts."""
    counts = sorted((c for c in citation_counts if c), reverse=True)
    h = 0
    for i, c in enumerate(counts, start=1):
        if c >= i:
            h = i
        else:
            break
    return h


def first_pub_year(year_counts, window=3, min_works=2):
    """Earliest year that starts a sustained publication record.

    Returns the earliest year Y with at least `min_works` works in [Y, Y+window].
    This ignores stray misattributed papers decades before a real career starts,
    a common artifact of automated author disambiguation.
    """
    years = sorted((int(y), n) for y, n in year_counts.items() if n and str(y).isdigit())
    years = [(y, n) for y, n in years if 1900 < y <= THIS_YEAR]
    for y, _ in years:
        if sum(n for yy, n in years if y <= yy <= y + window) >= min_works:
            return y
    return years[0][0] if years else None


def career_age(first_year, current_year=THIS_YEAR):
    return None if first_year is None else max(1, current_year - first_year + 1)


def composite_h(h, citations):
    """Geometric mean of h and sqrt(C)/2. Equals h when C = 4*h^2 (Hirsch's typical a)."""
    if not h or not citations:
        return 0.0
    return math.sqrt(h * math.sqrt(citations) / 2.0)


def metrics(h, citations, age):
    hc = composite_h(h, citations)
    return {
        "h_index": h,
        "citations": citations,
        "career_age": age,
        "m_quotient": h / age if age else None,
        "citations_per_year": citations / age if age else None,
        "h_composite": hc,
        "CNI": hc / age if age else None,
    }


def percentile_rank(value, population):
    """Share of population strictly below value plus half of ties (0-100)."""
    pop = [p for p in population if p is not None]
    if value is None or not pop:
        return None
    below = sum(p < value for p in pop)
    ties = sum(p == value for p in pop)
    return 100.0 * (below + 0.5 * ties) / len(pop)


def describe(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    q = statistics.quantiles(v, n=4) if len(v) >= 2 else [v[0]] * 3
    return {"n": len(v), "median": statistics.median(v), "q1": q[0], "q3": q[2],
            "p90": statistics.quantiles(v, n=10)[-1] if len(v) >= 2 else v[0]}


def field_share(topics, fields):
    """Share of an author's topic counts falling in the target OpenAlex fields."""
    total = sum(t.get("count", 0) for t in topics or [])
    if not total:
        return 0.0
    hit = sum(t.get("count", 0) for t in topics
              if (t.get("field") or {}).get("display_name") in fields)
    return hit / total


def norm_name(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", " ").split())


# --------------------------------------------------------------------------- OpenAlex client

class OpenAlex:
    def __init__(self, cache_dir, delay=0.12):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.extra = {}
        if os.environ.get("OPENALEX_API_KEY"):
            self.extra["api_key"] = os.environ["OPENALEX_API_KEY"]
        if os.environ.get("OPENALEX_MAILTO"):
            self.extra["mailto"] = os.environ["OPENALEX_MAILTO"]

    def get(self, path, **params):
        params = {k: v for k, v in params.items() if v is not None}
        key_params = dict(sorted(params.items()))
        key = hashlib.sha1(f"{path}?{urllib.parse.urlencode(key_params)}".encode()).hexdigest()
        cached = self.cache / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text())
        url = f"{API}{path}?{urllib.parse.urlencode({**params, **self.extra})}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise
            except urllib.error.URLError:
                if attempt < 4:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise
        time.sleep(self.delay)
        cached.write_text(json.dumps(data))
        return data

    def paged(self, path, **params):
        cursor = "*"
        while cursor:
            page = self.get(path, cursor=cursor, **params)
            yield from page.get("results", [])
            cursor = page.get("meta", {}).get("next_cursor")

    def institution(self, name):
        res = self.get("/institutions", search=name, filter="country_code:US", per_page=5)["results"]
        if not res:
            raise SystemExit(f"OpenAlex has no US institution matching {name!r}")
        exact = [r for r in res if norm_name(r["display_name"]) == norm_name(name)]
        return (exact or res)[0]

    def authors_at(self, inst_id, min_works):
        return self.paged("/authors", per_page=200, select=AUTHOR_FIELDS,
                          filter=f"last_known_institutions.id:{inst_id},works_count:>{min_works - 1}")

    def year_counts(self, author_id):
        g = self.get("/works", filter=f"authorships.author.id:{author_id}",
                     group_by="publication_year")["group_by"]
        return {x["key"]: x["count"] for x in g}

    def works(self, author_id):
        return self.paged("/works", per_page=200, filter=f"authorships.author.id:{author_id}",
                          select="id,publication_year,cited_by_count,title")

    def search_authors(self, name, inst_id=None):
        f = f"last_known_institutions.id:{inst_id}" if inst_id else None
        return self.get("/authors", search=name, filter=f, per_page=10,
                        select=AUTHOR_FIELDS)["results"]


def short_id(openalex_url):
    return openalex_url.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- pipeline

def resolve_focal(oa, cfg):
    fa = cfg["focal_author"]
    if fa.get("openalex_ids"):
        return [short_id(i) for i in fa["openalex_ids"]], []
    inst = oa.institution(fa["institution"])
    candidates, seen = [], set()
    for name in fa["search_names"]:
        for scope in (inst["id"], None):
            for a in oa.search_authors(name, short_id(scope) if scope else None):
                if a["id"] not in seen:
                    seen.add(a["id"])
                    candidates.append(a)
    target = norm_name(fa["name"]).split()
    surname = target[-1]
    keep = [a for a in candidates
            if surname in norm_name(a["display_name"]).split()
            and norm_name(a["display_name"]).split()[0][:1] == target[0][:1]
            and field_share(a.get("topics"), cfg["fields"]) >= cfg["min_field_share"]]
    return [short_id(a["id"]) for a in keep], candidates


def focal_metrics(oa, ids):
    """Exact h and C from the union of works across all (possibly split) OpenAlex IDs."""
    works, years = {}, {}
    for aid in ids:
        for w in oa.works(aid):
            works[w["id"]] = w
    for w in works.values():
        y = w.get("publication_year")
        if y:
            years[str(y)] = years.get(str(y), 0) + 1
    cites = [w.get("cited_by_count") or 0 for w in works.values()]
    fy = first_pub_year(years)
    m = metrics(h_index(cites), sum(cites), career_age(fy))
    m.update({"first_pub_year": fy, "works_count": len(works), "openalex_ids": ids})
    top = sorted(works.values(), key=lambda w: -(w.get("cited_by_count") or 0))[:10]
    m["top_works"] = [{"title": w.get("title"), "year": w.get("publication_year"),
                       "citations": w.get("cited_by_count")} for w in top]
    return m


def build_cohort(oa, cfg, group, entries, exclude_ids):
    rows = []
    for entry in entries:
        inst = ({"id": entry["openalex_id"], "display_name": entry["name"]}
                if entry.get("openalex_id") else oa.institution(entry["name"]))
        iid = short_id(inst["id"])
        print(f"  [{group}] {inst['display_name']} ({iid})", file=sys.stderr)
        for a in oa.authors_at(iid, cfg["min_works"]):
            aid = short_id(a["id"])
            if aid in exclude_ids:
                continue
            share = field_share(a.get("topics"), cfg["fields"])
            if share < cfg["min_field_share"]:
                continue
            fy = first_pub_year(oa.year_counts(aid))
            age = career_age(fy)
            if age is None or age < cfg["min_career_age"]:
                continue
            m = metrics(a["summary_stats"]["h_index"], a["cited_by_count"], age)
            rows.append({"group": group, "institution": inst["display_name"],
                         "openalex_id": aid, "name": a["display_name"],
                         "works_count": a["works_count"], "first_pub_year": fy,
                         "field_share": round(share, 2), **m})
    # One person can be "last known" at two listed institutions (e.g. university + med center).
    dedup = {}
    for r in rows:
        dedup.setdefault(r["openalex_id"], r)
    return list(dedup.values())


def build_roster(oa, cfg, label, people):
    rows = []
    for p in people:
        if p.get("openalex_id"):
            a = oa.get(f"/authors/{short_id(p['openalex_id'])}", select=AUTHOR_FIELDS)
        else:
            inst = oa.institution(p["institution"])
            hits = oa.search_authors(p["name"], short_id(inst["id"]))
            if not hits:
                print(f"  [{label}] no OpenAlex match for {p['name']}", file=sys.stderr)
                continue
            a = max(hits, key=lambda x: x["cited_by_count"])
        aid = short_id(a["id"])
        fy = first_pub_year(oa.year_counts(aid))
        m = metrics(a["summary_stats"]["h_index"], a["cited_by_count"], career_age(fy))
        rows.append({"group": label, "institution": p.get("institution", ""),
                     "openalex_id": aid, "name": a["display_name"],
                     "works_count": a["works_count"], "first_pub_year": fy,
                     "field_share": round(field_share(a.get("topics"), cfg["fields"]), 2), **m})
    return rows


METRIC_KEYS = ["h_index", "citations", "m_quotient", "citations_per_year", "CNI"]


def compare(focal, rows, band):
    out = {}
    age = focal["career_age"]
    matched = [r for r in rows if age and abs(r["career_age"] - age) <= band]
    for k in METRIC_KEYS:
        out[k] = {
            "focal": focal[k],
            "all": describe(r[k] for r in rows),
            "pct_all": percentile_rank(focal[k], [r[k] for r in rows]),
            "age_matched": describe(r[k] for r in matched),
            "pct_age_matched": percentile_rank(focal[k], [r[k] for r in matched]),
        }
    return out


# --------------------------------------------------------------------------- reporting

def fmt(x, nd=2):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:,.{nd}f}" if abs(x) < 100 else f"{x:,.0f}"
    return f"{x:,}"


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_report(path, cfg, focal, comparisons, by_inst, scholar):
    L = []
    L.append(f"# Citation benchmarking: {cfg['focal_author']['name']}\n")
    L.append(f"Source: OpenAlex, retrieved {dt.date.today().isoformat()}. "
             f"Cohorts: authors whose last known affiliation is the listed institution, "
             f"≥{cfg['min_field_share']:.0%} of topics in {' / '.join(cfg['fields'])}, "
             f"≥{cfg['min_works']} works, career age ≥{cfg['min_career_age']} y. "
             f"Age-matched = career age within ±{cfg['career_age_band']} y of the focal author.\n")
    L.append("## Focal author\n")
    L.append("| | h-index | Citations | First pub. | Career age | m = h/A | C/A | h_c | **CNI** |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append(f"| OpenAlex | {focal['h_index']} | {fmt(focal['citations'])} | {focal['first_pub_year']} | "
             f"{focal['career_age']} | {fmt(focal['m_quotient'])} | {fmt(focal['citations_per_year'])} | "
             f"{fmt(focal['h_composite'], 1)} | **{fmt(focal['CNI'])}** |")
    if scholar:
        L.append(f"| Google Scholar (manual) | {scholar['h_index']} | {fmt(scholar['citations'])} | "
                 f"{scholar['first_pub_year']} | {scholar['career_age']} | {fmt(scholar['m_quotient'])} | "
                 f"{fmt(scholar['citations_per_year'])} | {fmt(scholar['h_composite'], 1)} | **{fmt(scholar['CNI'])}** |")
        L.append("\n_Scholar counts run higher than OpenAlex (it indexes preprints, theses, books); "
                 "only the OpenAlex row is comparable to the cohorts below._")
    L.append("")
    L.append("## Percentile of the focal author within each cohort\n")
    L.append("| Cohort | n (all / age-matched) | Metric | Focal | Cohort median [IQR] | Pctl (all) | Age-matched median | Pctl (age-matched) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for group, comp in comparisons.items():
        for k in METRIC_KEYS:
            c = comp[k]
            a, m = c["all"], c["age_matched"]
            L.append(f"| {group} | {a['n']} / {m['n']} | {k} | {fmt(c['focal'])} | "
                     f"{fmt(a.get('median'))} [{fmt(a.get('q1'))}–{fmt(a.get('q3'))}] | "
                     f"{fmt(c['pct_all'], 0)} | {fmt(m.get('median'))} | {fmt(c['pct_age_matched'], 0)} |")
    L.append("")
    L.append("## CNI by institution\n")
    L.append("| Group | Institution | n | Median CNI | 90th pctl CNI | Focal percentile |")
    L.append("|---|---|---|---|---|---|")
    for (group, inst), vals in sorted(by_inst.items()):
        d = describe(vals)
        L.append(f"| {group} | {inst} | {d['n']} | {fmt(d.get('median'))} | {fmt(d.get('p90'))} | "
                 f"{fmt(percentile_rank(focal['CNI'], vals), 0)} |")
    L.append("\n## Focal author: top-cited works (OpenAlex)\n")
    for w in focal["top_works"]:
        L.append(f"- {w['title']} ({w['year']}) — {fmt(w['citations'])}")
    Path(path).write_text("\n".join(L) + "\n")


def plot(path, focal, cohorts):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return
    groups = list(cohorts)
    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(8, 1.1 + 0.9 * len(groups)), dpi=150)
    ink, muted, accent = "#0b0b0b", "#8a8984", "#2a78d6"
    for i, g in enumerate(groups):
        vals = [r["CNI"] for r in cohorts[g] if r["CNI"] is not None]
        if not vals:
            continue
        jitter = [rng.uniform(-0.22, 0.22) for _ in vals]
        ax.scatter(vals, [i + j for j in jitter], s=9, color=muted, alpha=0.45, linewidths=0)
        med = statistics.median(vals)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=ink, lw=2, solid_capstyle="round")
        pct = percentile_rank(focal["CNI"], vals)
        ax.annotate(f"{pct:.0f}th pctl", (focal["CNI"], i - 0.12), xytext=(0, 6),
                    textcoords="offset points", color=ink, fontsize=8, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.scatter([focal["CNI"]] * len(groups), range(len(groups)), s=70, color=accent,
               edgecolors="white", linewidths=2, zorder=3, label=f"{focal['name']}")
    ax.set_yticks(range(len(groups)), [f"{g}\n(n={len(cohorts[g])})" for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel("Career-normalized impact, CNI = √(h·√C / 2) / career age")
    ax.set_xlim(left=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8,
              handles=[ax.collections[-1],
                       plt.Line2D([], [], color=ink, lw=2, label="cohort median")])
    fig.tight_layout()
    fig.savefig(path)


# --------------------------------------------------------------------------- main

def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=here / "config.json")
    ap.add_argument("--out", default=here / "results")
    ap.add_argument("--cache", default=here / ".cache")
    ap.add_argument("--resolve-only", action="store_true",
                    help="print focal-author OpenAlex candidates and exit")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    oa = OpenAlex(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ids, candidates = resolve_focal(oa, cfg)
    if args.resolve_only or not ids:
        print("Focal-author candidates (pin the right ones in config focal_author.openalex_ids):")
        for a in candidates:
            insts = ", ".join(i["display_name"] for i in a.get("last_known_institutions") or [])
            mark = "*" if short_id(a["id"]) in ids else " "
            print(f" {mark} {short_id(a['id'])}  {a['display_name']:<28} works={a['works_count']:<4} "
                  f"cites={a['cited_by_count']:<6} h={a['summary_stats']['h_index']:<3} "
                  f"field={field_share(a.get('topics'), cfg['fields']):.2f}  {insts}")
        if not ids:
            raise SystemExit("No confident focal match; set focal_author.openalex_ids.")
        return

    print(f"Focal OpenAlex IDs: {ids}", file=sys.stderr)
    focal = focal_metrics(oa, ids)
    focal["name"] = cfg["focal_author"]["name"]

    scholar = None
    sm = cfg["focal_author"].get("scholar_manual") or {}
    if sm.get("citations") and sm.get("h_index"):
        fy = sm.get("first_pub_year") or focal["first_pub_year"]
        scholar = {**metrics(sm["h_index"], sm["citations"], career_age(fy)), "first_pub_year": fy}

    cohorts = {}
    for group, entries in cfg["groups"].items():
        cohorts[group] = build_cohort(oa, cfg, group, entries, set(ids))
    for label, people in cfg.get("rosters", {}).items():
        if label.startswith("_") or not people:
            continue
        cohorts[label] = build_roster(oa, cfg, label, people)

    comparisons = {g: compare(focal, rows, cfg["career_age_band"]) for g, rows in cohorts.items()}
    by_inst = {}
    for g, rows in cohorts.items():
        for r in rows:
            by_inst.setdefault((g, r["institution"]), []).append(r["CNI"])

    all_rows = [r for rows in cohorts.values() for r in rows]
    write_csv(out / "cohort_authors.csv", all_rows)
    (out / "focal_author.json").write_text(json.dumps({"openalex": focal, "scholar_manual": scholar}, indent=2))
    (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    write_report(out / "report.md", cfg, focal, comparisons, by_inst, scholar)
    plot(out / "cni_by_cohort.png", focal, cohorts)
    print(f"Wrote {out}/report.md", file=sys.stderr)


if __name__ == "__main__":
    main()
