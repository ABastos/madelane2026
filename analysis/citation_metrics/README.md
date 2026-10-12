# Career-normalized citation benchmarking

Places a focal author (default: Andre M. Bastos, Vanderbilt) within Neuroscience/Psychology
cohorts at Vanderbilt, peer institutions, and top institutions, using one metric that
combines total citations and h-index and normalizes by career age.

## Metric

Hirsch (2005) showed that over a career, `h ≈ m·A` (A = years since first paper) and
`C ≈ a·h²`, with `a` typically 3–5. So `√C` and `h` both grow roughly linearly with
career age, and they can be combined on the same scale:

| Symbol | Definition | Meaning |
|---|---|---|
| `h_c` | `√( h · √C / 2 )` | **Composite h**: the geometric mean of h (breadth of consistently cited work) and `√C/2` (total impact, in h-units). It equals `h` exactly when `C = 4h²`, rises above `h` when a few papers are very heavily cited, and falls below when citations are thin. |
| `A` | `current year − first publication year + 1` | Career age. The first year is the earliest year that starts a sustained record (≥2 works within 4 years), so a stray misattributed old paper doesn't inflate A. |
| **`CNI`** | `h_c / A` | **Career-normalized impact**: composite-h points gained per career year, read like Hirsch's m-quotient. Rough guide from Hirsch: m ≈ 1 is a successful scientist, ≈ 2 outstanding, ≈ 3 exceptional. Read it against the cohort percentiles, though, not against those cutoffs. |

The report also gives raw h, C, `m = h/A` and `C/A`, so you can see which component drives CNI.
Each value comes as a percentile within the full cohort and within an **age-matched**
subset (career age within ±5 years of the focal author's). The age-matched subset is the
fairest comparison, because even normalized metrics drift with career stage.

## Data source and why

- **OpenAlex** is used for everyone, including the focal author, so all numbers come from
  the same index. Google Scholar has no API, blocks automated access, and counts run
  higher (it indexes preprints, theses and books). Scholar figures you paste into
  `config.json → focal_author.scholar_manual` are shown on a separate line and never mixed
  into the cohort comparison.
- **Cohorts** are authors whose *last known affiliation* is a listed institution, with at least
  40% of their OpenAlex topics in Neuroscience or Psychology, at least 20 works, and a career age of 8 years or more.
  Vanderbilt includes both Vanderbilt University and VUMC. Peer and top lists are in
  `config.json` and can be edited.
- **Focal author**: works are collected from every matching OpenAlex ID (OpenAlex sometimes
  splits one person into several IDs), de-duplicated, and h and C are computed exactly.

## Running

```bash
# needs network access to api.openalex.org
export OPENALEX_API_KEY=...        # optional, raises rate limits
python citation_metrics.py --resolve-only   # check the focal-author OpenAlex IDs; pin them in config
python citation_metrics.py                  # full run → results/
pip install matplotlib                      # optional, for results/cni_by_cohort.png
python -m unittest test_citation_metrics    # offline tests (synthetic fixtures)
```

Outputs in `results/`:

- `report.md`: the focal author's row, percentile tables per cohort, and CNI by institution.
- `cohort_authors.csv`: every cohort member with all metrics, for auditing.
- `focal_author.json`: the focal author's metrics and top-cited works.
- `comparisons.json`: the percentile tables in machine-readable form.
- `cni_by_cohort.png`: a strip plot of CNI per cohort, with the focal author highlighted.

Responses are cached in `.cache/`, so re-runs with edited filters are fast.

## Caveats

- **The cohorts approximate departments; they aren't faculty rosters.** "Last known
  affiliation + topic share" sweeps in postdocs, research staff and clinicians, and misses
  faculty whose recent papers list another affiliation. The `career age ≥ 8` and
  `works ≥ 20` floors cut out most trainees. For an exact Vanderbilt Psychology / Neuroscience
  colleague comparison, list names under `rosters` in `config.json`.
- Author disambiguation errors in OpenAlex inflate some h-indices (merged homonyms). Check
  the top of `cohort_authors.csv` for implausible values before quoting percentiles.
- Citation norms differ between subfields (e.g., human neuroimaging vs. cognitive psychology vs.
  systems neurophysiology). Topic-share filtering doesn't remove that.
