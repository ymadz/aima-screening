# AIMA research records

This directory is the centralized thesis-traceability layer. The core project
log remains the implementation chronology; these records connect experiments,
private evidence bundles, verified aggregate results, and manuscript-ready
language.

| Record | Purpose |
| --- | --- |
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | Chronological research milestones, findings, decisions, limitations, and next steps. |
| [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) | One status-bearing entry per experiment or major analysis, plus consolidated comparisons. |
| [`METHODOLOGY.md`](METHODOLOGY.md) | Actual implemented methods, parameter values, splits, thresholds, and planned-method boundaries. |
| [`RESULTS.md`](RESULTS.md) | Verified performance tables, confusion counts, uncertainty, segmentation side-results, and figures. |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Scientific limitations, negative evidence, hypotheses, and external-validation requirements. |
| [`THESIS_HANDOFF.md`](THESIS_HANDOFF.md) | Chapter-organized manuscript handoff with source paths for quantitative claims. |
| [`RESEARCH_DOCUMENTATION_GUIDE.md`](RESEARCH_DOCUMENTATION_GUIDE.md) | Required workflow for every future experiment and documentation update. |

Run the lightweight read-only consistency check from the repository root:

```bash
python scripts/validate_research_docs.py
```

The validator checks central documents, registry statuses/IDs, and the
existence of the non-sensitive report/metric manifests named by the registry.
It does not read or export patient-level tables and it does not train models.

Experiment5 is now completed and verified: [aggregate results and methods](../experiments/eyelid-exp05-results-2026-10-13.md). Use `python scripts/validate_research_docs.py --source-root /Users/madz/Documents/aima-screening` from this isolated worktree to resolve the read-only private artifact root.
