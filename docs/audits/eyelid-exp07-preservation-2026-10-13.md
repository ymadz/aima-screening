# Experiment 7 preservation checkpoint — 2026-10-13

Status: PASS. Experiment 7 may now start. No previous model was fitted, no historical artifact was changed, and no registered private bundle was missing or damaged. All 2,717 files across 14 registered run, readiness, verification and archive directories were hashed and rechecked after validators.

| Experiment | Verified Git handoff | Saved-model reproduction |
|---|---|---|
| 1 | archive/eyelid-exp01-cielab-baseline / 4bf6e25; annotated preservation tag retained | 97 archive entries and ten fold models |
| 2 | experiment/eyelid-exp02-hybrid / f4b2cb5 | 35 pipelines; external source checks |
| 3 | experiment/eyelid-exp03-public-transfer / 66afa8e | 41 neural and 20 LAB checkpoints; fresh prefix maps |
| 4 | experiment/eyelid-exp04-color-optimization / 0b50504 | 720 classification and ten regression pipelines |
| 4B | experiment/eyelid-exp04b-glare-regional / fc64eed | 80 anatomy-only and 160 reviewed-comparison pipelines |
| 5 | experiment/eyelid-exp05-cp-transfer / 0943363 | 122 neural and 60 classical models; all fresh source inputs and both-view maps |
| 6 | experiment/eyelid-exp06-ensemble / 3b74755 | 40 referenced base pipelines, ensemble weights/cutoffs and metrics |

Private consolidated evidence: `data/processed/exp07_preservation_2026-10-13_v1/preservation_manifest.json`. It records exact paths, every file hash, Git refs, all worktree states, canonical source hashes/mtime, validator results and logs. Configurations, manifests, folds, logs, models, OOF predictions, reports, reproduction guides and limitations remain in their original versioned bundles. No patient-level rows appear in this report.

All experiment worktrees and the canonical worktree were clean. The older detached worktree retains four unrelated untracked draft documents unchanged. No merge, rebase, history rewrite or deletion was performed. Canonical main documents contain older implementation status; the verified Experiment 6 branch and experiment-specific reports describe the later evidence. Experiment 7 will use the clean Experiment 6 handoff as its implementation baseline.

Outgoing-history scan covered 45 commits and 238 changed blob/path versions relative to live remote main: no private artifact paths or credential patterns were found. The configured remote is `https://github.com/ymadz/aima-screening.git`; live refs were read and authenticated push dry-run succeeded. Normal branch synchronization is separately recorded in the audit log.

Backup limitation: these sealed bundles and the separate Experiment 1 archive are on the same Macintosh volume. No independent external backup was found; GitHub preserves eligible source history, not ignored images, metadata, annotations or checkpoints. Only about 1.1 GiB was free at this checkpoint. No new private-data copy was created.
