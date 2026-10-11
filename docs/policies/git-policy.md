# Git and Commit Policy

This policy is mandatory for any agent that modifies repository files, creates commits, changes Git history, or prepares a pull request. Read `AGENTS.md` first. Review-only and Q&A agents remain read-only and must not create commits.

## Commit granularity

- Create one commit per meaningful and logically independent feature, fix, refactor, test change, or documentation change.
- Do not combine unrelated changes into one large commit.
- Do not create meaningless, empty, or artificially fragmented commits solely to increase commit count.
- A commit should represent a unit of work that can be understood and reverted independently.

## Before committing

Before every commit:

1. Inspect `git status`.
2. Inspect the diff.
3. Confirm that only intended files are staged.
4. Run relevant tests, checks, linting, or validation.
5. Verify that no patient data, credentials, secrets, `.env` files, temporary files, or ignored private files are staged.
6. Verify compliance with `AGENTS.md` and this policy.
7. Keep unrelated existing changes out of the commit.

Do not commit known-broken work unless explicitly instructed.

## Commit messages

Use concise Conventional Commit-style messages with lowercase descriptions, for example:

```text
feat: add eyelid ROI extraction
fix: preserve subject IDs during preprocessing
test: add manifest validation tests
docs: document segmentation methodology
refactor: simplify canonical image loading
chore: update preprocessing dependencies
```

Avoid vague messages such as:

```text
update files
changes
fix stuff
codex changes
work done
```

## Task-based commits

- If one implementation task produces multiple independent completed changes, commit each verified change separately.
- Do not wait until the end of a large task and place unrelated work into one giant commit.
- Do not split a single trivial edit into multiple artificial commits.

## Agent permissions

- Review-only agents must not create commits.
- Q&A or advisory agents are strictly read-only.
- Implementation agents may create commits for verified changes within their assigned scope.
- Manuscript implementation tasks may commit approved manuscript changes.
- Research-level changes require approval from the main AIMA coordinator before being committed.

Research-level changes include:

- Hb eligibility;
- label definitions;
- inclusion/exclusion rules;
- dataset splitting;
- segmentation methodology;
- evaluation methodology;
- model comparison methodology;
- interpretation of experimental results.

## Research integrity

Never create commits that:

- fabricate Hb values;
- fabricate experiment results;
- change labels to improve performance;
- alter dataset membership without an approved reason;
- modify evaluation methodology merely to improve reported metrics;
- present planned functionality as implemented.

## Sensitive data

Never commit:

- raw patient images;
- ID-reference screenshots;
- CBC records containing identifiable information;
- personally identifiable information;
- API keys;
- credentials;
- `.env` files;
- private local datasets;
- generated patient ROI images or masks unless explicitly approved.

Verify `.gitignore` before performing data-related work.

## Generated files

Do not automatically commit:

- model checkpoints;
- training caches;
- large derived datasets;
- temporary QC outputs;
- generated patient images;
- generated masks;
- temporary CSV exports.

Prefer committing reproducibility artifacts such as:

- source code;
- configuration;
- tests;
- documentation;
- non-sensitive schemas;
- approved summary metrics.

## Existing work

- Never overwrite, revert, discard, amend, squash, or commit another task's unrelated work.
- Do not use force push or rewrite shared Git history unless explicitly instructed.
- For parallel implementation work, prefer separate Git worktrees.
- If another task appears to be editing the same file, report the conflict instead of silently resolving it.

## Smallest necessary change

Agents should:

- make the smallest change that fully solves the assigned task;
- avoid unrelated refactoring;
- inspect existing implementations before replacing them;
- reuse existing utilities where practical;
- avoid unnecessary dependencies;
- explain any new dependency;
- avoid silently changing schemas, filenames, APIs, directory structures, or experiment configuration.

## Tests

When behavior changes:

- add or update relevant tests when practical;
- run relevant checks before committing;
- do not report a task as fully verified if relevant tests were not run.

## Pull requests

Pull requests should include:

- a concise summary;
- validation commands and results;
- affected output files;
- changes to dataset assumptions.

Do not attach raw patient data or credentials.

## After committing

After every commit:

1. Confirm repository status.
2. Record the commit hash.
3. Record the commit message.
4. Record tests and checks performed.
5. State whether any changes remain uncommitted.

Do not automatically push unless the task explicitly allows pushing.

## Final implementation report

At the end of an implementation task, report:

- commits created;
- commit hashes;
- files changed;
- tests performed;
- test results;
- remaining uncommitted changes;
- known limitations;
- recommended follow-up work.

## Decision and implementation status

Agents must distinguish between:

- implemented behavior;
- approved but not yet implemented behavior;
- proposed behavior;
- experimental results;
- general advice.

Never silently convert a proposal into an implementation or research decision. If a question requires a research decision, refer it to the main AIMA coordinator.
