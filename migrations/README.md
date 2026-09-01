# Migrations

This project does not use Alembic for schema migrations, even though
`alembic` is listed in requirements.txt (it isn't currently wired up --
there is no `alembic/versions` directory). Introducing Alembic now, against
a database that may already have been migrated by these hand-rolled scripts
in an unknown subset/order, was judged riskier than documenting what's
actually here. If you want to move to Alembic, `alembic revision
--autogenerate` against a freshly-migrated database is the safe starting
point -- don't try to reconstruct history retroactively.

Instead, every script here is a standalone, **idempotent** Python script:
each one only adds a column/index/constraint if it doesn't already exist
(`add_column_if_not_exists`, `create_index_if_not_exists`, or the
equivalent inline pattern), so re-running any of them against a database
that's already up to date is a safe no-op. That's why running them out of
order is lower-risk than a normal ordered migration chain would be -- but
"lower risk" is not "no risk" (a later script's index can still depend on
an earlier script's column existing), so use the order below.

## Canonical order

Run once, in this order, against a fresh or partially-migrated database:

1. `001_production_hardening.py` -- organization_id + timezone on
   campaigns, unique constraints, indexes, send-attempt tracking fields.
2. `002_constraints_and_multitenancy.py` -- contact/message uniqueness
   constraints, organization_id on suppression_list, FK cascade
   documentation, hot-path indexes.
3. `002_work_claiming.py` -- **canonical** work-claiming migration:
   `claimed_by` / `claimed_at` on contacts, plus the composite
   `(status, next_action_at)` index the work-claim query in
   `followup/work_claiming.py` actually uses.
4. `../migrate_db.py` (repo root, not in this directory) -- semantic
   intelligence + state-machine columns: `last_reply_at`,
   `last_outbound_at`, `buying_stage`, `engagement_score`,
   `semantic_analysis`, `classification_success`,
   `classification_failure_reason`.

## Known duplicate

`002_add_work_claiming.py` adds the same two columns as
`002_work_claiming.py` -- both exist because they were written
independently at different points. It is **not** deleted (a real
deployment may have already run it, and rewriting migration history a
production database may have already applied is riskier than leaving a
confirmed-safe duplicate in place), but it is marked deprecated in its own
docstring and is safe to skip for new deployments: `002_work_claiming.py`
is the canonical one and additionally creates the composite index the
work-claim query needs, which the deprecated file does not.

The "002" prefix collision between the three files in this directory is
intentional-by-accident (two different concerns both landed on "002" at
different times) rather than a real ordering conflict -- all three are
idempotent and safe to run in the order listed above regardless.

## Running

```
python migrations/001_production_hardening.py
python migrations/002_constraints_and_multitenancy.py
python migrations/002_work_claiming.py
python migrate_db.py
```

Safe to re-run the whole sequence any time; every step no-ops on columns/
indexes/constraints that already exist.
