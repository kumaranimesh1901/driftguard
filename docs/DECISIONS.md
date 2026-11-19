# Decisions

- Numeric drift and decision defaults are initial values to be calibrated experimentally, not established standards.
- PostgreSQL owns relational integrity and decisions; MongoDB owns append-heavy prediction and detector documents. Shared IDs are the integration contract.
- The synthetic generator is the default source so no dataset is downloaded or redistributed.
- The initial classifier data generator deliberately includes low-importance controls (`num_credit_lines`, `home_ownership`).
