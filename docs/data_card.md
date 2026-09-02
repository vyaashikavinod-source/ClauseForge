# Data card

ClauseForge uses the public Contract Understanding Atticus Dataset (CUAD) as the
authoritative source for 41 contract-clause categories. Users obtain CUAD from
the Atticus Project and remain responsible for reviewing its provenance and
license terms; raw contents are not committed here.

Processing is deterministic where practical: source checksums, content-derived
IDs, authoritative span alignment, structural segmentation, contract-level
splits, manifests, and statistics are recorded. Train, validation, and test
contracts are disjoint. Segments are unlabeled candidates and never replace CUAD
annotations. Legal datasets remain imbalanced and annotation/taxonomy scope does
not cover every legal issue.

SEC EDGAR documents form a separate, unlabeled public OOD pipeline for validity
and distribution analysis; they never enter CUAD training. `demo/` fixtures are
original synthetic wording for interface demonstration only and provide no
performance evidence.

