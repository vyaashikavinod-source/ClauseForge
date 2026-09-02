# GPU resume plan

1. Obtain GPU access.
2. Restore and preserve pilot artifacts.
3. Evaluate checkpoint-25 on the historical 128-example validation subset.
4. Compare checkpoint-10 and checkpoint-25 diagnostics.
5. Decide between continuing canonical-question targets or versioning the
   model-facing target representation.
6. If representation changes, create a new experiment family and never reuse
   incompatible adapters.
7. Run a bounded validation pilot.
8. Select configuration using validation only.
9. Perform full training on adequate compute.
10. Run rank selection.
11. Lock the model and configuration.
12. Evaluate held-out test once.
13. Run final safety evaluation.
14. Run SEC EDGAR OOD evaluation.
15. Merge the adapter.
16. Quantize.
17. Validate the artifact and lineage.
18. Benchmark the real serving backend.
19. Complete the release checklist.

Do not overwrite historical artifacts or reinterpret new validation samples as
the original pilot. The held-out test remains sealed through step 11.

