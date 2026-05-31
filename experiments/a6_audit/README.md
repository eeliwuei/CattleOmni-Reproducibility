# A6 audit experiment

Genome-build / coordinate-mapping audit (claim control). See `docs/a6_audit.md` and
`src/audit/a6_stability.py`. Outputs land in `results/a6_audit/`.

This audit is reported as a reliability control: the evidence system prevented a potentially
incorrect biological (DGAT1-level) annotation caused by a genome-build/coordinate mismatch
(source map resolved to Btau_4.6.1, lifted to ARS-UCD1.2). The model-comparison results are
position-independent and unaffected.
