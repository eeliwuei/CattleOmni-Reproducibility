# A6: genome-build and coordinate audit (claim control)

This audit is a reliability control, not a routine experiment.

1. **What it could have caused.** The source Holstein marker map carries coordinates that match
   neither ARS-UCD1.2 nor UMD3.1; a fingerprint resolved the build to Btau_4.6.1 (bosTau7).
   Using the wrong build for position-based annotation would mis-place markers by megabases.
2. **Why the DGAT1 annotation cannot be written casually.** The top multi-trait peak carries a
   DGAT1-level signal but, after explicit liftover (bosTau7 -> ARS-UCD1.2), maps ~1 Mb away from
   DGAT1 with no LD link -- most consistent with a mis-mapped source coordinate. The peak is only
   region-level robust and the gene-level label is build-dependent and unverified.
3. **Build/map mismatch risk.** Position-, locus-, and pathway-level claims are unsafe without a
   verified build and a checked liftover.
4. **How the evidence system intercepts the error.** A claim gate marks position-based claims
   INVALID until the build is resolved and the liftover round-trips; the annotation was stopped.
5. **Relation to the paper's thesis.** *Reliable genomic prediction requires not only protection
   against evaluation leakage, but also protection against genome build, coordinate mapping, and
   biological annotation mismatch.* The model-comparison results (Track A) are position-independent
   and are unaffected by this audit.
