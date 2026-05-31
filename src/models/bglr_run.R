# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
# BGLR BayesB on one prepped fold. readBin X (n x p, row-major float64) + y (NaN=test).
# Predict test, save yHat. Trusted Bayesian variable-selection (sidesteps custom sampler).
.libPaths("~/Rlibs")
suppressMessages(library(BGLR))
mt <- scan("/tmp/bglr_fold_meta.txt", quiet=TRUE)
meta <- list(n=mt[1], p=mt[2], ntr=mt[3], nte=mt[4])
lab <- strsplit(readLines("/tmp/bglr_fold_label.txt")," ")[[1]]
meta$regime <- lab[1]; meta$fold <- lab[2]
n <- meta$n; p <- meta$p
con <- file("/tmp/bglr_fold_X.bin","rb"); X <- readBin(con, what="double", n=n*p); close(con)
X <- matrix(X, nrow=n, ncol=p, byrow=TRUE)
con <- file("/tmp/bglr_fold_y.bin","rb"); y <- readBin(con, what="double", n=n); close(con)
con <- file("/tmp/bglr_fold_ytrue.bin","rb"); ytrue <- readBin(con, what="double", n=meta$nte); close(con)
cat(sprintf("loaded X %d x %d, train %d test %d\n", nrow(X), ncol(X), meta$ntr, meta$nte))
ETA <- list(list(X=X, model="BayesB"))
nIter <- as.integer(Sys.getenv("NITER","6000")); burnIn <- as.integer(Sys.getenv("BURNIN","3000"))
fm <- BGLR(y=y, ETA=ETA, nIter=nIter, burnIn=burnIn, verbose=FALSE, saveAt="/tmp/bglr_")
yHat <- fm$yHat
te_idx <- (meta$ntr+1):n
pred <- yHat[te_idx]
pe <- cor(ytrue, pred)
cat(sprintf("BGLR-BayesB %s/%s  test Pearson=%.4f (nIter=%d burnIn=%d)\n", meta$regime, meta$fold, pe, nIter, burnIn))
write.csv(data.frame(regime=meta$regime, fold=meta$fold, pearson=pe, ntr=meta$ntr, nte=meta$nte),
          "/tmp/bglr_fold_result.csv", row.names=FALSE)
