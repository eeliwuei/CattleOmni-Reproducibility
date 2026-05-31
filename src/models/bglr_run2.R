# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
# BGLR ladder fold (WORKDIR, parallel-safe). MODE env: BRR | BayesB_sparse.
# Uses MANUAL beta-hat prediction (mu + Xte%*%b) = honest linear out-of-sample (yHat kept as xcheck).
# Emits per-sample CSV matching holstein_per_sample.csv schema. Usage: Rscript bglr_run2.R WORKDIR
.libPaths("~/Rlibs"); suppressMessages(library(BGLR))
WORK <- commandArgs(trailingOnly=TRUE)[1]
MODE <- Sys.getenv("MODE","BayesB_sparse")
MODEL <- if (MODE=="BRR") "BRR_full164k" else "BayesB_full164k"
mt <- scan(file.path(WORK,"meta.txt"), quiet=TRUE)
n<-mt[1]; p<-mt[2]; ntr<-mt[3]; nte<-mt[4]; tm<-mt[5]; ts<-mt[6]
lab <- strsplit(readLines(file.path(WORK,"label.txt"))," ")[[1]]; fam<-lab[1]; fold<-lab[2]
teids <- scan(file.path(WORK,"teids.txt"), quiet=TRUE)
con<-file(file.path(WORK,"X.bin"),"rb"); X<-readBin(con,"double",n=n*p); close(con)
X<-matrix(X, nrow=n, ncol=p, byrow=TRUE)
con<-file(file.path(WORK,"y.bin"),"rb"); y<-readBin(con,"double",n=n); close(con)
con<-file(file.path(WORK,"ytrue.bin"),"rb"); ytrue<-readBin(con,"double",n=nte); close(con)
file.remove(file.path(WORK,"X.bin"))
nIter<-as.integer(Sys.getenv("NITER","12000")); burnIn<-as.integer(Sys.getenv("BURNIN","2000"))
if (MODE=="BRR") {
  ETA<-list(list(X=X, model="BRR", R2=0.5))
} else {                      # conventional sparse BayesB (Meuwissen 2001 pi~0.99)
  ETA<-list(list(X=X, model="BayesB", probIn=0.01, counts=1e5, R2=0.5))
}
fm<-BGLR(y=y, ETA=ETA, nIter=nIter, burnIn=burnIn, verbose=FALSE, saveAt=file.path(WORK,"bglr_"))
te<-(ntr+1):n; b<-fm$ETA[[1]]$b; mu<-fm$mu
pred <- as.numeric(mu + X[te,,drop=FALSE] %*% b)   # manual beta-hat (honest linear pred)
pe<-cor(ytrue,pred); pe_yhat<-cor(ytrue,fm$yHat[te]); trainfit<-cor(y[1:ntr],fm$yHat[1:ntr])
yts<-(ytrue-tm)/ts; yps<-(pred-tm)/ts
df<-data.frame(model=MODEL, family=fam, fold=fold, sample_id=teids,
               y_true_raw=ytrue, y_pred_raw=pred, y_true_std=yts, y_pred_std=yps)
write.csv(df, file.path(WORK,"persample.csv"), row.names=FALSE)
write.csv(data.frame(model=MODEL,family=fam,fold=fold,pearson=pe,pearson_yhat=pe_yhat,trainfit=trainfit,ntr=ntr,nte=nte),
          file.path(WORK,"result.csv"), row.names=FALSE)
cat(sprintf("%s %s/%s pearson(manual)=%.4f yHat=%.4f trainfit=%.4f (nIter=%d)\n",
            MODEL, fam, fold, pe, pe_yhat, trainfit, nIter))
