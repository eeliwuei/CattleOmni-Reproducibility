# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
# Decisive test: is the 0.856 from fm$yHat[NA-rows] an extraction artifact?
# Compare fm$yHat[test] vs manual mu + Xte %*% bHat (posterior-mean marker effects).
.libPaths("~/Rlibs"); suppressMessages(library(BGLR))
W<-"/tmp/bglr_fold"
mt<-scan(paste0(W,"_meta.txt"),quiet=TRUE); n<-mt[1];p<-mt[2];ntr<-mt[3];nte<-mt[4]
con<-file(paste0(W,"_X.bin"),"rb");X<-readBin(con,"double",n=n*p);close(con);X<-matrix(X,n,p,byrow=TRUE)
con<-file(paste0(W,"_y.bin"),"rb");y<-readBin(con,"double",n=n);close(con)
con<-file(paste0(W,"_ytrue.bin"),"rb");ytrue<-readBin(con,"double",n=nte);close(con)
cat(sprintf("loaded %dx%d ntr=%d nte=%d  NA in y=%d\n",n,p,ntr,nte,sum(is.na(y))))
ETA<-list(list(X=X,model="BayesB"))
fm<-BGLR(y=y,ETA=ETA,nIter=2000,burnIn=500,verbose=FALSE,saveAt="/tmp/bglrd_")
te<-(ntr+1):n
yhat_te<-fm$yHat[te]
b<-fm$ETA[[1]]$b; mu<-fm$mu
manual<-as.numeric(mu + X[te,,drop=FALSE] %*% b)
cat(sprintf("cor(ytrue, fm$yHat[test])      = %.4f\n", cor(ytrue,yhat_te)))
cat(sprintf("cor(ytrue, mu + Xte %%*%% bHat)   = %.4f  (<- linear pred; expect ~0.34 if clean)\n", cor(ytrue,manual)))
cat(sprintf("cor(fm$yHat[test], manual)     = %.4f\n", cor(yhat_te,manual)))
cat(sprintf("mu=%.3f  range(yHat_te)=[%.1f,%.1f]  range(manual)=[%.1f,%.1f]\n",
    mu,min(yhat_te),max(yhat_te),min(manual),max(manual)))
# also: train fit corr (sanity)
cat(sprintf("train cor(y, yHat)=%.4f\n", cor(y[1:ntr],fm$yHat[1:ntr])))
