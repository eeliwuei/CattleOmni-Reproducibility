# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
# Test whether proper Bayesian specification fixes the full-164k overfit (vs default BayesB probIn->0.12).
# MODE=BRR (Bayesian ridge, ~GBLUP expected ~0.34) or BayesB_sparse (probIn=0.01 conventional, R2=h2).
# Reports MANUAL beta-hat prediction (mu + Xte%*%b), the honest linear out-of-sample number.
.libPaths("~/Rlibs"); suppressMessages(library(BGLR))
MODE<-Sys.getenv("MODE","BRR"); W<-"/tmp/bglr_fold"
mt<-scan(paste0(W,"_meta.txt"),quiet=TRUE); n<-mt[1];p<-mt[2];ntr<-mt[3];nte<-mt[4]
con<-file(paste0(W,"_X.bin"),"rb");X<-readBin(con,"double",n=n*p);close(con);X<-matrix(X,n,p,byrow=TRUE)
con<-file(paste0(W,"_y.bin"),"rb");y<-readBin(con,"double",n=n);close(con)
con<-file(paste0(W,"_ytrue.bin"),"rb");ytrue<-readBin(con,"double",n=nte);close(con)
if(MODE=="BRR"){
  ETA<-list(list(X=X,model="BRR",R2=0.5))
}else{ # conventional sparse BayesB
  ETA<-list(list(X=X,model="BayesB",probIn=0.01,counts=1e5,R2=0.5))
}
fm<-BGLR(y=y,ETA=ETA,nIter=3000,burnIn=1000,verbose=FALSE,saveAt=paste0("/tmp/bt_",MODE,"_"))
te<-(ntr+1):n; b<-fm$ETA[[1]]$b; mu<-fm$mu
manual<-as.numeric(mu + X[te,,drop=FALSE]%*%b)
pin<-tryCatch({d<-read.table(paste0("/tmp/bt_",MODE,"_ETA_1_parBayesB.dat"),header=TRUE);mean(tail(d$probIn,400))},error=function(e)NA)
cat(sprintf("MODE=%s  test Pearson(manual beta)=%.4f  yHat=%.4f  trainfit=%.4f  probIn~%s\n",
    MODE, cor(ytrue,manual), cor(ytrue,fm$yHat[te]), cor(y[1:ntr],fm$yHat[1:ntr]),
    ifelse(is.na(pin),"NA(BRR)",sprintf("%.4f",pin))))
