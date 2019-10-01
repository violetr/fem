library(EMMIXskew)
library(aricode)

dof = 3

p = 8
mu1 = c(0.15428758, 0.13369956, 0.36268547, 0.67910888, 0.19445006,
       0.25121038, 0.75841639, 0.55761859)
mu2 = rep(2, p)
mu3 = c(4, rep(1, p-1))
  
sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = diag(c(0.5, 2.6, 0.45, 0.45,1.5,0.5, 1, 1))
sigma3 = diag(p)

errormu1 = c()
errormu2 = c()
errormu3 = c()

error1 = c()
error2 = c()
error3 = c()

ARI1 = c()
ARI12 = c()
AMI1 = c()

archivos = list.files()
setwd("C:/Users/roizmanv/code/frem/python")



for (i in 0:999) {
  
  if (paste0("data_x", i, "_", dof, ".csv") %in% archivos) {
    
    data = read.table(paste0("data_x", i, "_", dof, ".csv"), sep = ",", header = FALSE)
    labels = unlist(read.table(paste0("labels_x", i, "_", dof, ".csv"), header = FALSE))
    res = EmSkew(data, 3, distr="mvt")
    
    est_labels = res$clust
    
    mus = res$mu
    errormu1 = c(errormu1, min(sqrt(sum((mu1 - mus[, 1])^2)), sqrt(sum((mu1 - mus[, 2])^2)), sqrt(sum((mu1 - mus[, 3])^2))))
    errormu2 = c(errormu2, min(sqrt(sum((mu2 - mus[, 1])^2)), sqrt(sum((mu2 - mus[, 2])^2)), sqrt(sum((mu2 - mus[, 3])^2))))
    errormu3 = c(errormu3, min(sqrt(sum((mu3 - mus[, 1])^2)), sqrt(sum((mu3 - mus[, 2])^2)), sqrt(sum((mu3 - mus[, 3])^2))))
    
    sigmas = res$sigma
    error1 = c(error1, min(norm(sigma1 - sigmas[, , 1], type = "F")/(p*p), norm(sigma1 - sigmas[, , 2], type = "F")/(p*p), norm(sigma1 - sigmas[, , 3], type = "F")/(p*p)))
    error2 = c(error2, min(norm(sigma2 - sigmas[, , 1], type = "F")/(p*p), norm(sigma2 - sigmas[, , 2], type = "F")/(p*p), norm(sigma2 - sigmas[, , 3], type = "F")/(p*p)))
    error3 = c(error3, min(norm(sigma3 - sigmas[, , 1], type = "F")/(p*p), norm(sigma3 - sigmas[, , 2], type = "F")/(p*p), norm(sigma3 - sigmas[, , 3], type = "F")/(p*p)))
    
    ARI1 = c(ARI1, ARI(labels, est_labels))
    AMI1 =c(AMI1, AMI(labels, est_labels))
    
  }
}


mean(errormu1/p)
mean(errormu2/p)
mean(errormu3/p)
sd(errormu1/p)
sd(errormu2/p)
sd(errormu3/p)

mean(error1)
mean(error2)
mean(error3)
sd(error1)
sd(error2)
sd(error3)

mean(ARI1)
sd(ARI1)
mean(AMI1)
sd(AMI1)

dof = 10

p = 8
mu1 = c(0.15428758, 0.13369956, 0.36268547, 0.67910888, 0.19445006,
        0.25121038, 0.75841639, 0.55761859)
mu2 = rep(9, p)
mu3 = c(1.55148029, 1.54677999, 1.5087176 , 1.58290954, 1.52986406,
        1.50313459, 1.56780058, 1.5903489)

sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = diag(c(0.5, 2.6, 0.45, 0.45,1.5,0.5, 1, 1))
sigma3 = diag(p)

errormu1 = c()
errormu2 = c()
errormu3 = c()

error1 = c()
error2 = c()
error3 = c()

ARI2 = c()
ARI22 = c()
AMI2 = c()

archivos = list.files()
setwd("C:/Users/roizmanv/code/frem/python")

for (i in 0:999) {
  
  if (paste0("data_x", i, "_", dof, ".csv") %in% archivos) {
    
    data = read.table(paste0("data_x", i, "_", dof, ".csv"), sep = ",", header = FALSE)
    labels = unlist(read.table(paste0("labels_x", i, "_", dof, ".csv"), header = FALSE))
    res = EmSkew(data, 3, distr="mvt")
    
    est_labels = res$clust
    
    mus = res$mu
    errormu1 = c(errormu1, min(sqrt(sum((mu1 - mus[, 1])^2)), sqrt(sum((mu1 - mus[, 2])^2)), sqrt(sum((mu1 - mus[, 3])^2))))
    errormu2 = c(errormu2, min(sqrt(sum((mu2 - mus[, 1])^2)), sqrt(sum((mu2 - mus[, 2])^2)), sqrt(sum((mu2 - mus[, 3])^2))))
    errormu3 = c(errormu3, min(sqrt(sum((mu3 - mus[, 1])^2)), sqrt(sum((mu3 - mus[, 2])^2)), sqrt(sum((mu3 - mus[, 3])^2))))
    
    sigmas = res$sigma
    error1 = c(error1, min(norm(sigma1 - sigmas[, , 1], type = "F")/(p*p), norm(sigma1 - sigmas[, , 2], type = "F")/(p*p), norm(sigma1 - sigmas[, , 3], type = "F")/(p*p)))
    error2 = c(error2, min(norm(sigma2 - sigmas[, , 1], type = "F")/(p*p), norm(sigma2 - sigmas[, , 2], type = "F")/(p*p), norm(sigma2 - sigmas[, , 3], type = "F")/(p*p)))
    error3 = c(error3, min(norm(sigma3 - sigmas[, , 1], type = "F")/(p*p), norm(sigma3 - sigmas[, , 2], type = "F")/(p*p), norm(sigma3 - sigmas[, , 3], type = "F")/(p*p)))
    
    ARI2 = c(ARI2, ARI(labels, est_labels))
    AMI2 =c(AMI2, AMI(labels, est_labels))
  }
}

mean(errormu1/p)
mean(errormu2/p)
mean(errormu3/p)
sd(errormu1/p)
sd(errormu2/p)
sd(errormu3/p)

mean(error1)
mean(error2)
mean(error3)
sd(error1)
sd(error2)
sd(error3)

mean(ARI2)
sd(ARI2)
mean(AMI2)
sd(AMI2)
