library(EMMIXskew)
library(aricode)
library(here)

read_fit_metric_tMM <- function(files, setup, p, 
                                mu1, mu2, mu3, 
                                sigma1, sigma2, sigma3, 
                                iteMC){
  
  errormu1 = c()
  errormu2 = c()
  errormu3 = c()
  
  error1 = c()
  error2 = c()
  error3 = c()
  
  ARI = c()
  AMI = c()
  
  for (i in 0:5) {

    if (paste0("data_x", i, "_s", setup, ".csv") %in% files) {
      
      data = read.table(here("python", "ultimoultimo", paste0("data_x", i, "_s", setup, ".csv")), sep = ",", header = FALSE)

      if (ncol(data) == p) {
        labels = unlist(read.table(here("python", "ultimoultimo", paste0("labels_x", i, "_s", setup, ".csv")), header = FALSE))
        
        res = EmSkew(data, 3, distr="mvt")
        
        est_labels = res$clust
        print(est_labels)
        print(unique(est_labels))

        mus = res$mu
        errormu1 = c(errormu1, min(sqrt(sum((mu1 - mus[, 1])^2)), sqrt(sum((mu1 - mus[, 2])^2)), sqrt(sum((mu1 - mus[, 3])^2))))
        errormu2 = c(errormu2, min(sqrt(sum((mu2 - mus[, 1])^2)), sqrt(sum((mu2 - mus[, 2])^2)), sqrt(sum((mu2 - mus[, 3])^2))))
        errormu3 = c(errormu3, min(sqrt(sum((mu3 - mus[, 1])^2)), sqrt(sum((mu3 - mus[, 2])^2)), sqrt(sum((mu3 - mus[, 3])^2))))
        
        sigmas = res$sigma
        error1 = c(error1, min(norm(sigma1 - sigmas[, , 1], type = "F")/(p*p), norm(sigma1 - sigmas[, , 2], type = "F")/(p*p), norm(sigma1 - sigmas[, , 3], type = "F")/(p*p)))
        error2 = c(error2, min(norm(sigma2 - sigmas[, , 1], type = "F")/(p*p), norm(sigma2 - sigmas[, , 2], type = "F")/(p*p), norm(sigma2 - sigmas[, , 3], type = "F")/(p*p)))
        error3 = c(error3, min(norm(sigma3 - sigmas[, , 1], type = "F")/(p*p), norm(sigma3 - sigmas[, , 2], type = "F")/(p*p), norm(sigma3 - sigmas[, , 3], type = "F")/(p*p)))
        
        ARI = c(ARI, ARI(labels, est_labels))
        AMI = c(AMI, AMI(labels, est_labels))
        
      }
      
    }
  }
  return(list(errormu1 = errormu1, 
              errormu2 = errormu2, 
              errormu3 = errormu3, 
              error1 = error1, 
              error2 = error2, 
              error3 = error3, 
              ARI = ARI, 
              AMI = AMI))
}

#-------------------------------------------------------#
##### SETUP 1 #####
p = 8
mu1 = c(0.15428758, 0.13369956, 0.36268547, 0.67910888, 0.19445006,
        0.25121038, 0.75841639, 0.55761859)
mu2 = rep(2, p)
mu3 = c(4, rep(1, p-1))
setup = 1
sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = diag(c(0.5, 2.6, 0.45, 0.45,1.5,0.5, 1, 1))
sigma3 = diag(p)

#-------------------------------------------------------#


files = list.files(here("python", "data_simu"))

ress = read_fit_metric_tMM(files, setup, p,
                          mu1, mu2, mu3,
                          sigma1, sigma2, sigma3)

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

#-------------------------------------------------------#
p = 8
mu1 = c(0.15428758, 0.13369956, 0.36268547, 0.67910888, 0.19445006,
        0.25121038, 0.75841639, 0.55761859)
mu2 = rep(9, p)
mu3 = c(1.55148029, 1.54677999, 1.5087176 , 1.58290954, 1.52986406,
        1.50313459, 1.56780058, 1.5903489)

sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = diag(c(0.5, 2.6, 0.45, 0.45,1.5,0.5, 1, 1))
sigma3 = diag(p)

#-------------------------------------------------------#

p = 8
mu1 = c(0.15428758, 0.13369956, 0.36268547, 0.67910888, 0.19445006,
        0.25121038, 0.75841639, 0.55761859)
mu2 = rep(5, p)
mu3 = c(1.55148029, 1.54677999, 1.5087176 , 1.58290954, 1.52986406, 
        1.50313459, 1.56780058, 1.5903489)

sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = toeplitz(c(1, 0, 0.25, 0.5, 0, 0.1, 0, 0))
sigma3 = diag(p)
setup = 7

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

##### SETUP 3 #####
#CHECKED
p = 8
mu1 = c(0.15428758, 0.13369956, 0.36268547, 0.67910888, 0.19445006,
        0.25121038, 0.75841639, 0.55761859)
mu2 = rep(9, p)
mu3 = c(1.55148029, 1.54677999, 1.5087176 , 1.58290954, 1.52986406,
        1.50313459, 1.56780058, 1.5903489)
setup = 3
sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = diag(c(0.5, 2.6, 0.45, 0.45,1.5,0.5, 1, 1))
sigma3 = diag(p)

#-------------------------------------------------------#


files = list.files(here("python", "data_simu"))

ress = read_fit_metric_tMM(files, setup, p,
                          mu1, mu2, mu3,
                          sigma1, sigma2, sigma3)

mean(res$errormu1/p)
mean(res$errormu2/p)
mean(res$errormu3/p)
sd(res$errormu1/p)
sd(res$errormu2/p)
sd(res$errormu3/p)

mean(res$error1)
mean(res$error2)
mean(res$error3)
sd(res$error1)
sd(res$error2)
sd(res$error3)

mean(res$ARI)
sd(res$ARI)
mean(res$AMI)
sd(res$AMI)


#--------
setup = 4
p = 8
mu1 = c(0.15428758, 0.13369956, 0.36268547, 0.67910888, 0.19445006,
        0.25121038, 0.75841639, 0.55761859)
mu2 = rep(5, p)
mu3 = c(1.55148029, 1.54677999, 1.5087176 , 1.58290954, 1.52986406,
        1.50313459, 1.56780058, 1.5903489)
sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = diag(c(0.5, 2.6, 0.45, 0.45,1.5,0.5, 1, 1))
sigma3 = diag(p)
files = list.files(here("python", "data_simu"))

ress = read_fit_metric_tMM(files, setup, p,
                          mu1, mu2, mu3,
                          sigma1, sigma2, sigma3)

errormu2
mean(res$errormu1/p)
mean(res$errormu2/p)
mean(res$errormu3/p)
sd(res$errormu1/p)
sd(res$errormu2/p)
sd(res$errormu3/p)

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

##### SETUP 5 #####
p = 8
mu1 = c(1.38730208, -1.06358067, -0.60398188, -0.74177785,  0.34010678,
        1.52536128, -2.89722159,  1.33189907)
mu2 = rep(9, p)
mu3 = c(5.00087176, 5.00829095, 5.00298641, 5.00031346, 5.00678006,
        5.00903489, 5.00514451, 5.00539105)
setup = 5
sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = toeplitz(rep(0.5, p)^seq(0, p-1))
sigma3 = diag(p)

#-------------------------------------------------------#


files = list.files(here("python"))

ress5 = read_fit_metric_tMM(files, setup, p,
                          mu1, mu2, mu3,
                          sigma1, sigma2, sigma3)

mean(ress5$errormu1/p)
mean(ress5$errormu2/p)
mean(ress5$errormu3/p)
sd(ress5$errormu1/p)
sd(ress5$errormu2/p)
sd(ress5$errormu3/p)

mean(ress5$error1)
mean(ress5$error2)
mean(ress5$error3)
sd(ress5$error1)
sd(ress5$error2)
sd(ress5$error3)

mean(ress$ARI)
sd(ress$ARI)
mean(ress$AMI)
sd(ress$AMI)

##### SETUP 6 #####
#checked
p = 8
mu1 = c(0.15428758, 0.13369956, 0.36268547, 0.67910888, 0.19445006,
        0.25121038, 0.75841639, 0.55761859)
mu2 = c(6.05148029, 6.04677999, 6.0087176 , 6.08290954, 6.02986406,
        6.00313459, 6.06780058, 6.0903489)
mu3 = c(1.50514451, 1.50539105, 1.50664328, 1.50634057, 1.50353419,
        1.50026643, 1.5016529 , 1.50879319)
setup = 6
sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = toeplitz(rep(0.7, p)^seq(0, p-1))
sigma3 = diag(p)

#-------------------------------------------------------#


files = list.files(here("python"))

ress6 = read_fit_metric_tMM(files, setup, p,
                            mu1, mu2, mu3,
                            sigma1, sigma2, sigma3)

mean(ress6$errormu1/p)
mean(ress6$errormu2/p)
mean(ress6$errormu3/p)

sd(ress6$errormu1/p)
sd(ress6$errormu2/p)
sd(ress6$errormu3/p)

mean(ress6$error1)
mean(ress6$error2)
mean(ress6$error3)
sd(ress6$error1)
sd(ress6$error2)
sd(ress6$error3)

mean(ress6$ARI)
sd(ress6$ARI)
mean(ress6$AMI)
sd(ress6$AMI)

##### SETUP 7 #####
p = 8
mu1 = c(0.15428758, 0.13369956, 0.36268547, 0.67910888, 0.19445006,
        0.25121038, 0.75841639, 0.55761859)
mu2 = rep(5, p)
mu3 = c(1.50514803, 1.504678  , 1.50087176, 1.50829095, 1.50298641,
        1.50031346, 1.50678006, 1.50903489)
setup = 7
sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = toeplitz(rep(0.3, p)^seq(0, p-1))
sigma3 = diag(p)

#-------------------------------------------------------#

files = list.files(here("python"))

ress7 = read_fit_metric_tMM(files, setup, p,
                            mu1, mu2, mu3,
                            sigma1, sigma2, sigma3)

mean(ress7$errormu1/p)
mean(ress7$errormu2/p)
mean(ress7$errormu3/p)
sd(ress7$errormu1/p)
sd(ress7$errormu2/p)
sd(ress7$errormu3/p)

mean(ress7$error1)
mean(ress7$error2)
mean(ress7$error3)
sd(ress7$error1)
sd(ress7$error2)
sd(ress7$error3)

mean(ress7$ARI)
sd(ress7$ARI)
mean(ress7$AMI)
sd(ress7$AMI)

##### SETUP 8 #####
p = 8
mu1 = c(1.27405293, -0.97675776, -0.55467723, -0.68122456,  0.31234296,
        1.40084199, -2.66071371,  1.22317261)
mu2 = c(6.0087176 , 6.08290954, 6.02986406, 6.00313459, 6.06780058,
        6.0903489 , 6.05144512, 6.05391055)
mu3 = c(3.09170838, 3.09170838, 3.09170838, 3.09170838, 3.09170838,
        3.09170838, 3.09170838, 3.09170838)
setup = 8
sigma1 = diag(c(2.75, 2, 0.5, 0.75,0.5,0.5, 0.5, 0.5))
sigma2 = toeplitz(rep(0.5, p)^seq(0, p-1))
sigma3 = diag(p)

#-------------------------------------------------------#

files = list.files(here("python"))

ress8 = read_fit_metric_tMM(files, setup, p,
                            mu1, mu2, mu3,
                            sigma1, sigma2, sigma3)

mean(ress8$errormu1/p)
mean(ress8$errormu2/p)
mean(ress8$errormu3/p)
sd(ress8$errormu1/p)
sd(ress8$errormu2/p)
sd(ress8$errormu3/p)

mean(ress8$error1)
mean(ress8$error2)
mean(ress8$error3)
sd(ress8$error1)
sd(ress8$error2)
sd(ress8$error3)

mean(ress8$ARI)
sd(ress8$ARI)
mean(ress8$AMI)
sd(ress8$AMI)



##### SETUP 9 #####
p = 10
mu1 = c(0.77143789, 0.6684978 , 1.81342733, 3.39554438, 0.97225029,
        1.25605192, 3.79208196, 2.78809295, 2.57401459, 2.33899931)
mu2 = rep(6, p)
mu3 = rep(7, p)
setup = 9
sigma1 = toeplitz(rep(0.2, p)^seq(0, p-1))
sigma2 = toeplitz(rep(0.5, p)^seq(0, p-1))
sigma3 = toeplitz(rep(0.8, p)^seq(0, p-1))

#-------------------------------------------------------#

files = list.files(here("python", "data_simu"))

ress9 = read_fit_metric_tMM(files, setup, p,
                            mu1, mu2, mu3,
                            sigma1, sigma2, sigma3)

paste0(round(mean(ress9$errormu1/p), 4), " & ", round(sd(ress9$errormu1/p), 4))
paste0(round(mean(ress9$errormu2/p), 4), " & ", round(sd(ress9$errormu2/p), 4))
paste0(round(mean(ress9$errormu3/p), 4), " & ", round(sd(ress9$errormu3/p), 4))

paste0(round(mean(ress9$error1), 4), " & ", round(sd(ress9$error1), 4))
paste0(round(mean(ress9$error2), 4), " & ", round(sd(ress9$error2), 4))
paste0(round(mean(ress9$error3), 4), " & ", round(sd(ress9$error3), 4))

paste0(round(mean(ress9$AMI), 4), " & ", round(sd(ress9$AMI), 4))
paste0(round(mean(ress9$ARI), 4), " & ", round(sd(ress9$ARI), 4))

# setup 9 BIS

p = 40
mu1 = rep(2, p)
mu2 = rep(6, p)
mu3 = rep(7, p)
setup = 9
sigma1 = toeplitz(rep(0.2, p)^seq(0, p-1))
sigma2 = toeplitz(rep(0,   p)^seq(0, p-1))
sigma3 = toeplitz(rep(0.5, p)^seq(0, p-1))

#-------------------------------------------------------#

files = list.files(here("python", "ultimo"))

ress9 = read_fit_metric_tMM(files, setup, p,
                            mu1, mu2, mu3,
                            sigma1, sigma2, sigma3, 100)

paste0(round(mean(ress9$errormu1/p), 4), " & ", round(sd(ress9$errormu1/p), 4))
paste0(round(mean(ress9$errormu2/p), 4), " & ", round(sd(ress9$errormu2/p), 4))
paste0(round(mean(ress9$errormu3/p), 4), " & ", round(sd(ress9$errormu3/p), 4))

paste0(round(mean(ress9$error1), 4), " & ", round(sd(ress9$error1), 4))
paste0(round(mean(ress9$error2), 4), " & ", round(sd(ress9$error2), 4))
paste0(round(mean(ress9$error3), 4), " & ", round(sd(ress9$error3), 4))

paste0(round(mean(ress9$AMI), 4), " & ", round(sd(ress9$AMI), 4))
paste0(round(mean(ress9$ARI), 4), " & ", round(sd(ress9$ARI), 4))

# setup 10

p = 50
mu1 = rep(5, p)
mu2 = rep(6, p)
mu3 = rep(7, p)
setup = 10
sigma1 = toeplitz(rep(0.2, p)^seq(0, p-1))
sigma2 = toeplitz(rep(0,   p)^seq(0, p-1))
sigma3 = toeplitz(rep(0.5, p)^seq(0, p-1))

#-------------------------------------------------------#

files = list.files(here("python", "ultimoultimo"))

ress10 = read_fit_metric_tMM(files, setup, p,
                            mu1, mu2, mu3,
                            sigma1, sigma2, sigma3, 100)

paste0(round(mean(ress10$errormu1/p), 4), " & ", round(sd(ress10$errormu1/p), 4))
paste0(round(mean(ress10$errormu2/p), 4), " & ", round(sd(ress10$errormu2/p), 4))
paste0(round(mean(ress10$errormu3/p), 4), " & ", round(sd(ress10$errormu3/p), 4))

paste0(round(mean(ress10$error1), 4), " & ", round(sd(ress10$error1), 4))
paste0(round(mean(ress10$error2), 4), " & ", round(sd(ress10$error2), 4))
paste0(round(mean(ress10$error3), 4), " & ", round(sd(ress10$error3), 4))

paste0(round(mean(ress10$AMI), 4), " & ", round(sd(ress10$AMI), 4))
paste0(round(mean(ress10$ARI), 4), " & ", round(sd(ress10$ARI), 4))
