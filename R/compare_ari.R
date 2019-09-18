source("https://bioconductor.org/biocLite.R")
biocLite("Melissa")

library(Melissa)
library(EMMIXskew)
library("CrossClustering")
install.packages("mclust")
library(mclust)
ARI = rep(0, 24-3)


archivos = list.files()
setwd("C:/Users/roizmanv/em-algo")
for (i in 0:24) {
  if (paste0("data_x", i, ".csv") %in% archivos) {
    data = read.table(paste0("data_x", i, ".csv"), sep = ",", header = FALSE)
    print(data)
    labels = unlist(read.table(paste0("labels_x", i, ".csv"), header = FALSE))
    res = EmSkew(data, 3, distr="mvt")
    est_labels = res$clust
    ARI[i+1] = adjustedRandIndex(labels, est_labels)
  }
}


i= 1
paste0("data_x", i, "c")

mean(ARI[ARI!=0.0])
