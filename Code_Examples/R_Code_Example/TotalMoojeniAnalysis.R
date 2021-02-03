setwd("~/Desktop/R Code Examples/Bothrops_Genetic_Analysis")
install.packages("adegenet") #multivariate analysis of genetic data 
library(adegenet)

#input file generated using STACKS pipline to process RADseq data

#input file must have .gen extension, must change output from STACKS
#example: x <- read.genepop("populations.haps.gen")

xtotal <- read.genepop("populationsTotalMarch4r0R.5populations.snps.gen")

#find distinct groups within the population using find.clusters()
#THIS WILL PROMPT YOU TO CHOOSE HOW MANY PCs TO RETAIN
#You want to account for 100 % of variation ... ~50

# Bayesian information criterion (BIC) should be lowest possible in the prefered model
# Number of clusters ~4
Clusters <- find.clusters(xtotal, max.n.clust=20)

#show group membership after find.clusters
Clusters$grp

#Run discriminant analysis of principle components to cluster and graph clustering
dapc<-dapc(xtotal, Clusters$grp)
#get as many as you get gain, when the slope levels off stop
# Number of PCs to retain ~5
# Try with different numbers of linear discriminants (1/2/3) to visualize genetic differences in populations

#make single axis or biplot of dapc discriminant scores
BiPlot<-scatter(dapc, posi.da = "topleft", legend = TRUE, posi.leg = "bottomleft", grid = FALSE, 
                bg = "white", pch = 20, cell = 0, col = rainbow(6),
                cstar = 0, lwd = 2, cex = 3, clab = 0)

#make a structure plot from the genetic data showing the probability of membership in each cluster for each individual
compoplot(dapc, posi="bottomright", txt.leg=paste("Cluster", 1:4), lab="",ncol=1, xlab="individuals", col=funky(4))

