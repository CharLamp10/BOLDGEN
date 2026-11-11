# Visualize ReX individual variation fields for contrastive embeddings.

library(data.table)
library(ReX)
library(ggplot2)

path = "C:\\Users\\106742\\Desktop\\BOLDGEN_results_HCP_2026\\ReX_files\\full_subsampled"
tau = 0.15

#CRVAE
path_crvae <- file.path(path, paste0("tau", tau), "rex_CR-VAE.csv")
data_crvae <- fread(path_crvae)
x <- data_crvae[, 4:ncol(data_crvae)]
sub <- data_crvae[,2]
sess <- data_crvae[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_crvae <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_crvae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_crvae, file = file.path(path, paste0("ICC_CR-VAE_tau",tau,".csv")), row.names = FALSE)

#BOLDGEN
path_boldgen <- file.path(path, paste0("tau", tau), "rex_BOLDGEN.csv")
data_boldgen <- fread(path_boldgen)
x <- data_boldgen[, 4:ncol(data_boldgen)]
sub <- data_boldgen[,2]
sess <- data_boldgen[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_boldgen <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_boldgen, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_boldgen, file = file.path(path, paste0("ICC_BOLDGEN_tau",tau,".csv")), row.names = FALSE)
