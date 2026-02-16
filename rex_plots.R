library(data.table)
library(ReX)
library(ggplot2)

path = "C:\\Users\\100063082\\Desktop\\rDCM-GEN_results\\ReX_files2"
tau = 0.05

#Noise 0.5
path_noise05 <- file.path(path, paste0("tau", tau), "rex_Noise-6dB.csv")
data_noise05 <- fread(path_noise05)
x <- data_noise05[, 4:ncol(data_noise05)]
sub <- data_noise05[,2]
sess <- data_noise05[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_noise05 <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_noise05, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_noise05, file = file.path(path, paste0("ICC_Noise-6dB_tau",tau,".csv")), row.names = FALSE)

#Noise 0.75
path_noise075 <- file.path(path, paste0("tau", tau), "rex_Noise-3dB.csv")
data_noise075 <- fread(path_noise075)
x <- data_noise075[, 4:ncol(data_noise075)]
sub <- data_noise075[,2]
sess <- data_noise075[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_noise075 <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_noise075, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_noise075, file = file.path(path, paste0("ICC_Noise-3dB_tau",tau,".csv")), row.names = FALSE)

#CRVAE
path_crvae <- file.path(path, paste0("tau", tau), "rex_CRVAE.csv")
data_crvae <- fread(path_crvae)
x <- data_crvae[, 4:ncol(data_crvae)]
sub <- data_crvae[,2]
sess <- data_crvae[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_crvae <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_crvae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_crvae, file = file.path(path, paste0("ICC_CRVAE_tau",tau,".csv")), row.names = FALSE)

#rDCM
path_rdcm <- file.path(path, paste0("tau", tau), "rex_rDCM-GEN.csv")
data_rdcm <- fread(path_rdcm)
x <- data_rdcm[, 4:ncol(data_rdcm)]
sub <- data_rdcm[,2]
sess <- data_rdcm[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_rdcm <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_rdcm, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_rdcm, file = file.path(path, paste0("ICC_rDCM-GEN_tau",tau,".csv")), row.names = FALSE)
