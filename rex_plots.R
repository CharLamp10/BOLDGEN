library(data.table)
library(ReX)
library(ggplot2)

path = "C:\\Users\\Hp\\Desktop\\renderex\\VarCoNet_GNN\\results_rDCM"
method = "BNT"
samples = 400
batch_size = 128

if (method=="BNT"){
  #Noise 0.25
  path_noise025 <- file.path(path, paste0("rex_rDCM_", method, "_noise0.25_batch_size",batch_size,"_tau0.05_length", samples, "_samples.csv"))
  data_noise025 <- fread(path_noise025)
  x <- data_noise025[, 4:ncol(data_noise025)]
  sub <- data_noise025[,2]
  sess <- data_noise025[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_noise025 <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_noise025, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_noise025, file = file.path(path, paste0("ICC_", method, "_noise0.25_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv")), row.names = FALSE)
  
  #Noise 0.5
  path_noise05 <- file.path(path, paste0("rex_rDCM_", method, "_noise0.5_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noise05 <- fread(path_noise05)
  x <- data_noise05[, 4:ncol(data_noise05)]
  sub <- data_noise05[,2]
  sess <- data_noise05[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_noise05 <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_noise05, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_noise05, file = file.path(path, paste0("ICC_", method, "_noise0.5_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv")), row.names = FALSE)
  
  #Noise 0.75
  path_noise075 <- file.path(path, paste0("rex_rDCM_", method, "_noise0.75_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noise075 <- fread(path_noise075)
  x <- data_noise075[, 4:ncol(data_noise075)]
  sub <- data_noise075[,2]
  sess <- data_noise075[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_noise075 <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_noise075, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_noise075, file = file.path(path, paste0("ICC_", method, "_noise0.75_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv")), row.names = FALSE)
  
  #CRVAE
  path_crvae <- file.path(path, paste0("rex_rDCM_", method, "_crvae_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_crvae <- fread(path_crvae)
  x <- data_crvae[, 4:ncol(data_crvae)]
  sub <- data_crvae[,2]
  sess <- data_crvae[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_crvae <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_crvae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_crvae, file = file.path(path, paste0("ICC_", method, "_crvae_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv")), row.names = FALSE)
  
  #rDCM
  path_rdcm <- file.path(path, paste0("rex_rDCM_", method, "_rDCM_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_rdcm <- fread(path_rdcm)
  x <- data_rdcm[, 4:ncol(data_rdcm)]
  sub <- data_rdcm[,2]
  sess <- data_rdcm[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_rdcm <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_rdcm, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_rdcm, file = file.path(path, paste0("ICC_", method, "_rDCM_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv")), row.names = FALSE)
  
  #Noise 0.15
  path_noiseFC01 <- file.path(path, paste0("rex_rDCM_", method, "_noiseFC0.1_batch_size", batch_size,"_tau0.05_length", samples, "_samples.csv"))
  data_noiseFC01 <- fread(path_noiseFC01)
  x <- data_noiseFC01[, 4:ncol(data_noiseFC01)]
  sub <- data_noiseFC01[,2]
  sess <- data_noiseFC01[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_noiseFC01 <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_noiseFC01, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_noiseFC01, file = file.path(path, paste0("ICC_", method, "_noiseFC0.1_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv")), row.names = FALSE)
  
  #Noise 0.2
  path_noiseFC02 <- file.path(path, paste0("rex_rDCM_", method, "_noiseFC0.2_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noiseFC02 <- fread(path_noiseFC02)
  x <- data_noiseFC02[, 4:ncol(data_noiseFC02)]
  sub <- data_noiseFC02[,2]
  sess <- data_noiseFC02[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_noiseFC02 <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_noiseFC02, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_noiseFC02, file = file.path(path, paste0("ICC_", method, "_noiseFC0.2_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv")), row.names = FALSE)
  
  #Noise 0.4
  path_noiseFC04 <- file.path(path, paste0("rex_rDCM_", method, "_noiseFC0.4_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noiseFC04 <- fread(path_noiseFC04)
  x <- data_noiseFC04[, 4:ncol(data_noiseFC04)]
  sub <- data_noiseFC04[,2]
  sess <- data_noiseFC04[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_noiseFC04 <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_noiseFC04, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_noiseFC04, file = file.path(path, paste0("ICC_", method, "_noiseFC0.4_batch_size", batch_size, "_tau0.05_length", samples,"_samples.csv")), row.names = FALSE)
  
  #CVAE
  path_cvae <- file.path(path, paste0("rex_rDCM_", method, "_CVAE_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_cvae <- fread(path_cvae)
  x <- data_cvae[, 4:ncol(data_cvae)]
  sub <- data_cvae[,2]
  sess <- data_cvae[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_cvae <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_cvae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_cvae, file = file.path(path, paste0("ICC_", method, "_CVAE_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv")), row.names = FALSE)
} else {
  #Noise 0.25
  path_noise025 <- file.path(path, paste0("rex_rDCM_", method, "_noise0.25_batch_size",batch_size,"_tau0.05_samples.csv"))
  data_noise025 <- fread(path_noise025)
  x <- data_noise025[, 4:ncol(data_noise025)]
  sub <- data_noise025[,2]
  sess <- data_noise025[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_noise025 <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_noise025, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_noise025, file = file.path(path, paste0("ICC_", method, "_noise0.25_batch_size", batch_size, "_tau0.05_samples.csv")), row.names = FALSE)
  
  #Noise 0.5
  path_noise05 <- file.path(path, paste0("rex_rDCM_", method, "_noise0.5_batch_size", batch_size, "_tau0.05_samples.csv"))
  data_noise05 <- fread(path_noise05)
  x <- data_noise05[, 4:ncol(data_noise05)]
  sub <- data_noise05[,2]
  sess <- data_noise05[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_noise05 <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_noise05, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_noise05, file = file.path(path, paste0("ICC_", method, "_noise0.5_batch_size", batch_size, "_tau0.05_samples.csv")), row.names = FALSE)
  
  #Noise 0.75
  path_noise075 <- file.path(path, paste0("rex_rDCM_", method, "_noise0.75_batch_size", batch_size, "_tau0.05_samples.csv"))
  data_noise075 <- fread(path_noise075)
  x <- data_noise075[, 4:ncol(data_noise075)]
  sub <- data_noise075[,2]
  sess <- data_noise075[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_noise075 <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_noise075, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_noise075, file = file.path(path, paste0("ICC_", method, "_noise0.75_batch_size", batch_size, "_tau0.05_samples.csv")), row.names = FALSE)
  
  #CRVAE
  path_crvae <- file.path(path, paste0("rex_rDCM_", method, "_crvae_batch_size", batch_size, "_tau0.05_samples.csv"))
  data_crvae <- fread(path_crvae)
  x <- data_crvae[, 4:ncol(data_crvae)]
  sub <- data_crvae[,2]
  sess <- data_crvae[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_crvae <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_crvae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_crvae, file = file.path(path, paste0("ICC_", method, "_crvae_batch_size", batch_size, "_tau0.05_samples.csv")), row.names = FALSE)
  
  #rDCM
  path_rdcm <- file.path(path, paste0("rex_rDCM_", method, "_rDCM_batch_size", batch_size, "_tau0.05_samples.csv"))
  data_rdcm <- fread(path_rdcm)
  x <- data_rdcm[, 4:ncol(data_rdcm)]
  sub <- data_rdcm[,2]
  sess <- data_rdcm[,3]
  x <- as.matrix(x)
  sub <- as.matrix(sub)
  sess <- as.matrix(sess)
  df_icc_rdcm <- data.frame(lme_ICC_1wayR(x, sub, sess))
  rex_plot.var.field(df_icc_rdcm, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
  write.csv(df_icc_rdcm, file = file.path(path, paste0("ICC_", method, "_rDCM_batch_size", batch_size, "_tau0.05_samples.csv")), row.names = FALSE)
}