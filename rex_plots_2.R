library(data.table)
library(ReX)
library(ggplot2)

path = "C:\\Users\\Hp\\Desktop\\renderex\\VarCoNet_GNN\\results_rDCM"
path_save = "C:\\Users\\Hp\\Desktop\\renderex\\VarCoNet_GNN\\plots"
method = "BNT"
batch_size = 128
samples = 400

if (method == "BNT"){
  path_crvae <- file.path(path, paste0("ICC_", method, "_crvae_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_crvae <- fread(path_crvae)
  path_noise025 <- file.path(path, paste0("ICC_", method, "_noise0.25_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noise025 <- fread(path_noise025)
  path_noise05 <- file.path(path, paste0("ICC_", method, "_noise0.5_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noise05 <- fread(path_noise05)
  path_noise075 <- file.path(path, paste0("ICC_", method, "_noise0.75_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noise075 <- fread(path_noise075)
  path_rdcm <- file.path(path, paste0("ICC_", method, "_rDCM_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_rdcm <- fread(path_rdcm)
  path_cvae <- file.path(path, paste0("ICC_", method, "_cvae_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_cvae <- fread(path_cvae)
  path_noiseFC01 <- file.path(path, paste0("ICC_", method, "_noiseFC0.1_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noiseFC01 <- fread(path_noiseFC01)
  path_noiseFC02 <- file.path(path, paste0("ICC_", method, "_noiseFC0.2_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noiseFC02 <- fread(path_noiseFC02)
  path_noiseFC04 <- file.path(path, paste0("ICC_", method, "_noiseFC0.4_batch_size", batch_size, "_tau0.05_length", samples, "_samples.csv"))
  data_noiseFC04 <- fread(path_noiseFC04)
} else{
  path_crvae <- file.path(path, paste0("ICC_", method, "_crvae_batch_size", batch_size, "_tau0.05_samples.csv"))
  data_crvae <- fread(path_crvae)
  path_noise025 <- file.path(path, paste0("ICC_", method, "_noise0.25_batch_size", batch_size, "_tau0.05_samples.csv"))
  data_noise025 <- fread(path_noise025)
  path_noise05 <- file.path(path, paste0("ICC_", method, "_noise0.5_batch_size", batch_size, "_tau0.05_samples.csv"))
  data_noise05 <- fread(path_noise05)
  path_noise075 <- file.path(path, paste0("ICC_", method, "_noise0.75_batch_size", batch_size, "_tau0.05_samples.csv"))
  data_noise075 <- fread(path_noise075)
  path_rdcm <- file.path(path, paste0("ICC_", method, "_rDCM_batch_size", batch_size, "_tau0.05_samples.csv"))
  data_rdcm <- fread(path_rdcm)
}

data_crvae$group <- "CR-VAE"
data_noise025$group <- "Noise-0.25"
data_noise05$group <- "Noise-0.5"
data_noise075$group <- "Noise-0.75"
data_rdcm$group <- "rDCM"
if (method == "BNT"){
  data_noiseFC01$group <- "FC Noise-0.1"
  data_noiseFC02$group <- "FC Noise-0.2"
  data_noiseFC04$group <- "FC Noise-0.4"
  data_cvae$group <- "CVAE"
}


if (method == "BNT"){
  combined_data <- rbind(data_crvae, data_rdcm)
  p_rdcm_crvae_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_CRVAE_batch_size", batch_size, "_tau0.05_length", samples, ".png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_crvae[,"sigma2_w"]), as.matrix(data_crvae[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - CR-VAE"
  p_rdcm_crvae_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_CRVAE_batch_size", batch_size, "_tau0.05_length", samples, "_grad_norm.png")))
  
  combined_data <- rbind(data_noise025, data_rdcm)
  p_rdcm_noise025_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_noise0.25_batch_size", batch_size, "_tau0.05_length", samples, ".png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noise025[,"sigma2_w"]), as.matrix(data_noise025[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - Noise-0.25"
  p_rdcm_noise025_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_noise0.25_batch_size", batch_size, "_tau0.05_length", samples, "_grad_norm.png")))
  
  combined_data <- rbind(data_noise05, data_rdcm)
  p_rdcm_noise05_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_noise0.5_batch_size", batch_size, "_tau0.05_length", samples, ".png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noise05[,"sigma2_w"]), as.matrix(data_noise05[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - Noise-0.5"
  p_rdcm_noise05_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_noise0.5_batch_size", batch_size, "_tau0.05_length", samples, "_grad_norm.png")))
  
  combined_data <- rbind(data_noise075, data_rdcm)
  p_rdcm_noise075_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_noise0.75_batch_size", batch_size, "_tau0.05_length", samples, ".png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noise075[,"sigma2_w"]), as.matrix(data_noise075[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - Noise-0.75"
  p_rdcm_noise075_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_noise0.75_batch_size", batch_size, "_tau0.05_length", samples, "_grad_norm.png")))
  
  combined_data <- rbind(data_cvae, data_rdcm)
  p_rdcm_cvae_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_CVAE_batch_size", batch_size, "_tau0.05_length", samples, ".png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_cvae[,"sigma2_w"]), as.matrix(data_cvae[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - CVAE"
  p_rdcm_cvae_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_CVAE_batch_size", batch_size, "_tau0.05_length", samples, "_grad_norm.png")))
  
  combined_data <- rbind(data_noiseFC01, data_rdcm)
  p_rdcm_noiseFC01_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_noiseFC0.1_batch_size", batch_size, "_tau0.05_length", samples, ".png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noiseFC01[,"sigma2_w"]), as.matrix(data_noiseFC01[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - FC Noise-0.1"
  p_rdcm_noiseFC01_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_noiseFC01_batch_size", batch_size, "_tau0.05_length", samples, "_grad_norm.png")))
  
  combined_data <- rbind(data_noiseFC02, data_rdcm)
  p_rdcm_noiseFC02_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_noiseFC02_batch_size", batch_size, "_tau0.05_length", samples, ".png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noiseFC02[,"sigma2_w"]), as.matrix(data_noiseFC02[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - FC Noise-0.2"
  p_rdcm_noiseFC02_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_noiseFC02_batch_size", batch_size, "_tau0.05_length", samples, "_grad_norm.png")))
  
  combined_data <- rbind(data_noiseFC04, data_rdcm)
  p_rdcm_noiseFC04_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_noiseFC04_batch_size", batch_size, "_tau0.05_length", samples, ".png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noiseFC04[,"sigma2_w"]), as.matrix(data_noiseFC04[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - FC Noise-0.4"
  p_rdcm_noiseFC04_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_noiseFC04_batch_size", batch_size, "_tau0.05_length", samples, "_grad_norm.png")))
} else {
  combined_data <- rbind(data_crvae, data_rdcm)
  p_rdcm_crvae_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_CRVAE_batch_size", batch_size, "_tau0.05.png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_crvae[,"sigma2_w"]), as.matrix(data_crvae[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - CR-VAE"
  p_rdcm_crvae_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_CRVAE_batch_size", batch_size, "_tau0.05_grad_norm.png")))
  
  combined_data <- rbind(data_noise025, data_rdcm)
  p_rdcm_noise025_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_noise0.25_batch_size", batch_size, "_tau0.05.png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noise025[,"sigma2_w"]), as.matrix(data_noise025[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - Noise-0.25"
  p_rdcm_noise025_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_noise0.25_batch_size", batch_size, "_tau0.05_grad_norm.png")))
  
  combined_data <- rbind(data_noise05, data_rdcm)
  p_rdcm_noise05_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_noise0.5_batch_size", batch_size, "_tau0.05.png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noise05[,"sigma2_w"]), as.matrix(data_noise05[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - Noise-0.5"
  p_rdcm_noise05_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_noise0.5_batch_size", batch_size, "_tau0.05_grad_norm.png")))
  
  combined_data <- rbind(data_noise075, data_rdcm)
  p_rdcm_noise075_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
  ggsave(file.path(path_save,paste0(method, "_noise0.75_batch_size", batch_size, "_tau0.05.png")))
  df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noise075[,"sigma2_w"]), as.matrix(data_noise075[,"sigma2_b"]))
  df_VarPairedComp$contrast <- "rDCM - Noise-0.75"
  p_rdcm_noise075_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
  ggsave(file.path(path_save, paste0(method, "_noise0.75_batch_size", batch_size, "_tau0.05_grad_norm.png")))
}


