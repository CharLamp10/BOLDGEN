library(data.table)
library(ReX)
library(ggplot2)

path = "C:\\Users\\100063082\\Desktop\\rDCM-GEN_results\\ReX_files"
path_save = "C:\\Users\\100063082\\Desktop\\rDCM-GEN\\plots"
tau = 0.05

path_crvae <- file.path(path, paste0("ICC_CRVAE_tau", tau, ".csv"))
data_crvae <- fread(path_crvae)
path_noise05 <- file.path(path, paste0("ICC_Noise-6dB_tau", tau, ".csv"))
data_noise05 <- fread(path_noise05)
path_noise075 <- file.path(path, paste0("ICC_Noise-3dB_tau", tau, ".csv"))
data_noise075 <- fread(path_noise075)
path_rdcm <- file.path(path, paste0("ICC_rDCM-GEN_tau", tau, ".csv"))
data_rdcm <- fread(path_rdcm)


data_crvae$group <- "CR-VAE"
data_noise05$group <- "Noise-6dB"
data_noise075$group <- "Noise-3dB"
data_rdcm$group <- "rDCM-GEN"


combined_data <- rbind(data_crvae, data_rdcm)
p_rdcm_crvae_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
ggsave(file.path(path_save,paste0("CRVAE_tau", tau, ".png")))
df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_crvae[,"sigma2_w"]), as.matrix(data_crvae[,"sigma2_b"]))
df_VarPairedComp$contrast <- "rDCM - CR-VAE"
p_rdcm_crvae_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
ggsave(file.path(path_save, paste0("CRVAE_tau", tau, "_grad_norm.png")))

combined_data <- rbind(data_noise05, data_rdcm)
p_rdcm_noise05_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
ggsave(file.path(path_save,paste0( "Noise-6dB_tau", tau, ".png")))
df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noise05[,"sigma2_w"]), as.matrix(data_noise05[,"sigma2_b"]))
df_VarPairedComp$contrast <- "rDCM - Noise-6dB"
p_rdcm_noise05_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
ggsave(file.path(path_save, paste0("Noise-6dB_tau", tau, "_grad_norm.png")))

combined_data <- rbind(data_noise075, data_rdcm)
p_rdcm_noise075_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
ggsave(file.path(path_save,paste0("Noise-3dB_tau", tau, ".png")))
df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_noise075[,"sigma2_w"]), as.matrix(data_noise075[,"sigma2_b"]))
df_VarPairedComp$contrast <- "rDCM - Noise-3dB"
p_rdcm_noise075_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
ggsave(file.path(path_save, paste0("Noise-3dB_tau", tau, "_grad_norm.png")))



