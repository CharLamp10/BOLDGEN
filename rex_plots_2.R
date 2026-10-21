library(data.table)
library(ReX)
library(ggplot2)

path = "C:\\Users\\106742\\Desktop\\rDCM-GEN_results_HCP_2026\\ReX_files\\full_subsampled"
path_save = "C:\\Users\\106742\\Desktop\\rDCM-GEN_results_HCP_2026\\plots\\ReX\\full_subsampled"
tau = 0.15

path_crvae <- file.path(path, paste0("ICC_CRVAE_tau", tau, ".csv"))
data_crvae <- fread(path_crvae)
path_rdcm <- file.path(path, paste0("ICC_rDCM-GEN_tau", tau, ".csv"))
data_rdcm <- fread(path_rdcm)

rex_plot.var.field(data_crvae, size.point = 2, alpha.density = 0.3, color.point.fill = "blue", color = "blue")
ggsave(file.path(path_save, paste0("CRVAE_tau", tau, ".png")))
rex_plot.var.field(data_crvae, size.point = 2, alpha.density = 0.3, color.point.fill = "blue", color = "blue", show.icc_slope = FALSE)
ggsave(file.path(path_save, paste0("CRVAE_tau", tau, "_no_slope.png")))

rex_plot.var.field(data_rdcm, size.point = 2, alpha.density = 0.3, color.point.fill = "cyan", color = "cyan")
ggsave(file.path(path_save, paste0("rDCM-GEN_tau", tau, ".png")))
rex_plot.var.field(data_rdcm, size.point = 2, alpha.density = 0.3, color.point.fill = "cyan", color = "cyan", show.icc_slope = FALSE)
ggsave(file.path(path_save, paste0("rDCM-GEN_tau", tau, "_no_slope.png")))


data_crvae$group <- "CR-VAE"
data_rdcm$group <- "rDCM-GEN"


combined_data <- rbind(data_crvae, data_rdcm)
p_rdcm_crvae_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3, show.icc_slope=FALSE)
ggsave(file.path(path_save,paste0("rDCM-GEN_CRVAE_tau", tau, ".png")))
df_VarPairedComp <- icc_gradient_flow(as.matrix(data_rdcm[,"sigma2_w"]), as.matrix(data_rdcm[,"sigma2_b"]), as.matrix(data_crvae[,"sigma2_w"]), as.matrix(data_crvae[,"sigma2_b"]))
df_VarPairedComp$contrast <- "rDCM - CR-VAE"
p_rdcm_crvae_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
ggsave(file.path(path_save, paste0("rDCM-GEN_CRVAE_tau", tau, "_grad_norm.png")))




