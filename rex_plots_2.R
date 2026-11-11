# Plot normalized ReX gradient flow comparisons of learned embeddings.

library(data.table)
library(ReX)
library(ggplot2)

path = "C:\\Users\\106742\\Desktop\\BOLDGEN_results_HCP_2026\\ReX_files\\half_original"
path_save = "C:\\Users\\106742\\Desktop\\BOLDGEN_results_HCP_2026\\plots\\ReX\\half_original"
tau = 0.15

path_crvae <- file.path(path, paste0("ICC_CR-VAE_tau", tau, ".csv"))
data_crvae <- fread(path_crvae)
path_boldgen <- file.path(path, paste0("ICC_BOLDGEN_tau", tau, ".csv"))
data_boldgen <- fread(path_boldgen)

p <- rex_plot.var.field(data_crvae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
# Increase axis tick labels and axis titles
p <- p + theme(
  axis.text  = element_text(size = 14),
  axis.title = element_text(size = 16)
)

for (i in seq_along(p$layers)) {
  if (inherits(p$layers[[i]]$geom, "GeomText")) {
    p$layers[[i]]$aes_params$size  <- 5
    p$layers[[i]]$aes_params$hjust <- 1
  }
}

print(p)
ggsave(file.path(path_save, paste0("CRVAE_tau", tau, ".png")))

p <- rex_plot.var.field(data_boldgen, size.point = 2, alpha.density = 0.3, color.point.fill = "blue", color = "blue")
p <- p + theme(
  axis.text  = element_text(size = 14),
  axis.title = element_text(size = 16)
)

for (i in seq_along(p$layers)) {
  if (inherits(p$layers[[i]]$geom, "GeomText")) {
    p$layers[[i]]$aes_params$size  <- 5
    p$layers[[i]]$aes_params$hjust <- 1
  }
}
print(p)
ggsave(file.path(path_save, paste0("BOLDGEN_tau", tau, ".png")))


data_crvae$group <- "CR-VAE"
data_boldgen$group <- "BOLDGEN"


combined_data <- rbind(data_crvae, data_boldgen)
p_boldgen_crvae_1 <- rex_plot.var.field.n(
  combined_data,
  size.point = 2,
  alpha.density = 0.3,
  show.icc_slope = TRUE
) +
  theme(
    axis.text    = element_text(size = 14),
    axis.title   = element_text(size = 16),
    legend.text  = element_text(size = 14),
    legend.title = element_text(size = 16)
  )

# Applies if show.icc_slope = TRUE
for (i in seq_along(p_boldgen_crvae_1$layers)) {
  if (inherits(p_boldgen_crvae_1$layers[[i]]$geom, "GeomText")) {
    p_boldgen_crvae_1$layers[[i]]$aes_params$size  <- 5
    p_boldgen_crvae_1$layers[[i]]$aes_params$hjust <- 1
  }
}
print(p_boldgen_crvae_1)
ggsave(file.path(path_save,paste0("BOLDGEN_CR-VAE_tau", tau, ".png")))
df_VarPairedComp <- icc_gradient_flow(as.matrix(data_boldgen[,"sigma2_w"]), as.matrix(data_boldgen[,"sigma2_b"]), as.matrix(data_crvae[,"sigma2_w"]), as.matrix(data_crvae[,"sigma2_b"]))
df_VarPairedComp$contrast <- "BOLDGEN - CR-VAE"
p_boldgen_crvae_2 <- rex_plot.icc.gradient.norm(
  df_VarPairedComp,
  show.contour = FALSE
) +
  theme(
    axis.text    = element_text(size = 14),
    axis.title   = element_text(size = 16),
    legend.text  = element_text(size = 14),
    legend.title = element_text(size = 16)
  )

# Enlarge the ΔICC and Optimal annotations and move them inward
for (i in seq_along(p_boldgen_crvae_2$layers)) {
  if (inherits(p_boldgen_crvae_2$layers[[i]]$geom, "GeomText")) {
    p_boldgen_crvae_2$layers[[i]]$aes_params$size  <- 5
    p_boldgen_crvae_2$layers[[i]]$aes_params$hjust <- 0.5
    
    p_boldgen_crvae_2$layers[[i]]$data$x <-
      p_boldgen_crvae_2$layers[[i]]$data$x * 0.85
    
    p_boldgen_crvae_2$layers[[i]]$data$y <-
      p_boldgen_crvae_2$layers[[i]]$data$y * 0.85
  }
}

for (i in seq_along(p_boldgen_crvae_2$layers)) {
  if (inherits(p_boldgen_crvae_2$layers[[i]]$geom, "GeomText")) {
    label <- p_boldgen_crvae_2$layers[[i]]$aes_params$label
    
    if (any(grepl("Optimal", label))) {
      p_boldgen_crvae_2$layers[[i]]$aes_params$vjust <- 0.1
    } else if (any(grepl("ICC[<>]0", label))) {
      p_boldgen_crvae_2$layers[[i]]$aes_params$vjust <- 0.9
    } else if (any(grepl("ICC=0", label))) {
      p_boldgen_crvae_2$layers[[i]]$aes_params$vjust <- 0.9
    }
  }
}

print(p_boldgen_crvae_2)
ggsave(file.path(path_save, paste0("BOLDGEN_CR-VAE_tau", tau, "_grad_norm.png")))




