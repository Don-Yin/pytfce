# Minimal R pTFCE wrapper for the experiment pipeline.
#
# Usage:
#   Rscript r_ptfce_pipeline.R <z_map.nii.gz> <mask.nii.gz> <output_dir>
#
# Outputs (in output_dir):
#   r_ptfce_Z.nii.gz        - enhanced Z-score map
#   r_ptfce_diag.json       - timing + GRF diagnostics

user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))

library(pTFCE)
library(oro.nifti)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript r_ptfce_pipeline.R <z_map> <mask> <output_dir> [Rd] [V] [resels]")
}

z_path   <- args[1]
mask_path <- args[2]
out_dir   <- args[3]
override_Rd     <- if (length(args) >= 4) as.numeric(args[4]) else NA
override_V      <- if (length(args) >= 5) as.integer(args[5]) else NA
override_resels <- if (length(args) >= 6) as.numeric(args[6]) else NA
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

Z    <- readNIfTI(z_path)
MASK <- readNIfTI(mask_path)

t0 <- proc.time()
if (!is.na(override_Rd) && !is.na(override_V)) {
  if (!is.na(override_resels)) {
    result <- ptfce(Z, MASK, Rd = override_Rd, V = override_V,
                    resels = override_resels, verbose = FALSE)
  } else {
    result <- ptfce(Z, MASK, Rd = override_Rd, V = override_V,
                    verbose = FALSE)
  }
} else {
  result <- ptfce(Z, MASK, verbose = FALSE)
}
elapsed <- (proc.time() - t0)["elapsed"]

writeNIfTI(result$Z, file.path(out_dir, "r_ptfce_Z"))

diag_lines <- paste0(
  "{\n",
  '  "method": "R_pTFCE",\n',
  '  "elapsed_s": ', as.numeric(elapsed), ',\n',
  '  "Z_enh_max": ', as.numeric(max(result$Z)), ',\n',
  '  "fwer_z_thresh": ', as.numeric(result$fwer0.05.Z), ',\n',
  '  "n_resels": ', as.numeric(result$number_of_resels), '\n',
  "}"
)
writeLines(diag_lines, file.path(out_dir, "r_ptfce_diag.json"))

cat("R pTFCE done:", as.numeric(elapsed), "s\n")
