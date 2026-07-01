args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_dir <- args[2]

#cat("Input file:", input_file, "\n")
#cat("Output dir:", output_dir, "\n")

library(MetaCycle)

tryCatch({
  meta2dout <- meta2d(
    infile = input_file,
    filestyle = "txt",
    timepoints = "line1",
    cycMethod = c("JTK","LS", "ARS"),
    minper = 22,
    maxper = 32,
    outputFile = FALSE,
    outdir = output_dir,
    outIntegration = "onlyIntegration"
  )
  #print(meta2dout)
  write.csv(meta2dout, file.path(output_dir, "meta2d_result.csv"), row.names = FALSE)
  cat("meta2d run completed successfully.\n")
}, error = function(e) {
  cat("Error in meta2d:\n")
  cat(conditionMessage(e), "\n")
  quit(status = 1)
})
