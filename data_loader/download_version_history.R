#!/usr/bin/env Rscript
# Download clinical trial version history using cthist package
# Usage: Rscript download_version_history.R NCT02119676 output_path.json

suppressPackageStartupMessages({
  library(cthist)
  library(jsonlite)
})

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat('{"error": "Usage: Rscript download_version_history.R <NCT_ID> <output_path>"}')
  quit(status = 1)
}

nct_id <- args[1]
output_path <- args[2]

tryCatch({
  # Download version history
  message(sprintf("Downloading version history for %s...", nct_id))
  hist_df <- clinicaltrials_gov_download(nct_id)
  
  if (is.null(hist_df) || nrow(hist_df) == 0) {
    result <- list(
      success = FALSE,
      nct_id = nct_id,
      message = "No version history available",
      versions = list()
    )
  } else {
    # Sort by version number
    hist_df <- hist_df[order(hist_df$version_number), ]
    
    # Convert to list format
    versions <- list()
    for (i in 1:nrow(hist_df)) {
      row <- as.list(hist_df[i, ])
      # Convert any factors to characters
      row <- lapply(row, function(x) {
        if (is.factor(x)) as.character(x) else x
      })
      versions[[i]] <- row
    }
    
    # Detect changes between versions
    changes <- list()
    if (nrow(hist_df) >= 2) {
      fields <- setdiff(names(hist_df), c("nctid", "version_number", "version_date", "download_date"))
      
      for (i in 2:nrow(hist_df)) {
        prev <- hist_df[i - 1, ]
        curr <- hist_df[i, ]
        
        version_changes <- list(
          from_version = prev$version_number,
          to_version = curr$version_number,
          from_date = as.character(prev$version_date),
          to_date = as.character(curr$version_date),
          changed_fields = list()
        )
        
        for (field in fields) {
          prev_value <- as.character(prev[[field]])
          curr_value <- as.character(curr[[field]])
          
          if (!identical(prev_value, curr_value) && 
              !(is.na(prev_value) && is.na(curr_value))) {
            version_changes$changed_fields[[field]] <- list(
              old_value = if(is.na(prev_value)) NULL else prev_value,
              new_value = if(is.na(curr_value)) NULL else curr_value
            )
          }
        }
        
        if (length(version_changes$changed_fields) > 0) {
          changes[[length(changes) + 1]] <- version_changes
        }
      }
    }
    
    result <- list(
      success = TRUE,
      nct_id = nct_id,
      total_versions = nrow(hist_df),
      first_version_date = as.character(hist_df$version_date[1]),
      last_version_date = as.character(hist_df$version_date[nrow(hist_df)]),
      versions = versions,
      changes = changes
    )
  }
  
  # Write to JSON
  writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE, null = "null"), output_path)
  message(sprintf("Successfully saved to %s", output_path))
  
}, error = function(e) {
  result <- list(
    success = FALSE,
    nct_id = nct_id,
    message = conditionMessage(e),
    versions = list()
  )
  writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), output_path)
  message(sprintf("Error: %s", conditionMessage(e)))
})

