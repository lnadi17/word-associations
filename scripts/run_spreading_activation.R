#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(spreadr)
  library(igraph)
})

# Compute final activation matrix for a list of initial nodes.
spreadSimple <- function(g, word_list, run_label = "run", progress_every = 10) {
  nodes <- V(g)$name
  if (is.null(nodes) || length(nodes) == 0) {
    stop("Graph has no named vertices.")
  }

  t <- ceiling(2 * diameter(g, weights = NA))
  df_final <- data.frame("node" = nodes)

  if (is_weighted(g)) {
    adj <- as_adjacency_matrix(g, sparse = FALSE, attr = "weight")
  } else {
    adj <- as_adjacency_matrix(g, sparse = FALSE)
  }

  act <- length(nodes)
  node_index <- seq_along(nodes)
  names(node_index) <- nodes

  # Template reused each prime to avoid rebuilding full data frames repeatedly.
  start_activation <- rep(0, length(nodes))
  start_run <- data.frame(node = nodes, activation = start_activation)

  total <- length(word_list)
  run_start <- Sys.time()
  message(sprintf("[%s] Starting spread for %d primes (nodes=%d, t=%d)", run_label, total, length(nodes), t))
  flush.console()
  for (i in seq_along(word_list)) {
    word <- tolower(word_list[i])
    idx <- node_index[[word]]
    if (is.null(idx) || is.na(idx)) {
      # Keep matrix dimensions stable if a prime is OOV for a graph.
      df_final[[word]] <- rep(0, length(nodes))
      if (i == 1 || i %% progress_every == 0 || i == total) {
        elapsed <- as.numeric(difftime(Sys.time(), run_start, units = "secs"))
        message(sprintf("[%s] %d/%d (OOV: %s) elapsed=%.1fs", run_label, i, total, word, elapsed))
        flush.console()
      }
      next
    }
    start_run$activation[] <- 0
    start_run$activation[idx] <- act
    result <- spreadr(adj, start_run, time = t, include_t0 = TRUE)
    final_t <- max(result$time)
    final_rows <- result[result$time == final_t, c("node", "activation")]
    # Align to original node order for stable output matrices.
    final_act <- final_rows$activation[match(nodes, final_rows$node)]
    final_act[is.na(final_act)] <- 0
    df_final[[word]] <- final_act
    if (i == 1 || i %% progress_every == 0 || i == total) {
      elapsed <- as.numeric(difftime(Sys.time(), run_start, units = "secs"))
      message(sprintf("[%s] %d/%d prime=%s elapsed=%.1fs", run_label, i, total, word, elapsed))
      flush.console()
    }
  }
  total_elapsed <- as.numeric(difftime(Sys.time(), run_start, units = "secs"))
  message(sprintf("[%s] Finished in %.1fs", run_label, total_elapsed))
  flush.console()
  df_final
}

args_all <- commandArgs(trailingOnly = FALSE)
args <- commandArgs(trailingOnly = TRUE)
file_arg <- args_all[grep("^--file=", args_all)][1]
if (length(file_arg) == 0 || is.na(file_arg)) {
  stop("Could not resolve script path from command arguments.")
}
script_path <- sub("^--file=", "", file_arg)
script_dir <- dirname(normalizePath(script_path))
project_root <- normalizePath(file.path(script_dir, ".."))

analyses_dir <- file.path(project_root, "data", "LDT_analyses")
matrices_dir <- file.path(analyses_dir, "FA_matrices")
graphs_dir <- file.path(project_root, "data", "graphs", "igraphs")
dir.create(matrices_dir, recursive = TRUE, showWarnings = FALSE)

ldt_primes_path <- file.path(analyses_dir, "LDT_primes.csv")
gender_primes_path <- file.path(analyses_dir, "gender_primes.csv")

ldt_df <- read.csv(ldt_primes_path, header = TRUE, stringsAsFactors = FALSE)
gender_df <- read.csv(gender_primes_path, header = TRUE, stringsAsFactors = FALSE)

if (!("prime" %in% names(ldt_df))) {
  stop("Expected 'prime' column in LDT_primes.csv")
}
if (!("genderWord" %in% names(gender_df))) {
  stop("Expected 'genderWord' column in gender_primes.csv")
}

ldt_primes <- unique(tolower(trimws(as.character(ldt_df$prime))))
gender_primes <- unique(tolower(trimws(as.character(gender_df$genderWord))))

models <- c("humans", "mistral", "llama3", "haiku", "gpt5_mini", "gemma")
model_labels <- list(
  humans = "Humans",
  mistral = "Mistral",
  llama3 = "Llama3",
  haiku = "Haiku",
  gpt5_mini = "Gpt5_mini",
  gemma = "Gemma"
)

selected_models <- models
for (arg in args) {
  if (grepl("^--models=", arg)) {
    raw <- sub("^--models=", "", arg)
    requested <- tolower(trimws(unlist(strsplit(raw, ","))))
    requested <- requested[requested != ""]
    unknown <- setdiff(requested, models)
    if (length(unknown) > 0) {
      stop(sprintf("Unknown model(s): %s. Valid values: %s",
                   paste(unknown, collapse = ", "),
                   paste(models, collapse = ", ")))
    }
    selected_models <- requested
  }
}

for (model in selected_models) {
  graph_path <- file.path(graphs_dir, paste0("FA_", model, "_ig.graphml"))
  if (!file.exists(graph_path)) {
    warning(paste("Skipping", model, "- graph not found:", graph_path))
    next
  }
  g <- read_graph(graph_path, format = "graphml")

  model_label <- model_labels[[model]]
  ldt_out <- file.path(matrices_dir, paste0(model_label, "_LDT.csv"))
  gender_out <- file.path(matrices_dir, paste0(model_label, "_gender.csv"))
  message(sprintf("=== Model %s ===", model_label))
  flush.console()

  ldt_mat <- spreadSimple(g, ldt_primes, run_label = paste0(model_label, ":LDT"), progress_every = 10)
  write.csv(ldt_mat, ldt_out, row.names = FALSE)

  gender_mat <- spreadSimple(g, gender_primes, run_label = paste0(model_label, ":gender"), progress_every = 5)
  write.csv(gender_mat, gender_out, row.names = FALSE)

  message(paste("Wrote:", ldt_out))
  message(paste("Wrote:", gender_out))
  flush.console()
}
