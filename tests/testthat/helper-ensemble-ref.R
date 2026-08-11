# Shared loader for the wrapper-layer parity reference.
#
# Regenerate with:
#
#   .venvs/ref/bin/python inst/parity/ensemble_reference.py \
#       --out inst/parity/ensemble/ensemble.safetensors \
#       --mitra-src <dir holding AutoGluon's mitra/_internal>
#   gzip -9 inst/parity/ensemble/ensemble.safetensors
#
# Needs no model weights, so everything built on it runs anywhere.

ensemble_ref <- function() {
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("safetensors")
  path <- tabfound_file("parity", "ensemble", "ensemble.safetensors.gz")
  skip_if(!nzchar(path), "ensemble reference not found")
  list(
    # Arrays go through safetensors: the fixtures carry NaN, which JSON
    # cannot round-trip.
    t = read_reference_tensors(path),
    s = jsonlite::fromJSON(sub("\\.safetensors\\.gz$", ".json", path),
                           simplifyVector = FALSE)
  )
}

# safetensors -> plain double matrix, dropping the tensor's attributes so
# `expect_equal()` compares numbers rather than metadata.
ref_mat <- function(x) {
  m <- as.array(x)
  if (!is.matrix(m)) m <- matrix(m, ncol = 1L)
  storage.mode(m) <- "double"
  m
}

ref_vec <- function(x) as.numeric(as.array(x))

# Unpack a JSON list-of-lists into an integer vector / logical vector.
ref_int <- function(x) as.integer(unlist(x))
ref_lgl <- function(x) as.logical(unlist(x))
ref_num <- function(x) as.numeric(unlist(x))
