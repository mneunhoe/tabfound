.onLoad <- function(libname, pkgname) {
  register_tabpfn_backend()
  register_tabpfn26_backend()
  register_tabpfn3_backend()
  register_tabpfn35_backend()
  register_tabfm_backend()
  register_tabicl_backend()
  register_mitra_backend()
  invisible()
}
