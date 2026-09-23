# The memory probe caps host numbers by the process's cgroup. The files
# only exist on Linux, so these tests build fake cgroup trees.

fake_tree <- function(files) {
  root <- tempfile("cg")
  for (f in names(files)) {
    dir.create(dirname(file.path(root, f)), recursive = TRUE, showWarnings = FALSE)
    writeLines(files[[f]], file.path(root, f))
  }
  root
}
GB <- 1e9

test_that("cgroup v2: the process's own cgroup, not the mount root", {
  root <- fake_tree(list(
    "proc/cgroup" = "0::/kubepods/pod1/ctr",
    "cg/memory.max" = "max",
    "cg/kubepods/pod1/memory.max" = as.character(8 * GB),
    "cg/kubepods/pod1/memory.current" = as.character(3 * GB),
    "cg/kubepods/pod1/ctr/memory.max" = "max",
    "cg/kubepods/pod1/ctr/memory.current" = as.character(3 * GB)
  ))
  got <- .cgroup_memory(physical = 64 * GB,
                        proc_cgroup = file.path(root, "proc/cgroup"),
                        mountinfo = file.path(root, "none"),
                        v2_root = file.path(root, "cg"))
  expect_equal(got$limit, 8 * GB)
  expect_equal(got$headroom, 5 * GB)
  expect_identical(got$version, "v2")
})

test_that("the tightest ancestor wins, and uncapped levels are skipped", {
  root <- fake_tree(list(
    "proc/cgroup" = "0::/a/b",
    "cg/a/memory.max" = as.character(4 * GB),
    "cg/a/memory.current" = as.character(3.5 * GB),
    "cg/a/b/memory.max" = as.character(100 * GB),   # >= physical: no cap
    "cg/a/b/memory.current" = as.character(1 * GB)
  ))
  got <- .cgroup_memory(physical = 64 * GB,
                        proc_cgroup = file.path(root, "proc/cgroup"),
                        v2_root = file.path(root, "cg"))
  expect_equal(got$headroom, 0.5 * GB)
  expect_equal(got$limit, 4 * GB)
})

test_that("no limit anywhere means no cap", {
  root <- fake_tree(list("proc/cgroup" = "0::/", "cg/memory.max" = "max"))
  expect_null(.cgroup_memory(physical = 64 * GB,
                             proc_cgroup = file.path(root, "proc/cgroup"),
                             v2_root = file.path(root, "cg")))
  # ...and missing files are silence, not an error.
  expect_null(.cgroup_memory(proc_cgroup = tempfile()))
})

test_that("a limit without readable usage counts as no headroom", {
  root <- fake_tree(list("proc/cgroup" = "0::/x",
                         "cg/x/memory.max" = as.character(2 * GB)))
  got <- .cgroup_memory(physical = 64 * GB,
                        proc_cgroup = file.path(root, "proc/cgroup"),
                        v2_root = file.path(root, "cg"))
  expect_equal(got$headroom, 0)
})

test_that("hybrid hosts fall back to the v1 memory controller", {
  root <- fake_tree(list(
    "proc/cgroup" = c("12:cpu,cpuacct:/docker/abc", "4:memory:/docker/abc", "0::/"),
    "v2/memory.max" = "max",
    "v1/docker/abc/memory.limit_in_bytes" = as.character(2 * GB),
    "v1/docker/abc/memory.usage_in_bytes" = as.character(0.5 * GB)
  ))
  got <- .cgroup_memory(physical = 64 * GB,
                        proc_cgroup = file.path(root, "proc/cgroup"),
                        v2_root = file.path(root, "v2"),
                        v1_root = file.path(root, "v1"))
  expect_identical(got$version, "v1")
  expect_equal(got$headroom, 1.5 * GB)
  # v1's "unlimited" is a huge number, at or above physical: not a cap.
  writeLines("9223372036854771712", file.path(root, "v1/docker/abc/memory.limit_in_bytes"))
  expect_null(.cgroup_memory(physical = 64 * GB,
                             proc_cgroup = file.path(root, "proc/cgroup"),
                             v2_root = file.path(root, "v2"),
                             v1_root = file.path(root, "v1")))
})

test_that("mountinfo resolves bind mounts of a sub-hierarchy", {
  root <- fake_tree(list(
    "proc/cgroup" = "0::/system.slice/app.service",
    "mnt/app.service/memory.max" = as.character(1 * GB),
    "mnt/app.service/memory.current" = as.character(0.25 * GB)
  ))
  mi <- file.path(root, "proc/mountinfo")
  writeLines(sprintf(
    "35 24 0:30 /system.slice %s rw,nosuid - cgroup2 cgroup2 rw",
    file.path(root, "mnt")), mi)
  got <- .cgroup_memory(physical = 64 * GB,
                        proc_cgroup = file.path(root, "proc/cgroup"),
                        mountinfo = mi)
  expect_equal(got$headroom, 0.75 * GB)

  # A cgroup outside the mounted sub-hierarchy is not under it.
  expect_null(.cgroup_under_mount("/m", "/system.slice", "/user.slice/x"))
  expect_identical(.cgroup_under_mount("/m", "/", "/a/b"), file.path("/m", "a", "b"))
})

test_that("mountinfo octal escapes are decoded", {
  m <- .cgroup_mounts("35 24 0:30 / /sys/fs/my\\040cgroup rw - cgroup2 cgroup2 rw",
                      "cgroup2", NULL, "/default")
  expect_identical(m[[1]]$point, "/sys/fs/my cgroup")
})
