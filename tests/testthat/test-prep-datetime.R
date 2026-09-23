# The datetime encoder, rule by rule.
#
# Each test pins one of the places where skrub's `DatetimeEncoder` does
# something other than the obvious thing, so a regression names its own
# cause. The encoder is also checked against the reference's actual output,
# inside the mixed frame in `test-prep-frame.R`.

test_that("a timestamp yields skrub's features, in skrub's order", {
  ts <- as.POSIXct("2024-01-31 10:15:30", tz = "UTC")
  fit <- fit_datetime_encoder(ts, "ts")
  expect_identical(fit$outputs, c(
    "ts_year", "ts_minute", "ts_second", "ts_total_seconds", "ts_day_of_year",
    "ts_month_circular_0", "ts_month_circular_1",
    "ts_day_circular_0", "ts_day_circular_1",
    "ts_hour_circular_0", "ts_hour_circular_1",
    "ts_weekday_circular_0", "ts_weekday_circular_1"
  ))
  out <- transform_datetime_encoder(ts, fit)
  expect_equal(unname(out[1, c("ts_year", "ts_minute", "ts_second", "ts_day_of_year")]),
               c(2024, 15, 30, 31))
  # Circular features are `sin(v / period * 2 * pi)` and its cosine.
  expect_equal(unname(out[1, "ts_month_circular_0"]), sin(1 / 12 * 2 * pi))
  expect_equal(unname(out[1, "ts_hour_circular_1"]), cos(10 / 24 * 2 * pi))
})

test_that("seconds since the epoch are float32, as the reference stores them", {
  # 2024-01-31 10:15:30 UTC is 1706696130 seconds; float32 near 1.7e9 has a
  # resolution of 128, so the reference stores 1706696192. A port that kept
  # the double would be more precise than its reference, and so different.
  ts <- as.POSIXct("2024-01-31 10:15:30", tz = "UTC")
  out <- transform_datetime_encoder(ts, fit_datetime_encoder(ts, "ts"))
  expect_identical(unname(out[1, "ts_total_seconds"]), 1706696192)
  # Two timestamps a minute apart can therefore come out identical.
  two <- ts + c(0, 60)
  out2 <- transform_datetime_encoder(two, fit_datetime_encoder(two, "ts"))
  expect_identical(out2[1, "ts_total_seconds"], out2[2, "ts_total_seconds"])
})

test_that("weekdays run Monday = 1 to Sunday = 7, and the day period is 30", {
  # 2024-01-01 is a Monday, 2024-01-07 a Sunday. R counts Sunday as 0.
  d <- as.Date(c("2024-01-01", "2024-01-07"))
  out <- transform_datetime_encoder(d, fit_datetime_encoder(d, "d"))
  expect_equal(unname(out[, "d_weekday_circular_0"]), sin(c(1, 7) / 7 * 2 * pi))
  # The 31st is past a full turn of a 30-day period, not at the end of one.
  d31 <- as.Date("2024-01-31")
  o31 <- transform_datetime_encoder(d31, fit_datetime_encoder(d31, "d"))
  expect_equal(unname(o31[1, "d_day_circular_0"]), sin(31 / 30 * 2 * pi))
})

test_that("a column entirely at midnight is a date, and gets no time of day", {
  midnight <- as.POSIXct(c("2024-01-01", "2024-02-01"), tz = "UTC")
  fit <- fit_datetime_encoder(midnight, "m")
  expect_true(fit$date_only)
  # Anchored: `total_seconds` is not a time-of-day feature.
  expect_false(any(grepl("_(hour|minute|second)(_|$)", fit$outputs)))
  # Replayed at predict even when a later value has a time: the width is
  # decided at fit.
  later <- as.POSIXct("2024-03-01 13:45:00", tz = "UTC")
  expect_identical(colnames(transform_datetime_encoder(later, fit)), fit$outputs)
  # One value off midnight, fractional second included, and it is not.
  frac <- as.POSIXct("2024-01-01 00:00:00.5", tz = "UTC")
  expect_false(fit_datetime_encoder(c(midnight, frac), "m")$date_only)
  # A `Date` is always a date.
  expect_true(fit_datetime_encoder(as.Date("2024-01-01"), "d")$date_only)
})

test_that("calendar features are read in the column's own timezone", {
  # 03:30 UTC is 22:30 the previous evening in New York, across midnight.
  utc <- as.POSIXct("2024-03-10 03:30:00", tz = "UTC")
  ny <- utc; attr(ny, "tzone") <- "America/New_York"
  o_utc <- transform_datetime_encoder(utc, fit_datetime_encoder(utc, "t"))
  o_ny  <- transform_datetime_encoder(ny,  fit_datetime_encoder(ny,  "t"))
  expect_equal(unname(o_utc[1, "t_day_of_year"]), 70)
  expect_equal(unname(o_ny[1, "t_day_of_year"]), 69)
  # The epoch count is an instant, and does not move with the zone.
  expect_identical(o_utc[1, "t_total_seconds"], o_ny[1, "t_total_seconds"])
})

test_that("a missing value makes every feature of its row missing", {
  ts <- as.POSIXct(c("2024-01-31 10:15:30", NA), tz = "UTC")
  out <- transform_datetime_encoder(ts, fit_datetime_encoder(ts, "ts"))
  expect_true(all(is.na(out[2, ])))
  expect_false(anyNA(out[1, ]))
})

test_that("new rows in another zone, or of another class, are refused", {
  ts <- as.POSIXct("2024-01-31 10:15:30", tz = "UTC")
  fit <- fit_datetime_encoder(ts, "ts")
  other <- ts; attr(other, "tzone") <- "Europe/Berlin"
  expect_error(transform_datetime_encoder(other, fit), "timezone")
  expect_error(transform_datetime_encoder(as.Date("2024-01-31"), fit),
               "POSIXct.*Date|Date.*POSIXct")
  expect_error(transform_datetime_encoder(1:3, fit), "held dates")
})

test_that("a duration is its length in seconds, whatever its units", {
  expect_identical(duration_seconds(as.difftime(1.5, units = "mins")), 90)
  expect_identical(duration_seconds(as.difftime(2, units = "hours")), 7200)
  # `as.numeric()` alone returns the magnitude in the object's own unit;
  # the encoder no longer does.
  expect_identical(
    unname(.encode_predictors(data.frame(d = as.difftime(1.5, units = "mins")))[1, 1]),
    90
  )
})
