"""Constants shared by fetch and dataset, kept free of third-party imports."""

# Lead offsets in days. Target day D+h is forecast with offset h+1, so every value comes
# from a model run that was available at the 06:00 issue time on day D.
OFFSETS = range(2, 8)
WEATHER_MODEL = "ecmwf_ifs025"
