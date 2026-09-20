# Portable, dependency-free C99 core shared by the firmware and the host tests.
#
# It has no ESP-IDF dependencies on purpose: it is what machine-checked parity
# tests (tests/test_features.py::test_c_frontend_matches_python) compile against
# the Python reference on a laptop, and the same bytes run on the device.

idf_component_register(
    SRCS "kws_frontend.c" "kws_g711.c"
    INCLUDE_DIRS "."
    REQUIRES esp-tflite-micro
)
