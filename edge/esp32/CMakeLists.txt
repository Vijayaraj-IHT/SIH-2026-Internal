# ESP-IDF project for the SIH26172 voice activator (ESP32-S3 reference target).
#
#   idf.py set-target esp32s3
#   idf.py -DKWS_WIFI_SSID=... -DKWS_WIFI_PASS=... -DKWS_SERVER_URL=... build flash monitor
#
# TFLite-Micro comes from Espressif's component registry (idf_component.yml in
# main/), which pins a known-good revision so the firmware builds offline once
# the components are vendored (`idf.py reconfigure` with the component cache).

cmake_minimum_required(VERSION 3.16)

# The portable front-end and the mu-law codec live outside main/ so the host
# simulator and the firmware compile byte-identical sources.
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../common")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(sih26172_voice_activator)
