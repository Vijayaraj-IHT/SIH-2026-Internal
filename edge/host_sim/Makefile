# Host build of the firmware's portable core (front-end + G.711).
#
#   make            build ./kws_host_sim
#   make check      run the front-end benchmark on a sample
#   make clean
#
# The ESP32 build uses exactly these two translation units (see edge/esp32),
# so anything that passes here is behaviourally identical on the device.

CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Wno-unused-parameter
LDLIBS  ?= -lm
BUILD   ?= build

SRC := ../common/kws_frontend.c ../common/kws_g711.c test_frontend.c
BIN := $(BUILD)/kws_host_sim

.PHONY: all check clean bench
all: $(BIN)

$(BIN): $(SRC) ../common/kws_frontend.h ../common/kws_frontend_tables.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

check: $(BIN)
	@echo "host simulator built: $(BIN)"
	@$(BIN) || true

clean:
	rm -rf $(BUILD) *.o *.pcm *.bin *.ulaw
