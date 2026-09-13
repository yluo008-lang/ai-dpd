# AI-DPD deployment build (C inference + Verilog layer check)
CC  ?= gcc
IV  ?= iverilog

all: c rtl

c: nn_dpd_infer
nn_dpd_infer: nn_dpd_infer.c weights.h
	$(CC) -O2 -I. -o $@ nn_dpd_infer.c -lm

rtl: tb.vvp
tb.vvp: tb_nn_dpd_layer.v nn_dpd_layer.v req_params.vh
	$(IV) -g2005 -o $@ tb_nn_dpd_layer.v nn_dpd_layer.v

run: all
	./nn_dpd_infer
	vvp tb.vvp

clean:
	rm -f nn_dpd_infer tb.vvp

.PHONY: all c rtl run clean
