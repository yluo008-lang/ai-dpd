"""export_only.py - reload the trained RVTDNN and (re)run quantisation + export
without retraining.  Produces deploy/weights.h, *.mem, ref_io.txt, layer0_ref.txt.
"""
import dsp, pa, quant_export
from models import RVTDNN

m = RVTDNN(4, 32, "relu")
m.load_state_dict(__import__("torch").load("RVTDNN.pt"))
x_tr = dsp.scale(dsp.gen_ofdm(8192, seed=11), 0.25)
print("quantise + export (RVTDNN):")
quant_export.quantize_and_export(m, y_in=None, x_cal=x_tr)
