import numpy as np
import random
from BSS import *
import time

NUM_SOURCES = 2
SAMPLE_RATE = 8000
DO_PLOT = False
USE_REVERB_REF = False
RT60 = 0.1
num_mics = 8
factor = 2
    
i = 1
seed = i
random.seed(seed)
np.random.seed(seed)
down_factor = int(2**factor)
fix_scale = True
winsize = 1024 if SAMPLE_RATE == 16000 else 512
hopsize = winsize//4
fftsize = winsize
source_length = 10 # seconds
rt60 = np.random.rand()*0.2 + 0.1

bss = BSS(USE_REVERB_REF)

source_path = r'./LibriSpeechExample'
bss.load_sources(source_path, source_length, num_sources=NUM_SOURCES, fs=SAMPLE_RATE)
bss.create_room(num_mics=num_mics, rt60=rt60, do_plot=DO_PLOT)
bss.generate_mixture(True)
bss.set_stft_setup(winsize=winsize, hopsize=hopsize, fftsize=fftsize)

t_start = time.perf_counter()

# Without WPE
bss.compute_auxiva(down_factor=1, flag=fix_scale, postfix='_long')
bss.compute_auxiva(down_factor=down_factor, flag=fix_scale, postfix='_short')
bss.compute_pconv(down_factor)
bss.compute_proposed(down_factor=down_factor)

# WITH WPE
bss.perform_wpe(down_factor=down_factor)
bss.compute_auxiva(down_factor=1, flag=fix_scale, postfix="_wpe_long")
bss.compute_auxiva(down_factor=down_factor, flag=fix_scale, postfix="_wpe_short")
bss.compute_pconv(postfix="_wpe")
bss.compute_proposed(down_factor=down_factor, postfix="_wpe")

t_end = time.perf_counter()
print("Time usage: {}".format(np.round(t_end-t_start)/60, 2))