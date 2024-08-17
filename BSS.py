import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import soundfile as sf
import scipy.signal as signal
import libss
from libss.separation.metrics import *
from tqdm import tqdm
import random
import os
from nara_wpe.wpe import wpe

def stft(xx, win=None, Nw=1024, Nl=512, Nfft=1024, pad_front=True):
    if len(xx.shape) == 1:
        xx = xx[:, None]
    if win is None or win.shape[0] != Nw:
        win = np.sin(np.arange(Nw)/Nw*np.pi)[:, None]
    if len(win.shape) == 1:
        win = win[:, None]
    Npad_front = Nw - Nl if pad_front else 0
    Npad_end = Nw - Nl
    Nx = int(np.ceil(xx.shape[0]/Nl)*Nl) + Npad_front + Npad_end
    x = np.zeros([Nx, xx.shape[1]])
    x[Npad_front:Npad_front+xx.shape[0], :] = xx
    Nb = int((Nx-Nw)/Nl)
    X = np.zeros([Nfft//2+1, xx.shape[1], Nb], dtype=complex)
    i = 0
    b = 0
    while i + Nw < Nx:
        X[:, :, b] = np.fft.rfft(x[i:i+Nw, :]*win, Nfft, axis=0)
        i += Nl
        b += 1
    return X

def istft(X, win=None, Nw=1024, Nl=512, Nfft=1024):
    if win is None or win.shape[0] != Nw:
        win = np.sin(np.arange(Nw)/Nw*np.pi)[:, None]
    if len(win.shape) == 1:
        win = win[:, None]
    Nk, L, Nb = X.shape
    Nx = Nb*Nl + Nw
    x = np.zeros([Nx, L])
    i = 0
    for b in range(Nb):
        x[i:i+Nw, :] += np.fft.irfft(X[:, :, b], axis=0)[:Nw, :]*win
        i += Nl
    return x

def fix_scaling_ambiguity(X, Y, W, transposed=False):
    # X is a shape of (Nk, num_sources, num_frames)
    # Y is a shape of (Nk, num_sources, num_frames)
    # W is a shape of (Nk, num_sources, num_mics)
    Nk = X.shape[0]
    for k in range(Nk):
        scale = np.diag(np.diag(X[k, ...] @ Y[k, ...].T.conj() @ np.linalg.pinv(Y[k, ...] @ Y[k, ...].T.conj())))
        if transposed:
            W[k, ...] = (scale @ W[k, ...].T.conj()).T.conj()
        else:
            W[k, ...] = scale @ W[k, ...]
    return W

def fix_delay_and_permutation(ref, est, fix_delay=True):
    n = ref.shape[1]
    y = np.zeros_like(ref)
    P = np.zeros([n, n])
    delays = np.zeros(n)
    for i in range(n):
        est[:, i] -= np.mean(est[:, i])
        ref[:, i] *= 0.95/np.max(np.abs(ref[:, i]))
        est[:, i] *= 0.95/np.max(np.abs(est[:, i]))
    for i in range(n):
        idx = -1
        maxcorr = 0
        delay = 0
        for j in range(n):
            tmp = signal.fftconvolve(ref[:, i], est[::-1, j]) # xcorrelation
            if np.max(np.abs(tmp)) > maxcorr:
                maxcorr = np.max(np.abs(tmp))
                if fix_delay:
                    delay = np.argmax(np.abs(tmp)) - ref.shape[0]
                idx = j
        y[:, i] = np.roll(est[:, idx], delay)
        P[i, idx] = 1
        delays[i] = delay
    return y, delays, P

class BSS(object):
    def __init__(self, use_reverb_ref=False):
        self.x = None
        self.Xd = None
        self.use_reverb_ref = use_reverb_ref
        pass

    def create_room(self, num_mics=8, room_dim=[5,3,3], rt60=0.1, do_plot=False):
        self.num_mics = num_mics
        self.room_dim = room_dim
        # self.room_dim[0] += np.random.rand()*1 - 0.5
        # self.room_dim[1] += np.random.rand()*1 - 0.5
        # self.room_dim[2] += np.random.rand()*1 - 0.5
        # print(self.room_dim)
        num_sources = self.num_sources
        fs = self.fs

        rt60_tgt = rt60  # seconds
        room_dim = np.array(room_dim).tolist() # meters

        # We invert Sabine's formula to obtain the parameters for the ISM simulator
        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

        # Create the room
        room = pra.ShoeBox(
            room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
        )

        # place the source in the room
        source_locs = np.array([(np.random.rand(num_sources) * room_dim[i]).tolist() for i in range(len(room_dim))]).T
        for i in range(num_sources):
            room.add_source(source_locs[i, :])

        # define the locations of the microphones
        mic_locs = np.array([(np.random.rand(num_mics) * room_dim[i]).tolist() for i in range(len(room_dim))]).T
        # finally place the array in the room
        room.add_microphone_array(mic_locs.T)

        # Run the simulation (this will also build the RIR automatically)
        room.compute_rir()
        maxLen = np.max(np.array([[len(room.rir[i][j]) for j in range(num_sources)] for i in range(num_mics)]))
        rir = np.zeros([maxLen, num_mics, num_sources])
        for i in range(num_mics):
            for j in range(num_sources):
                rir[:len(room.rir[i][j]), i, j] = room.rir[i][j]
        self.rir = rir
        if do_plot:
            plt.scatter(source_locs[:,0], source_locs[:,1])
            plt.scatter(mic_locs[:,0], mic_locs[:,1])
            plt.xlim(0, room_dim[0])
            plt.ylim(0, room_dim[1])
            plt.savefig("mygraph.png")
            # plt.show()

    def load_sources(self, path, length, num_sources=3, fs=16000):
        self.num_sources = num_sources
        self.fs = fs
        num_samples = length*self.fs
        s = np.zeros([num_samples, self.num_sources])
        file_lists = [f for f in os.listdir(path) if f[-4:] == '.wav' or f[-5:] == '.flac']
        random.shuffle(file_lists)
        for i, f in enumerate(file_lists[:self.num_sources]):
            # print(f)
            tmp, fs_s = sf.read(r'{}/{}'.format(path, f))
            tmp = signal.resample_poly(tmp, self.fs, fs_s)
            minLen = min(num_samples, tmp.shape[0])
            s[:minLen, i] = tmp[:minLen]
        self.s = s

    def generate_mixture(self, save_wav=False):
        num_samples = self.s.shape[0]
        x_raw = np.zeros([num_samples, self.num_mics])
        s_rev = np.zeros([num_samples, self.num_sources, self.num_mics])
        i = 0
        # for j in range(self.num_sources):
            # s_rev[:, j] += signal.fftconvolve(self.s[:, j], self.rir[:,i,j])[:num_samples]
        for i in range(self.num_mics):
            tmp = 0
            for j in range(self.num_sources):
                s_rev[:, j, i] = signal.fftconvolve(self.s[:, j], self.rir[:,i,j])[:num_samples]
                tmp += s_rev[:, j, i]
            x_raw[:, i] = tmp
        self.s_rev = s_rev.copy()
        self.x = x_raw.copy()
        # do reference
        self.idx = random.choice(np.arange(self.num_mics))
        self.mix_ref = self.x[:, self.idx].copy()
        if save_wav:
            sf.write('./save_wav/mixture.wav', x_raw, self.fs)
            for j in range(self.num_sources):
                sf.write('./save_wav/source{}.wav'.format(j), s_rev[:,j,:], self.fs)

    def perform_wpe(self, down_factor=4):
        Nw, Nl, Nfft, winA, winS = self.stft_setup
        Nw_s = Nw//down_factor
        Nl_s = Nl//down_factor
        Nfft_s = Nfft//down_factor
        Nk_s = Nfft_s//2 + 1
        winA_s = np.hamming(Nw_s)
        winS_s = pra.transform.stft.compute_synthesis_window(winA_s, Nl_s)

        delay = 3
        iterations = 5
        taps = 10
        alpha = 0.9999
        X = stft(self.x, winA_s, Nw=Nw_s, Nl=Nl_s, Nfft=Nfft_s)
        self.Xd = wpe(
            X,
            taps=taps,
            delay=delay,
            iterations=iterations,
            statistics_mode='full'
        )
        self.xd = istft(self.Xd, winS_s, Nw=Nw_s, Nl=Nl_s, Nfft=Nfft_s)[:self.x.shape[0], :]
        self.x = self.xd.copy()
        sf.write('./save_wav/mixture_wpe.wav', self.x, self.fs)

    def set_stft_setup(self, winsize=1024, hopsize=256, fftsize=1024):
        Nw = winsize
        Nl = hopsize
        Nfft = fftsize
        winA = np.hamming(Nw)
        winS = pra.transform.stft.compute_synthesis_window(winA, Nl)
        self.stft_setup = [winsize, hopsize, fftsize, winA, winS]

    def compute_auxiva(self, down_factor=1, flag=False, postfix=""):
        Nw, Nl, Nfft, winA, winS = self.stft_setup
        Nw = Nw // down_factor
        Nl = Nl // down_factor
        Nfft = Nfft // down_factor
        Nk = Nfft//2 + 1
        winA = np.hamming(Nw)
        winS = pra.transform.stft.compute_synthesis_window(winA, Nl)

        # Project to lower number of mixture so that num_mics = num_sources
        if self.num_mics > self.num_sources:
            X_raw = stft(self.x, winA, Nw=Nw, Nl=Nl, Nfft=Nfft)
            X = np.zeros([X_raw.shape[0], self.num_sources, X_raw.shape[2]], dtype=complex)
            U = np.zeros([Nk, self.num_sources, self.num_mics], dtype=complex)
            # for k in tqdm(range(Nk)):
            for k in range(Nk):
                u,d,v = np.linalg.svd(X_raw[k, ...] @ X_raw[k, ...].T.conj())
                U[k, ...] = u[:, :self.num_sources].T.conj()
            X = U @ X_raw
        elif self.num_mics == self.num_sources:
            X_raw = stft(self.x, winA, Nw=Nw, Nl=Nl, Nfft=Nfft)
            X = X_raw
            U = np.eye(self.num_sources)[None, ...]
        else:
            return

        # AuxIVA
        mix_tf = np.moveaxis(X, 2, 0)
        separator = libss.separation.AuxIVA(
            mix_tf,
            update_demix_filter="IP1",
            update_source_model="Gauss",
            ref_mic=0,
        )
        n_iter = 50
        # for it in tqdm(range(n_iter)):
        for it in range(n_iter):
            separator.step()
        A = np.linalg.inv(separator.demix_filter)
        A = np.array([np.diag(A[k, 0, :]).tolist() for k in range(Nk)])
        W = A @ separator.demix_filter
        Y = W @ X

        # Fix scaling ambiguity
        if flag:
            W = fix_scaling_ambiguity(X, Y, W, transposed=False)
            Y = W @ X
        y = istft(Y, winS, Nw=Nw, Nl=Nl, Nfft=Nfft)[:self.x.shape[0], :]
        # Fix global permutation and delay
        # s_ref = self.s_rev[..., 0] if self.use_reverb_ref else self.s
        _, _, P = fix_delay_and_permutation(self.s, y)
        
        if down_factor == 1:
            self.W = P @ W @ U
            self.A = np.linalg.pinv(self.W)
            self.Y = P @ W @ X
            self.X = X_raw

        y_gain = y * 0.95/np.max(np.abs(y))
        sf.write('./save_wav/output_auxiva{}.wav'.format(postfix), y_gain, self.fs)

    def compute_proposed(self, down_factor=4, postfix=""):
        Nw, Nl, Nfft, winA, winS = self.stft_setup
        Nw_s = Nw//down_factor
        Nl_s = Nl//down_factor
        Nfft_s = Nfft//down_factor
        Nk_s = Nfft_s//2 + 1
        winA_s = np.hamming(Nw_s)
        winS_s = pra.transform.stft.compute_synthesis_window(winA_s, Nl_s)

        num_samples = self.x.shape[0]
        W_s = np.zeros([Nk_s, self.num_mics, self.num_sources], dtype=complex)
        for i in range(self.num_sources):
            # Mixture of single target source
            Xs = self.A[:, :, [i]] @ self.Y[:, [i], :]
            xs = istft(Xs, winS, Nw=Nw, Nl=Nl, Nfft=Nfft)[:num_samples, :]
            Xs_s = stft(xs, winA_s, Nw=Nw_s, Nl=Nl_s, Nfft=Nfft_s)
            # Mixture without target source
            xo = istft((self.X - Xs), winS, Nw=Nw, Nl=Nl, Nfft=Nfft)[:num_samples, :]
            Xo_s = stft(xo, winA_s, Nw=Nw_s, Nl=Nl_s, Nfft=Nfft_s)
            # First principal component of the target source as steering vector
            Rxs = Xs_s @ np.swapaxes(Xs_s, 1, 2).conj()
            u, d, v = np.linalg.svd(Rxs)
            A_ref = u[:, :, [0]]
            # Compute covariance of the mixture of the other sources
            Rx = (Xo_s) @ np.swapaxes((Xo_s), 1, 2).conj()
            iRx = np.linalg.inv(Rx)
            # Demixing vector estimation
            W_s[:, :, [i]] = iRx @ A_ref / (np.swapaxes(A_ref, 1, 2).conj() @ iRx @ A_ref)

        X_s = stft(self.x, winA_s, Nw=Nw_s, Nl=Nl_s, Nfft=Nfft_s) #if self.Xd is None else self.Xd
        Y_s = np.swapaxes(W_s, 1, 2).conj() @ X_s
        # Fix scaling ambiguity
        W_s = fix_scaling_ambiguity(X_s, Y_s, W_s, transposed=True)
        Y_s = np.swapaxes(W_s, 1, 2).conj() @ X_s

        y_s = istft(Y_s, winS_s, Nw=Nw_s, Nl=Nl_s, Nfft=Nfft_s)[:num_samples, :]
        # Fix global permutation and delay
        # s_ref = self.s_rev[..., 0] if self.use_reverb_ref else self.s
        # y_s, delays, P = fix_delay_and_permutation(s_ref, y_s)

        self.W_s = W_s
        self.X_s = X_s
        self.Y_s = Y_s
        # compute_metrics(s_ref, y_s, self.mix_ref, self.metrics)
        # compute_asteroid_metrics(self.asteroid_metrics, self.s_rev[..., 0], self.s, y_s, self.mix_ref, self.fs, metrics_list=self.metrics_list)
        y_gain = y_s * 0.95/np.max(np.abs(y_s))
        sf.write('./save_wav/output_proposed{}.wav'.format(postfix), y_gain, self.fs)

    def compute_conv(self, postfix=""):
        num_samples = self.x.shape[0]
        Nw = self.stft_setup[0]
        w = np.roll(np.fft.irfft(self.W, axis=0), Nw//2, axis=0)
        idx = np.min(np.argmax(np.reshape(w, [Nw, -1]), axis=0))
        w = np.roll(w, -idx, axis=0)
        w[-idx:, ...] = 0
        y = np.zeros([num_samples, self.num_sources])
        for i in range(self.num_sources):
            tmp = 0
            for j in range(self.num_mics):
                tmp += signal.fftconvolve(self.x[:, j], w[:,i,j])[:num_samples]
            y[:, i] = tmp
        # Fix global permutation and delay
        # s_ref = self.s_rev[..., 0] if self.use_reverb_ref else self.s
        # y, delays, P = fix_delay_and_permutation(s_ref, y)
        y_gain = y * 0.95/np.max(np.abs(y))
        sf.write('./save_wav/output_conv{}.wav'.format(postfix), y_gain, self.fs)

    def compute_pconv(self, down_factor=4, postfix=""):
        w = np.fft.irfft(self.W, axis=0)
        Nw = w.shape[0]
        Nw_s = Nw//down_factor
        Nk_s = Nw_s//2 + 1
        Nb = down_factor * 2
        blockLen = Nw // Nb
        W_s = np.zeros([Nk_s, w.shape[1], w.shape[2]*Nb], dtype=complex)
        w = np.roll(w, Nw//2, axis=0)
        # idx = np.min(np.argmax(np.reshape(w, [Nw, -1]), axis=0))
        idx = -Nw//4
        w = np.roll(w, -idx, axis=0)
        w[-idx:, ...] = 0
        for b in range(Nb):
            W_s[:, :, b*w.shape[2]:(b+1)*w.shape[2]] = np.fft.rfft(w[b*blockLen:(b+1)*blockLen, ...], n=Nw_s, axis=0)

        y = np.zeros([self.x.shape[0], self.num_sources])
        X_s = np.zeros([Nk_s, self.num_mics*Nb, 1], dtype=complex)
        i = 0
        Nhalf = Nw_s//2
        while i + Nw_s < self.x.shape[0]:
            X_s[:, :self.num_mics, 0] = np.fft.rfft(self.x[i:i+Nw_s, :], axis=0)
            y[i:i+Nhalf, :] = np.fft.irfft(W_s @ X_s, axis=0)[Nhalf:, :, 0]
            X_s[:, self.num_mics:, 0] = X_s[:, :-self.num_mics, 0]
            i += Nhalf

        # Fix global permutation and delay
        # s_ref = self.s_rev[..., 0] if self.use_reverb_ref else self.s
        # y, delays, P = fix_delay_and_permutation(s_ref, y)
        y_gain = y * 0.95/np.max(np.abs(y))
        sf.write('./save_wav/output_pconv{}.wav'.format(postfix), y_gain, self.fs)