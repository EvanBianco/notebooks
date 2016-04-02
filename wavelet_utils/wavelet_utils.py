
import numpy as np
from scipy import fft, ifft
from obspy.io.segy import segy
import matplotlib.pyplot as plt


def norm(data):
    return data / np.amax(data)


def get_seismic(target):
    # Copied verbatim from Seisplot

    # Read the file.
    section = segy._read_segy(target, unpack_headers=True)

    # Make the data array.
    data = np.vstack([t.data for t in section.traces]).T

    # Collect some other data. Use a for loop because there are several.
    elev, esp, ens, tsq = [], [], [], []
    for i, trace in enumerate(section.traces):
        elev.append(trace.header.receiver_group_elevation)
        esp.append(trace.header.energy_source_point_number)
        ens.append(trace.header.ensemble_number)
        tsq.append(trace.header.trace_sequence_number_within_line)

    return data


def get_sample_rate_in_seconds(target):
    section = segy._read_segy(target, unpack_headers=True)
    return 1e-6 * section.traces[0].header.sample_interval_in_ms_for_this_trace


def get_nsamples(target):
    section = segy._read_segy(target, unpack_headers=True)
    return section.traces[0].header.number_of_samples_in_this_trace


def get_ntraces(target):
    section = segy._read_segy(target, unpack_headers=True)
    return len(section.traces)


def get_tbase(target):
    # Read the file.
    section = segy._read_segy(target, unpack_headers=True)
    nsamples = section.traces[0].header.number_of_samples_in_this_trace
    dt = section.traces[0].header.sample_interval_in_ms_for_this_trace
    return 0.001 * np.arange(0, nsamples * dt, dt)


def get_trace_header_stuff(target):
    # Copied verbatim from Seisplot

    # Read the file.
    section = segy._read_segy(target, unpack_headers=True)

    # Make the data array.
    data = np.vstack([t.data for t in section.traces]).T

    # Collect some other data. Use a for loop because there are several.
    elev, esp, ens, tsq = [], [], [], []
    for i, trace in enumerate(section.traces):
        elev.append(trace.header.receiver_group_elevation)
        esp.append(trace.header.energy_source_point_number)
        ens.append(trace.header.ensemble_number)
        tsq.append(trace.header.trace_sequence_number_within_line)

    return elev, esp, ens, tsq


def seismic_and_well_plot(seismic,
                          ilx=1176,
                          xlrange=(1598, 1001),
                          well_name='L-30',
                          gain=20,
                          tstart=0, tstop=3.0,
                          tidx1=0, tidx2=750,
                          aspect=250.0,
                          log=None,
                          t_r=None,
                          gain_imp=50):

        x = ilx - xlrange[1]

        # make figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        # plot seismic
        ax.imshow(norm(seismic)[tidx1:tidx2, :], cmap='Greys', aspect=aspect,
                  clim=(-0.5, 0.5),
                  extent=[xlrange[1], xlrange[0], tstop, tstart])
        # plot well
        ax.axvline(ilx, color='k', lw=0.25)
        # gained plot of impedance
        if log is not None:
            ax.plot(gain_imp * (norm(log[1:])) + ilx, t_r, 'k', lw=0.5)
            ax.text(ilx, 3.0, s=r'$\rho V_{P}$',
                    ha='center', va='bottom', fontsize=14,
                    bbox=dict(facecolor='white', alpha=1.0, lw=0.5))
        # label at the top of well
        ax.text(ilx, 0, s=well_name, ha='center', va='top',
                bbox=dict(facecolor='white', alpha=1.0, lw=0.5)
                )
        # annotationing
        ax.set_xlim((xlrange[1], xlrange[0]))
        ax.set_ylim((tstop, tstart))
        ax.invert_xaxis()
        ax.set_ylabel('two-way time (s)')
        ax.set_xlabel('X-line')
        ax.text(0.025, 0.975, s='Inline: ' + str(ilx),
                ha='left', va='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.5, lw=0.0)
                )

        return ax


def in_ms(x):
    return 1000.0 * x

def get_faxis(signal, dt):
    return np.fft.fftfreq(len(signal), d=dt)

def get_spectrum(data, x, dt, ilx, r=1, return_ax=True):
    ntraces = data.shape[1]
    Ss = np.zeros(data.shape[0])
    for trace in data[:, x - r: x + r + 1].T:
        Ss += abs(np.fft.fft(trace))
    Ss /= ntraces
    s = data[:, x]
    faxis = get_faxis(s, dt)
    Y = np.log10(Ss[0:len(faxis) // 2])  # power spectrum
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.plot(faxis[:len(faxis)//2], Y,'m', lw=1)
    ax.set_xlabel('frequency [Hz]', fontsize=12)
    ax.set_ylabel('power [dB]', fontsize=12)
    ax.grid()
    ax.set_title('traces %i to %i' % (ilx - r, ilx + r))
    if return_ax:
        return Ss, ax
    else:
        return Ss

def tstart_wvlt(w, tstart, dt):
    return np.arange(tstart * in_ms(dt), (tstart + len(w)) * in_ms(dt), in_ms(dt))


def plot_trace_segment(data, x, t, dt, w, tw, top_sample, bottom_sample, ax=None, markersize=4):
    s = data[:, x]

    if ax is None:
        fig = plt.figure(figsize=(2.5, 7))
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    ax.plot(norm(s[top_sample:bottom_sample]), t[top_sample:bottom_sample],
            'ko-', ms=markersize)
    ax.axvline(0, color='k')
    ax.set_xlim(-1.5, 1.5)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_ylabel('two-way time (ms)')
    ax.grid()

    ax2.plot(w, tw, 'ko-', ms=4)
    ax2.set_xticks([])
    ax2.set_ylim((in_ms(dt) * top_sample, in_ms(dt) * bottom_sample))
    ax2.set_xlim((-np.amax(w), np.amax(w)))
    ax2.axvline(0, color='k')
    ax2.invert_yaxis()
    ax2.set_yticks([])

    return ax, ax2


def plot_wavelet(wavelet, tbase, ax=None, ylim=(-1.1, 1.1), xlabel=False, 
                 norm=True, points=False, label=None):
    dt = tbase[1] - tbase[0]
    if ax is None:
        fig = plt.figure(figsize=(2.5, 7))
        ax = fig.add_subplot(111)
    if points:
        point = 'ko-'
    else:
        point = 'k'
    ax.plot(tbase, wavelet, point, ms=4, label=label)
    ax.axhline(0, color='k')
    if norm:
        ax.set_xlim(-len(tbase) * dt / 2, len(tbase) * dt / 2)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_yticks([-0.6, 0, 1.0])
    if xlabel:
        ax.set_xlabel('time (s)')
    ax.text(0, 0, s=label, ha='left', va='top', transform=ax.transAxes)
    return ax


def smooth(x, window_len=11,window='hanning'):
    "Smoothes a curve"
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y
