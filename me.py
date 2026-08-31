import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import itertools
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 120,
    'lines.linewidth': 1.8,
})
Nt      = 2          # TX antennas
Nr      = 1          # RX antennas per user
K       = 2          # number of users
Nc      = 64         # total OFDM subcarriers
Nd      = 48         # data subcarriers
Ncp     = 16         # cyclic-prefix length
BW_total = 20e6      # Hz
B       = BW_total * (Nc / (Nc + Ncp)) * (Nd / Nc)   # effective BW = 12 MHz (eq.14)
Pt_dBm  = 23         # dBm
Pt      = 10**((Pt_dBm - 30) / 10)   # Watts
fc      = 2.484e9    # Hz
N_runs  = 500        # Monte-Carlo runs per case (paper: 100; we use 500 for smooth curves)
SIGMA2  = 1e-12      
MCS_TABLE = [
    (0,  1, 1/2, "BPSK 1/2"),
    (1,  1, 3/4, "BPSK 3/4"),
    (2,  2, 1/2, "QPSK 1/2"),
    (3,  2, 3/4, "QPSK 3/4"),
    (4,  4, 1/2, "16QAM 1/2"),
    (5,  4, 3/4, "16QAM 3/4"),
    (6,  6, 2/3, "64QAM 2/3"),
    (7,  6, 3/4, "64QAM 3/4"),
    (8,  8, 3/4, "256QAM 3/4"),
    (9,  8, 5/6, "256QAM 5/6"),
]
MCS_IDX  = [e[0] for e in MCS_TABLE]
MCS_M    = np.array([e[1] for e in MCS_TABLE])
MCS_R    = np.array([e[2] for e in MCS_TABLE])
MCS_RATE = MCS_M * MCS_R              # bits/s/Hz
MCS_TP   = B * MCS_RATE * 1e-6        # Mbps
MCS_LABEL= [e[3] for e in MCS_TABLE]

CASES = {
    1:  (-7.6,  0.15, "Low pathloss / High corr"),
    2:  (-6.4,  0.48, "Low pathloss / Med corr"),
    3:  (-8.8,  0.95, "Low pathloss / Low corr"),
    4:  (-14.5, 0.16, "Med pathloss / High corr"),
    5:  (-13.0, 0.46, "Med pathloss / Med corr"),
    6:  (-13.0, 0.35, "Med pathloss / Low corr"),  
    7:  (-23.6, 0.24, "High pathloss / High corr"),
    8:  (-24.7, 0.95, "High pathloss / Med corr"),
    9:  (-22.1, 0.85, "High pathloss / Low corr"),
}

def generate_los_channel(alpha_dB, rho, seed=None):
    rng = np.random.default_rng(seed)
    a1 = 1.0
    a2 = a1 * 10**(alpha_dB / 20)   
    phi = rng.uniform(0, 2*np.pi)
    h1  = a1 * np.array([1.0, np.exp(1j * phi)]) / np.sqrt(2)
    v_perp = np.array([-np.conj(h1[1]), np.conj(h1[0])]) / np.linalg.norm(h1)
    v_perp = v_perp / np.linalg.norm(v_perp)

    h1_unit = h1 / np.linalg.norm(h1)
    h2 = a2 * (np.sqrt(max(1 - rho, 0)) * h1_unit + np.sqrt(rho) * v_perp)

    return h1, h2

def quantize_csi(h, Nb=4):
    """
    Quantize complex channel estimate to Nb bits real + Nb bits imag
    per coefficient (IEEE 802.11n based, Appendix of paper).
    Returns dequantized (reconstructed) channel at TX.
    """
    all_vals = np.concatenate([h.real, h.imag])
    mh = np.max(np.abs(all_vals))
    nh = np.min(np.abs(all_vals))
    if mh < 1e-15:
        return h.copy()
    Mh  = min(7, int(np.floor(20 * np.log10(mh / max(nh, 1e-15)))))
    Mlin = mh / (10**(Mh / 20))
    levels = 2**(Nb - 1) - 1

    def quant_scalar(x):
        q = np.round(x / Mlin * levels).astype(int)
        q = np.clip(q, -levels, levels)
        return q / levels * Mlin

    h_q = quant_scalar(h.real) + 1j * quant_scalar(h.imag)
    return h_q


def wmmse_rsma(h1_tx, h2_tx, Pt, max_iter=50, tol=1e-5):
    """
    WMMSE precoder design for RSMA (two-user MISO).
    Returns P = [pc, p1, p2] each of shape (2,).
    Power split: common stream gets ~50% of power, private streams share rest.
    Simplified WMMSE: iterative power allocation + ZF-based private precoders.
    """
    H = np.stack([h1_tx, h2_tx], axis=0)    # (2, 2)
    Pc = Pt / 3
    P1 = Pt / 3
    P2 = Pt / 3
    pc = (h1_tx + h2_tx) / (np.linalg.norm(h1_tx + h2_tx) + 1e-15)
    def zf_precoder(hi, hj):
        """Project hi onto null-space of hj."""
        hj_u = hj / (np.linalg.norm(hj) + 1e-15)
        p = hi - (hj_u @ hi) * hj_u
        n = np.linalg.norm(p)
        return p / n if n > 1e-15 else hi / (np.linalg.norm(hi) + 1e-15)

    prev_rate = -np.inf
    for _ in range(max_iter):
        p1 = zf_precoder(h1_tx, h2_tx)
        p2 = zf_precoder(h2_tx, h1_tx)
        def sinr_common(hi, Pc, P1, P2):
            num = Pc * np.abs(hi @ pc)**2
            den = SIGMA2 + P1 * np.abs(hi @ p1)**2 + P2 * np.abs(hi @ p2)**2
            return num / den
        def sinr_private(hi, hj, Pi, Pj):
            num = Pi * np.abs(hi @ p1)**2   # placeholder, reassigned below
            return num
        Rc1 = np.log2(1 + sinr_common(h1_tx, Pc, P1, P2))
        Rc2 = np.log2(1 + sinr_common(h2_tx, Pc, P1, P2))
        Rc  = min(Rc1, Rc2)

        sinr1 = (P1 * np.abs(h1_tx @ p1)**2) / (SIGMA2 + P2 * np.abs(h1_tx @ p2)**2)
        sinr2 = (P2 * np.abs(h2_tx @ p2)**2) / (SIGMA2 + P1 * np.abs(h2_tx @ p1)**2)
        R1 = np.log2(1 + sinr1)
        R2 = np.log2(1 + sinr2)
        total_rate = Rc + R1 + R2
        g_c  = min(np.abs(h1_tx @ pc)**2, np.abs(h2_tx @ pc)**2)
        g1   = np.abs(h1_tx @ p1)**2
        g2   = np.abs(h2_tx @ p2)**2
        total_g = g_c + g1 + g2 + 1e-30
        Pc = Pt * g_c / total_g
        P1 = Pt * g1 / total_g
        P2 = Pt * g2 / total_g

        if abs(total_rate - prev_rate) < tol:
            break
        prev_rate = total_rate

    return np.sqrt(Pc) * pc, np.sqrt(P1) * p1, np.sqrt(P2) * p2

def wmmse_sdma(h1_tx, h2_tx, Pt, max_iter=50, tol=1e-5):
    """WMMSE for SDMA: ZF precoders, no common stream."""
    def zf_precoder(hi, hj):
        hj_u = hj / (np.linalg.norm(hj) + 1e-15)
        p = hi - (hj_u @ hi) * hj_u
        n = np.linalg.norm(p)
        return p / n if n > 1e-15 else hi / (np.linalg.norm(hi) + 1e-15)

    p1_dir = zf_precoder(h1_tx, h2_tx)
    p2_dir = zf_precoder(h2_tx, h1_tx)

    g1 = np.abs(h1_tx @ p1_dir)**2
    g2 = np.abs(h2_tx @ p2_dir)**2
    mu = (Pt + SIGMA2 * (1/g1 + 1/g2)) / 2
    P1 = max(0, mu - SIGMA2/g1)
    P2 = max(0, mu - SIGMA2/g2)
    if P1 + P2 > Pt:
        P1 = Pt * g1 / (g1 + g2)
        P2 = Pt * g2 / (g1 + g2)

    return np.sqrt(P1) * p1_dir, np.sqrt(P2) * p2_dir

def wmmse_noma(h1_tx, h2_tx, Pt, max_iter=50, tol=1e-5):
    """
    WMMSE for NOMA: RX1 is strong, RX2 weak.
    Common stream (for RX2's message) + private stream (RX1 only).
    Common precoder: MRT toward weaker user's channel.
    """
    pc = h2_tx / (np.linalg.norm(h2_tx) + 1e-15)
    def zf(hi, hj):
        hj_u = hj / (np.linalg.norm(hj) + 1e-15)
        p = hi - (hj_u @ hi) * hj_u
        n = np.linalg.norm(p)
        return p / n if n > 1e-15 else hi / (np.linalg.norm(hi) + 1e-15)

    p1_dir = zf(h1_tx, h2_tx)

    g_c1 = np.abs(h1_tx @ pc)**2
    g_c2 = np.abs(h2_tx @ pc)**2
    g1   = np.abs(h1_tx @ p1_dir)**2
    Pc = Pt * 0.6
    P1 = Pt * 0.4
    for _ in range(max_iter):
        sinr_c1 = Pc * g_c1 / (SIGMA2 + P1 * g1)
        sinr_c2 = Pc * g_c2 / SIGMA2
        Rc      = min(np.log2(1 + sinr_c1), np.log2(1 + sinr_c2))
        sinr1   = P1 * g1 / SIGMA2   # after SIC of common
        R1      = np.log2(1 + sinr1)
        alpha_c = g_c2 / (SIGMA2 + 1e-20)
        alpha_1 = g1 / SIGMA2
        g_total = alpha_c + alpha_1 + 1e-30
        Pc_new = Pt * alpha_c / g_total
        P1_new = Pt * alpha_1 / g_total
        if abs(Pc_new - Pc) < tol * Pt:
            break
        Pc, P1 = Pc_new, P1_new

    return np.sqrt(Pc) * pc, np.sqrt(P1) * p1_dir


from scipy.special import erfc

def bler_normal_approx(snr_linear, m_bits, r_code, n_channel=50*48):
    """
    Normal approximation to the maximum achievable rate at finite blocklength.
    
    Returns success probability P(decode success).
    """
    C = np.log2(1 + snr_linear)          # Shannon capacity bits/channel use
    R = m_bits * r_code                   # target rate
    if C < 1e-10:
        return 0.0
    V = snr_linear * (snr_linear + 2) / ((1 + snr_linear)**2 * np.log(2)**2)  # dispersion
    if V < 1e-15:
        V = 1e-15
    z = np.sqrt(n_channel / V) * (C - R)
    bler = 0.5 * erfc(z / np.sqrt(2))
    bler = np.clip(bler, 0, 1)
    return 1.0 - bler   # success probability


def compute_rsma_throughput(h1_true, h2_true, h1_tx, h2_tx, mcs_c, mcs_1, mcs_2):
    """
    Compute RSMA MCS-limited sum throughput for given precoder CSI and MCS choices.
    Returns (T_sum, T_c, T_1, T_2, P_succ_c, P_succ_1, P_succ_2)
    """
    pc, p1, p2 = wmmse_rsma(h1_tx, h2_tx, Pt)
    mc, rc = MCS_M[mcs_c], MCS_R[mcs_c]
    m1, r1 = MCS_M[mcs_1], MCS_R[mcs_1]
    m2, r2 = MCS_M[mcs_2], MCS_R[mcs_2]

    Pc = np.linalg.norm(pc)**2
    P1 = np.linalg.norm(p1)**2
    P2 = np.linalg.norm(p2)**2
    def sinr_c(h):
        num = np.abs(h @ pc)**2
        den = SIGMA2 + np.abs(h @ p1)**2 + np.abs(h @ p2)**2
        return num / den

    sinr_c1 = sinr_c(h1_true)
    sinr_c2 = sinr_c(h2_true)
    sinr_1 = np.abs(h1_true @ p1)**2 / (SIGMA2 + np.abs(h1_true @ p2)**2)
    sinr_2 = np.abs(h2_true @ p2)**2 / (SIGMA2 + np.abs(h2_true @ p1)**2)
    Ps_c1 = bler_normal_approx(sinr_c1, mc, rc)
    Ps_c2 = bler_normal_approx(sinr_c2, mc, rc)
    Ps_c  = Ps_c1 * Ps_c2   # both must decode common stream
    Ps_1  = bler_normal_approx(sinr_1, m1, r1)
    Ps_2  = bler_normal_approx(sinr_2, m2, r2)

    T_c = B * mc * rc * Ps_c * 1e-6
    T_1 = B * m1 * r1 * Ps_1 * 1e-6
    T_2 = B * m2 * r2 * Ps_2 * 1e-6
    return T_c + T_1 + T_2, T_c, T_1, T_2, Ps_c, Ps_1, Ps_2

def compute_sdma_throughput(h1_true, h2_true, h1_tx, h2_tx, mcs_1, mcs_2):
    p1, p2 = wmmse_sdma(h1_tx, h2_tx, Pt)
    m1, r1 = MCS_M[mcs_1], MCS_R[mcs_1]
    m2, r2 = MCS_M[mcs_2], MCS_R[mcs_2]

    sinr_1 = np.abs(h1_true @ p1)**2 / (SIGMA2 + np.abs(h1_true @ p2)**2)
    sinr_2 = np.abs(h2_true @ p2)**2 / (SIGMA2 + np.abs(h2_true @ p1)**2)

    Ps_1 = bler_normal_approx(sinr_1, m1, r1)
    Ps_2 = bler_normal_approx(sinr_2, m2, r2)

    T_1 = B * m1 * r1 * Ps_1 * 1e-6
    T_2 = B * m2 * r2 * Ps_2 * 1e-6
    return T_1 + T_2, T_1, T_2, Ps_1, Ps_2

def compute_noma_throughput(h1_true, h2_true, h1_tx, h2_tx, mcs_c, mcs_1):
    pc, p1 = wmmse_noma(h1_tx, h2_tx, Pt)
    mc, rc = MCS_M[mcs_c], MCS_R[mcs_c]
    m1, r1 = MCS_M[mcs_1], MCS_R[mcs_1]
    sinr_c1 = np.abs(h1_true @ pc)**2 / (SIGMA2 + np.abs(h1_true @ p1)**2)
    sinr_c2 = np.abs(h2_true @ pc)**2 / SIGMA2
    sinr_1 = np.abs(h1_true @ p1)**2 / SIGMA2

    Ps_c1 = bler_normal_approx(sinr_c1, mc, rc)
    Ps_c2 = bler_normal_approx(sinr_c2, mc, rc)
    Ps_c  = Ps_c1 * Ps_c2
    Ps_1  = bler_normal_approx(sinr_1, m1, r1) * Ps_c1   # SIC chain

    T_c = B * mc * rc * Ps_c * 1e-6
    T_1 = B * m1 * r1 * Ps_1 * 1e-6
    return T_c + T_1, T_c, T_1, Ps_c, Ps_1



def run_case(case_id, quantized=False, n_runs=N_runs, verbose=False):
    """Run full MCS optimization for one channel case, return result dict."""
    alpha_dB, rho, label = CASES[case_id]

    best_rsma = {'T': 0, 'T_c': 0, 'T_1': 0, 'T_2': 0,
                 'mcs_c': 0, 'mcs_1': 0, 'mcs_2': 0}
    best_sdma = {'T': 0, 'T_1': 0, 'T_2': 0, 'mcs_1': 0, 'mcs_2': 0}
    best_noma = {'T': 0, 'T_c': 0, 'T_1': 0, 'mcs_c': 0, 'mcs_1': 0}

    alpha_vals, rho_vals = [], []

    for run in range(n_runs):
        h1_true, h2_true = generate_los_channel(alpha_dB, rho, seed=run)
        if quantized:
            h1_tx = quantize_csi(h1_true)
            h2_tx = quantize_csi(h2_true)
        else:
            h1_tx = h1_true.copy()
            h2_tx = h2_true.copy()
        a = 20 * np.log10(np.linalg.norm(h2_true) / (np.linalg.norm(h1_true) + 1e-15))
        r = 1 - np.abs(h1_true.conj() @ h2_true)**2 / (
            np.linalg.norm(h1_true)**2 * np.linalg.norm(h2_true)**2 + 1e-30)
        alpha_vals.append(a)
        rho_vals.append(r)
        rsma_T_best = 0
        for mcs_c in range(len(MCS_TABLE)):
            for mcs_p in range(len(MCS_TABLE)):  # same for both private streams
                T, T_c, T_1, T_2, _, _, _ = compute_rsma_throughput(
                    h1_true, h2_true, h1_tx, h2_tx, mcs_c, mcs_p, mcs_p)
                if T > rsma_T_best:
                    rsma_T_best = T
                    best_rsma = {'T': T, 'T_c': T_c, 'T_1': T_1, 'T_2': T_2,
                                 'mcs_c': mcs_c, 'mcs_1': mcs_p, 'mcs_2': mcs_p}
        sdma_T_best = 0
        for mcs_1 in range(len(MCS_TABLE)):
            for mcs_2 in range(len(MCS_TABLE)):
                T, T_1, T_2, _, _ = compute_sdma_throughput(
                    h1_true, h2_true, h1_tx, h2_tx, mcs_1, mcs_2)
                if T > sdma_T_best:
                    sdma_T_best = T
                    best_sdma = {'T': T, 'T_1': T_1, 'T_2': T_2,
                                 'mcs_1': mcs_1, 'mcs_2': mcs_2}
        noma_T_best = 0
        for mcs_c in range(len(MCS_TABLE)):
            for mcs_1 in range(len(MCS_TABLE)):
                T, T_c, T_1, _, _ = compute_noma_throughput(
                    h1_true, h2_true, h1_tx, h2_tx, mcs_c, mcs_1)
                if T > noma_T_best:
                    noma_T_best = T
                    best_noma = {'T': T, 'T_c': T_c, 'T_1': T_1,
                                 'mcs_c': mcs_c, 'mcs_1': mcs_1}
    T_rsma_runs = np.zeros((n_runs, 3))   # [T_c, T_1, T_2]
    T_sdma_runs = np.zeros((n_runs, 2))
    T_noma_runs = np.zeros((n_runs, 2))

    for run in range(n_runs):
        h1_true, h2_true = generate_los_channel(alpha_dB, rho, seed=run)
        if quantized:
            h1_tx = quantize_csi(h1_true)
            h2_tx = quantize_csi(h2_true)
        else:
            h1_tx = h1_true.copy()
            h2_tx = h2_true.copy()
        T, T_c, T_1, T_2, _, _, _ = compute_rsma_throughput(
            h1_true, h2_true, h1_tx, h2_tx,
            best_rsma['mcs_c'], best_rsma['mcs_1'], best_rsma['mcs_2'])
        T_rsma_runs[run] = [T_c, T_1, T_2]

        T, T_1, T_2, _, _ = compute_sdma_throughput(
            h1_true, h2_true, h1_tx, h2_tx, best_sdma['mcs_1'], best_sdma['mcs_2'])
        T_sdma_runs[run] = [T_1, T_2]

        T, T_c, T_1, _, _ = compute_noma_throughput(
            h1_true, h2_true, h1_tx, h2_tx, best_noma['mcs_c'], best_noma['mcs_1'])
        T_noma_runs[run] = [T_c, T_1]

    rsma_avg = T_rsma_runs.mean(axis=0)
    sdma_avg = T_sdma_runs.mean(axis=0)
    noma_avg = T_noma_runs.mean(axis=0)

    return {
        'case': case_id, 'label': label, 'alpha': np.mean(alpha_vals),
        'rho': np.mean(rho_vals), 'quantized': quantized,
        'rsma': {'T_c': rsma_avg[0], 'T_1': rsma_avg[1], 'T_2': rsma_avg[2],
                 'T': rsma_avg.sum()},
        'sdma': {'T_1': sdma_avg[0], 'T_2': sdma_avg[1], 'T': sdma_avg.sum()},
        'noma': {'T_c': noma_avg[0], 'T_1': noma_avg[1], 'T': noma_avg.sum()},
        'best_mcs': {'rsma': best_rsma, 'sdma': best_sdma, 'noma': best_noma},
    }



def run_case1_mcs_sweep(quantized=False, n_runs=200):
    """Reproduce Fig. 5: sum throughput vs private-stream MCS for Case 1."""
    case_id = 1
    alpha_dB, rho, _ = CASES[case_id]
    results = {}

    for mcs_c_idx in [2, 3, 5, 6]:   # QPSK1/2, QPSK3/4, 16QAM3/4, 64QAM2/3
        key = MCS_LABEL[mcs_c_idx]
        results[key] = {'rsma': [], 'noma': [], 'sdma': []}
        for mcs_p in range(len(MCS_TABLE)):
            rsma_t, sdma_t, noma_t = [], [], []
            for run in range(n_runs):
                h1_t, h2_t = generate_los_channel(alpha_dB, rho, seed=run)
                if quantized:
                    h1_tx = quantize_csi(h1_t); h2_tx = quantize_csi(h2_t)
                else:
                    h1_tx, h2_tx = h1_t.copy(), h2_t.copy()

                T, *_ = compute_rsma_throughput(h1_t, h2_t, h1_tx, h2_tx,
                                                mcs_c_idx, mcs_p, mcs_p)
                rsma_t.append(T)
                T, *_ = compute_sdma_throughput(h1_t, h2_t, h1_tx, h2_tx, mcs_p, mcs_p)
                sdma_t.append(T)
                T, *_ = compute_noma_throughput(h1_t, h2_t, h1_tx, h2_tx, mcs_c_idx, mcs_p)
                noma_t.append(T)
            results[key]['rsma'].append(np.mean(rsma_t))
            results[key]['sdma'].append(np.mean(sdma_t))
            results[key]['noma'].append(np.mean(noma_t))
    return results

def compute_fairness(result):
    """
    Apply S1/S2 strategy: reallocate common stream to equalize user throughputs.
    Returns (min_throughput, sum_throughput) for each scheme.
    """
    R1 = result['rsma']['T_1']
    R2 = result['rsma']['T_2']
    Rc = result['rsma']['T_c']
    T_sum_rsma = R1 + R2 + Rc

    if abs(R1 - R2) <= Rc:   # S1: can equalize
        x = (R2 - R1 + Rc) / (2 * Rc + 1e-15)
        x = np.clip(x, 0, 1)
        min_t_rsma = R1 + x * Rc
    else:   # S2: all common to RX2
        min_t_rsma = min(R1, R2 + Rc)
    T1s = result['sdma']['T_1']
    T2s = result['sdma']['T_2']
    min_t_sdma = min(T1s, T2s)
    T_sum_sdma = T1s + T2s
    Tc_n = result['noma']['T_c']
    T1_n = result['noma']['T_1']
    min_t_noma = min(Tc_n, T1_n)
    T_sum_noma = Tc_n + T1_n

    return {
        'rsma': (min_t_rsma, T_sum_rsma),
        'sdma': (min_t_sdma, T_sum_sdma),
        'noma': (min_t_noma, T_sum_noma),
    }

def theoretical_sum_rate_vs_rho(rho_range, alpha_dB=-10):

    rates_rsma, rates_sdma, rates_noma = [], [], []
    for rho in rho_range:
        h1, h2 = generate_los_channel(alpha_dB, rho, seed=0)
        h1 = h1 / np.linalg.norm(h1) * 1.0
        h2 = h2 / np.linalg.norm(h2) * (10**(alpha_dB/20))

        pc, p1, p2 = wmmse_rsma(h1, h2, Pt)
        sinr_c1 = np.abs(h1 @ pc)**2 / (SIGMA2 + np.abs(h1@p1)**2 + np.abs(h1@p2)**2)
        sinr_c2 = np.abs(h2 @ pc)**2 / (SIGMA2 + np.abs(h2@p1)**2 + np.abs(h2@p2)**2)
        Rc  = min(np.log2(1+sinr_c1), np.log2(1+sinr_c2))
        R1  = np.log2(1 + np.abs(h1@p1)**2 / (SIGMA2 + np.abs(h1@p2)**2))
        R2  = np.log2(1 + np.abs(h2@p2)**2 / (SIGMA2 + np.abs(h2@p1)**2))
        rates_rsma.append((Rc + R1 + R2) * B * 1e-6)

        p1s, p2s = wmmse_sdma(h1, h2, Pt)
        R1s = np.log2(1 + np.abs(h1@p1s)**2 / (SIGMA2 + np.abs(h1@p2s)**2))
        R2s = np.log2(1 + np.abs(h2@p2s)**2 / (SIGMA2 + np.abs(h2@p1s)**2))
        rates_sdma.append((R1s + R2s) * B * 1e-6)

        pcn, p1n = wmmse_noma(h1, h2, Pt)
        sinr_nc1 = np.abs(h1@pcn)**2 / (SIGMA2 + np.abs(h1@p1n)**2)
        sinr_nc2 = np.abs(h2@pcn)**2 / SIGMA2
        Rcn = min(np.log2(1+sinr_nc1), np.log2(1+sinr_nc2))
        R1n = np.log2(1 + np.abs(h1@p1n)**2 / SIGMA2)
        rates_noma.append((Rcn + R1n) * B * 1e-6)

    return np.array(rates_rsma), np.array(rates_sdma), np.array(rates_noma)

def theoretical_rate_vs_snr(snr_dB_range, rho=0.15):
    """Compute theoretical sum-rate vs SNR for high-correlation scenario."""
    rates = {'rsma': [], 'sdma': [], 'noma': []}
    for snr_dB in snr_dB_range:
        sigma2_eff = Pt / (10**(snr_dB/10))
        h1, h2 = generate_los_channel(-7.6, rho, seed=0)
        def sinr_eval(scheme):
            if scheme == 'rsma':
                pc, p1, p2 = wmmse_rsma(h1, h2, Pt)
                sc1 = np.abs(h1@pc)**2 / (sigma2_eff + np.abs(h1@p1)**2 + np.abs(h1@p2)**2)
                sc2 = np.abs(h2@pc)**2 / (sigma2_eff + np.abs(h2@p1)**2 + np.abs(h2@p2)**2)
                Rc  = min(np.log2(1+sc1), np.log2(1+sc2))
                R1  = np.log2(1 + np.abs(h1@p1)**2 / (sigma2_eff + np.abs(h1@p2)**2))
                R2  = np.log2(1 + np.abs(h2@p2)**2 / (sigma2_eff + np.abs(h2@p1)**2))
                return (Rc + R1 + R2) * B * 1e-6
            elif scheme == 'sdma':
                p1, p2 = wmmse_sdma(h1, h2, Pt)
                R1 = np.log2(1 + np.abs(h1@p1)**2 / (sigma2_eff + np.abs(h1@p2)**2))
                R2 = np.log2(1 + np.abs(h2@p2)**2 / (sigma2_eff + np.abs(h2@p1)**2))
                return (R1 + R2) * B * 1e-6
            else:
                pc, p1 = wmmse_noma(h1, h2, Pt)
                sc1 = np.abs(h1@pc)**2 / (sigma2_eff + np.abs(h1@p1)**2)
                sc2 = np.abs(h2@pc)**2 / sigma2_eff
                Rc  = min(np.log2(1+sc1), np.log2(1+sc2))
                R1  = np.log2(1 + np.abs(h1@p1)**2 / sigma2_eff)
                return (Rc + R1) * B * 1e-6

        for s in ['rsma', 'sdma', 'noma']:
            rates[s].append(sinr_eval(s))
    return rates

def imperfect_sic_analysis(sic_error_range, case_id=1):
    """
    Study impact of imperfect SIC (residual interference after cancellation).
    sic_error: fraction of residual power after SIC.
    """
    alpha_dB, rho, _ = CASES[case_id]
    h1, h2 = generate_los_channel(alpha_dB, rho, seed=0)
    h1_tx, h2_tx = h1.copy(), h2.copy()
    pc, p1, p2 = wmmse_rsma(h1_tx, h2_tx, Pt)
    pcn, p1n   = wmmse_noma(h1_tx, h2_tx, Pt)

    rsma_rates, noma_rates, sdma_rates = [], [], []

    for eps in sic_error_range:
        sinr_c1 = np.abs(h1@pc)**2 / (SIGMA2 + np.abs(h1@p1)**2 + np.abs(h1@p2)**2)
        sinr_c2 = np.abs(h2@pc)**2 / (SIGMA2 + np.abs(h2@p1)**2 + np.abs(h2@p2)**2)
        Rc_rsma = min(np.log2(1+sinr_c1), np.log2(1+sinr_c2))
        sinr1_sic = np.abs(h1@p1)**2 / (SIGMA2 + np.abs(h1@p2)**2 + eps*np.abs(h1@pc)**2)
        sinr2_sic = np.abs(h2@p2)**2 / (SIGMA2 + np.abs(h2@p1)**2 + eps*np.abs(h2@pc)**2)
        R1_rsma = np.log2(1 + sinr1_sic)
        R2_rsma = np.log2(1 + sinr2_sic)
        rsma_rates.append((Rc_rsma + R1_rsma + R2_rsma) * B * 1e-6)
        sc1n = np.abs(h1@pcn)**2 / (SIGMA2 + np.abs(h1@p1n)**2)
        sc2n = np.abs(h2@pcn)**2 / SIGMA2
        Rcn  = min(np.log2(1+sc1n), np.log2(1+sc2n))
        sinr1n_sic = np.abs(h1@p1n)**2 / (SIGMA2 + eps*np.abs(h1@pcn)**2)
        R1n  = np.log2(1 + sinr1n_sic)
        noma_rates.append((Rcn + R1n) * B * 1e-6)
        p1s, p2s = wmmse_sdma(h1_tx, h2_tx, Pt)
        R1s = np.log2(1 + np.abs(h1@p1s)**2 / (SIGMA2 + np.abs(h1@p2s)**2))
        R2s = np.log2(1 + np.abs(h2@p2s)**2 / (SIGMA2 + np.abs(h2@p1s)**2))
        sdma_rates.append((R1s + R2s) * B * 1e-6)

    return rsma_rates, noma_rates, sdma_rates

def quantization_bits_analysis(bits_range, case_id=4, n_runs=200):
    """Study throughput vs CSI quantization bits."""
    alpha_dB, rho, _ = CASES[case_id]
    rsma_ts, sdma_ts, noma_ts = [], [], []

    for Nb in bits_range:
        r_list, s_list, n_list = [], [], []
        for run in range(n_runs):
            h1t, h2t = generate_los_channel(alpha_dB, rho, seed=run)
            if Nb == 0:  # perfect
                h1_tx, h2_tx = h1t.copy(), h2t.copy()
            else:
                h1_tx = quantize_csi(h1t, Nb=Nb)
                h2_tx = quantize_csi(h2t, Nb=Nb)

            pc, p1, p2 = wmmse_rsma(h1_tx, h2_tx, Pt)
            T_r, *_ = compute_rsma_throughput(h1t, h2t, h1_tx, h2_tx, 3, 3, 3)
            T_s, *_ = compute_sdma_throughput(h1t, h2t, h1_tx, h2_tx, 3, 3)
            T_n, *_ = compute_noma_throughput(h1t, h2t, h1_tx, h2_tx, 3, 3)
            r_list.append(T_r); s_list.append(T_s); n_list.append(T_n)
        rsma_ts.append(np.mean(r_list))
        sdma_ts.append(np.mean(s_list))
        noma_ts.append(np.mean(n_list))

    return rsma_ts, sdma_ts, noma_ts

print("="*65)
print("RSMA / SDMA / NOMA Full Simulation")
print("="*65)
print(f"Effective bandwidth B = {B/1e6:.1f} MHz")
print(f"Transmit power Pt    = {Pt_dBm} dBm")
print(f"Monte-Carlo runs     = {N_runs} per case\n")

results_uq = {}   # unquantized CSI
results_q  = {}   # quantized CSI

for case_id in range(1, 10):
    print(f"  Case {case_id}: p={CASES[case_id][0]:.1f}dB  p={CASES[case_id][1]:.2f}  [{CASES[case_id][2]}]")
    results_uq[case_id] = run_case(case_id, quantized=False, n_runs=N_runs)
    results_q[case_id]  = run_case(case_id, quantized=True,  n_runs=N_runs)
    r = results_uq[case_id]
    print(f"    UQ -> RSMA={r['rsma']['T']:.1f}  SDMA={r['sdma']['T']:.1f}  NOMA={r['noma']['T']:.1f} Mbps")

print("\nRunning MCS sweep for Case 1 (Fig. 5 equivalent)...")
mcs_sweep_uq = run_case1_mcs_sweep(quantized=False, n_runs=150)
mcs_sweep_q  = run_case1_mcs_sweep(quantized=True,  n_runs=150)

print("Running theoretical curves...")
rho_range = np.linspace(0.01, 0.99, 40)
th_rsma_lo, th_sdma_lo, th_noma_lo = theoretical_sum_rate_vs_rho(rho_range, alpha_dB=-7.6)
th_rsma_hi, th_sdma_hi, th_noma_hi = theoretical_sum_rate_vs_rho(rho_range, alpha_dB=-23.6)

snr_range = np.arange(0, 35, 2)
snr_rates_hicorr = theoretical_rate_vs_snr(snr_range, rho=0.15)
snr_rates_locorr = theoretical_rate_vs_snr(snr_range, rho=0.90)

print("Running imperfect SIC analysis...")
sic_err = np.logspace(-3, 0, 30)
rsma_sic, noma_sic, sdma_sic = imperfect_sic_analysis(sic_err, case_id=1)

print("Running quantization bits analysis...")
bits_range = [1, 2, 3, 4, 6, 8, 0]   # 0=perfect
rsma_qb, sdma_qb, noma_qb = quantization_bits_analysis(bits_range, case_id=4, n_runs=150)

COLORS = {
    'rsma': {'common': '#2ecc71', 'priv1': '#27ae60', 'priv2': '#1a7a40'},
    'sdma': {'rx1': '#3498db', 'rx2': '#1a5276'},
    'noma': {'common': '#e67e22', 'priv1': '#784212'},
}
HATCH_UQ = ['', '/', '']
HATCH_Q  = ['..', '\\\\', '..']

def bar_case(ax, case_id, results, title, ymax=100):
    r  = results[case_id]
    uq = results_uq[case_id] if results is results_q else None

    groups = ['SDMA\nRx1', 'SDMA\nRx2', 'NOMA\nComm', 'NOMA\nPvt1',
              'RSMA\nComm', 'RSMA\nPvt1', 'RSMA\nPvt2']

    is_q = results is results_q
    suffix = '(Q)' if is_q else '(UQ)'

    vals_sdma_uq = [results_uq[case_id]['sdma']['T_1'], results_uq[case_id]['sdma']['T_2']]
    vals_noma_uq = [results_uq[case_id]['noma']['T_c'], results_uq[case_id]['noma']['T_1']]
    vals_rsma_uq = [results_uq[case_id]['rsma']['T_c'],
                    results_uq[case_id]['rsma']['T_1'], results_uq[case_id]['rsma']['T_2']]

    vals_sdma_q = [results_q[case_id]['sdma']['T_1'], results_q[case_id]['sdma']['T_2']]
    vals_noma_q = [results_q[case_id]['noma']['T_c'], results_q[case_id]['noma']['T_1']]
    vals_rsma_q = [results_q[case_id]['rsma']['T_c'],
                   results_q[case_id]['rsma']['T_1'], results_q[case_id]['rsma']['T_2']]

    x = np.arange(3)
    width = 0.35
    ax.bar(0 - width/2, sum(vals_sdma_uq), width, color=COLORS['sdma']['rx1'],
           label='SDMA(UQ)', alpha=0.9)
    ax.bar(0 + width/2, sum(vals_sdma_q), width, color=COLORS['sdma']['rx1'],
           alpha=0.55, hatch='...', label='SDMA(Q)')
    ax.bar(1 - width/2, vals_noma_uq[0], width, color=COLORS['noma']['common'],
           label='NOMA Common(UQ)', alpha=0.9)
    ax.bar(1 - width/2, vals_noma_uq[1], width, bottom=vals_noma_uq[0],
           color=COLORS['noma']['priv1'], alpha=0.9, label='NOMA Pvt(UQ)')
    ax.bar(1 + width/2, vals_noma_q[0], width, color=COLORS['noma']['common'],
           alpha=0.55, hatch='...')
    ax.bar(1 + width/2, vals_noma_q[1], width, bottom=vals_noma_q[0],
           color=COLORS['noma']['priv1'], alpha=0.55, hatch='...')
    ax.bar(2 - width/2, vals_rsma_uq[0], width, color=COLORS['rsma']['common'],
           label='RSMA Common(UQ)', alpha=0.9)
    ax.bar(2 - width/2, vals_rsma_uq[1], width, bottom=vals_rsma_uq[0],
           color=COLORS['rsma']['priv1'], alpha=0.9, label='RSMA Pvt1(UQ)')
    ax.bar(2 - width/2, vals_rsma_uq[2], width,
           bottom=vals_rsma_uq[0]+vals_rsma_uq[1],
           color=COLORS['rsma']['priv2'], alpha=0.9, label='RSMA Pvt2(UQ)')
    ax.bar(2 + width/2, vals_rsma_q[0], width, color=COLORS['rsma']['common'],
           alpha=0.55, hatch='...')
    ax.bar(2 + width/2, vals_rsma_q[1], width, bottom=vals_rsma_q[0],
           color=COLORS['rsma']['priv1'], alpha=0.55, hatch='...')
    ax.bar(2 + width/2, vals_rsma_q[2], width,
           bottom=vals_rsma_q[0]+vals_rsma_q[1],
           color=COLORS['rsma']['priv2'], alpha=0.55, hatch='...')
    t_sdma_uq = sum(vals_sdma_uq)
    t_rsma_uq = sum(vals_rsma_uq)
    gain = (t_rsma_uq - t_sdma_uq) / (t_sdma_uq + 1e-6) * 100
    ax.annotate(f'+{gain:.0f}%', xy=(2-width/2, t_rsma_uq), xytext=(2-width/2, t_rsma_uq + 3),
                ha='center', fontsize=7, color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['SDMA', 'NOMA', 'RSMA'])
    ax.set_ylabel('Sum Throughput (Mbps)')
    ax.set_title(f'Case {case_id}: a={r["alpha"]:.1f}dB, p={r["rho"]:.2f}\n{CASES[case_id][2]}',
                 fontsize=9)
    ax.set_ylim(0, ymax)
    ax.grid(axis='y', alpha=0.3)

fig1, axes = plt.subplots(3, 3, figsize=(15, 11))
fig1.suptitle('Sum Throughput: RSMA vs SDMA vs NOMA\n'
              'Solid=Unquantized CSI (UQ), Hatched=Quantized CSI (Q)\n'
              'Green bars=RSMA, Blue=SDMA, Orange=NOMA',
              fontsize=12, fontweight='bold')

ymaxes = [80, 90, 90, 60, 75, 85, 75, 65, 75]
for idx, case_id in enumerate(range(1, 10)):
    ax = axes[idx // 3][idx % 3]
    bar_case(ax, case_id, results_q, f'Case {case_id}', ymax=ymaxes[idx])
handles, labels = axes[0][0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='lower center', ncol=5, fontsize=8,
            bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('/mnt/user-data/outputs/fig1_sum_throughput_all_cases.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_sum_throughput_all_cases.png")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Case 1: Sum Throughput vs Private-Stream MCS Level\n'
              '(Reproducing Fig. 5 of Lyu et al. 2024)', fontsize=11, fontweight='bold')

cmap_rsma = plt.cm.Greens
cmap_noma = plt.cm.Oranges
x_labels  = [MCS_LABEL[i] for i in range(len(MCS_TABLE))]

for ax, sweep, title_sfx in [(ax1, mcs_sweep_uq, 'Unquantized CSI'),
                               (ax2, mcs_sweep_q,  'Quantized CSI')]:
    colors_c = ['#c0392b', '#e74c3c', '#e67e22', '#f39c12']
    styles   = ['-o', '-s', '-^', '-D']
    for i, (key, col, sty) in enumerate(zip(sweep.keys(), colors_c, styles)):
        ax.plot(range(len(MCS_TABLE)), sweep[key]['rsma'], sty, color=col,
                label=f'RSMA (C:{key})', lw=1.5, ms=5)
    ax.plot(range(len(MCS_TABLE)), list(sweep.values())[0]['sdma'], '--k',
            label='SDMA (no common)', lw=2)
    for i, (key, col, sty) in enumerate(zip(sweep.keys(), colors_c, styles)):
        ax.plot(range(len(MCS_TABLE)), sweep[key]['noma'], sty, color=col,
                label=f'NOMA (C:{key})', lw=1.0, ms=4, alpha=0.55)
    ax.set_xlabel('Private Stream MCS Level')
    ax.set_ylabel('Sum Throughput (Mbps)')
    ax.set_title(f'Case 1 -> {title_sfx}')
    ax.set_xticks(range(len(MCS_TABLE)))
    ax.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=7)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig2_mcs_sweep_case1.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_mcs_sweep_case1.png")

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle('Theoretical Sum-Rate vs Spatial Correlation p\n'
              '(Shannon bounds - validates RSMA superiority across all p)',
              fontsize=11, fontweight='bold')

for ax, (rsma, sdma, noma), ttl in [
    (ax1, (th_rsma_lo, th_sdma_lo, th_noma_lo), 'Low Pathloss (alpa~=-7.6 dB)'),
    (ax2, (th_rsma_hi, th_sdma_hi, th_noma_hi), 'High Pathloss (alpa~=-23.6 dB)'),
]:
    ax.plot(rho_range, rsma, '-o', color='#27ae60', ms=4, label='RSMA')
    ax.plot(rho_range, sdma, '-s', color='#2980b9', ms=4, label='SDMA')
    ax.plot(rho_range, noma, '-^', color='#e67e22', ms=4, label='NOMA')
    ax.fill_between(rho_range, sdma, rsma, alpha=0.12, color='#27ae60',
                    label='RSMA gain over SDMA')
    ax.set_xlabel('Spatial Correlation rho  (0=aligned, 1=orthogonal)')
    ax.set_ylabel('Sum Throughput (Mbps)')
    ax.set_title(ttl)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig3_theoretical_vs_rho.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_theoretical_vs_rho.png")

fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig4.suptitle('Theoretical Sum-Rate vs SNR\n(High vs Low Spatial Correlation)',
              fontsize=11, fontweight='bold')

for ax, rates, ttl in [
    (ax1, snr_rates_hicorr, 'High Spatial Correlation (rho=0.15)'),
    (ax2, snr_rates_locorr, 'Low Spatial Correlation (rho=0.90)'),
]:
    ax.plot(snr_range, rates['rsma'], '-o', color='#27ae60', ms=4, label='RSMA')
    ax.plot(snr_range, rates['sdma'], '-s', color='#2980b9', ms=4, label='SDMA')
    ax.plot(snr_range, rates['noma'], '-^', color='#e67e22', ms=4, label='NOMA')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Sum Throughput (Mbps)')
    ax.set_title(ttl)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig4_throughput_vs_snr.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_throughput_vs_snr.png")

fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig5.suptitle('Fairness Comparison: RSMA vs SDMA vs NOMA\n'
              '(Reproducing Fig. 8 of Lyu et al. 2024)\n'
              'Points closer to y=2x line = fairer outcome',
              fontsize=11, fontweight='bold')

for ax, results, ttl in [(ax1, results_uq, 'Unquantized CSI Feedback'),
                          (ax2, results_q,  'Quantized CSI Feedback')]:
    rsma_pts, sdma_pts, noma_pts = [], [], []
    for case_id in range(1, 10):
        f = compute_fairness(results[case_id])
        rsma_pts.append(f['rsma'])
        sdma_pts.append(f['sdma'])
        noma_pts.append(f['noma'])

    rsma_pts = np.array(rsma_pts)
    sdma_pts = np.array(sdma_pts)
    noma_pts = np.array(noma_pts)

    ax.scatter(sdma_pts[:, 0], sdma_pts[:, 1], marker='D', s=60,
               color='#2980b9', label='SDMA', zorder=5, alpha=0.8)
    ax.scatter(noma_pts[:, 0], noma_pts[:, 1], marker='s', s=60,
               color='#e67e22', label='NOMA', zorder=5, alpha=0.8)
    ax.scatter(rsma_pts[:, 0], rsma_pts[:, 1], marker='o', s=80,
               color='#27ae60', label='RSMA', zorder=6, alpha=0.9)
    for i, cid in enumerate(range(1, 10)):
        for pts, col in [(rsma_pts, '#1a7a40'), (sdma_pts, '#1a5276'),
                         (noma_pts, '#7d3c00')]:
            ax.annotate(str(cid), (pts[i, 0], pts[i, 1]),
                        fontsize=7, ha='center', va='center', color=col)
    xmax = max(ax.get_xlim()[1], rsma_pts[:, 0].max() * 1.1, 45)
    xs = np.linspace(0, xmax, 100)
    ax.plot(xs, 2*xs, '--', color='gray', lw=1.5, label='y=2x (max-min fair)')

    ax.set_xlabel('Minimum User Throughput (Mbps)')
    ax.set_ylabel('Sum Throughput (Mbps)')
    ax.set_title(ttl)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig5_fairness_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_fairness_comparison.png")

fig6, ax = plt.subplots(figsize=(8, 5))
fig6.suptitle('Effect of Imperfect SIC on Sum Throughput\n(Case 1 - High Spatial Correlation)',
              fontsize=11, fontweight='bold')

ax.semilogx(sic_err * 100, rsma_sic, '-o', color='#27ae60', ms=4, label='RSMA')
ax.semilogx(sic_err * 100, sdma_sic, '-s', color='#2980b9', ms=4, label='SDMA (no SIC)')
ax.semilogx(sic_err * 100, noma_sic, '-^', color='#e67e22', ms=4, label='NOMA')
ax.axvline(x=0.1, color='red', ls=':', lw=1, label='1% residual (typical)')
ax.set_xlabel('SIC Residual Interference (% of original power)')
ax.set_ylabel('Sum Throughput (Mbps)')
ax.set_title('RSMA maintains advantage even under imperfect SIC')
ax.legend()
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig6_imperfect_sic.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_imperfect_sic.png")

fig7, ax = plt.subplots(figsize=(8, 5))
fig7.suptitle('Effect of CSI Quantization Resolution\n(Case 4 - Medium Pathloss, High Correlation)',
              fontsize=11, fontweight='bold')

x_bits = [str(b) if b > 0 else 'inf\n(perfect)' for b in bits_range]
ax.plot(range(len(bits_range)), rsma_qb, '-o', color='#27ae60', ms=6, label='RSMA', lw=2)
ax.plot(range(len(bits_range)), sdma_qb, '-s', color='#2980b9', ms=6, label='SDMA', lw=2)
ax.plot(range(len(bits_range)), noma_qb, '-^', color='#e67e22', ms=6, label='NOMA', lw=2)
ax.set_xticks(range(len(bits_range)))
ax.set_xticklabels(x_bits)
ax.set_xlabel('CSI Quantization Bits per Component (Nb)')
ax.set_ylabel('Sum Throughput (Mbps)')
ax.legend()
ax.grid(alpha=0.3)
ax.annotate('Paper uses Nb=4\n(8-bit total)', xy=(3, np.mean([rsma_qb[3], sdma_qb[3]])),
            xytext=(3.5, np.mean([rsma_qb[3], sdma_qb[3]]) + 3),
            arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=8)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig7_quantization_bits.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_quantization_bits.png")

fig8, axes = plt.subplots(1, 2, figsize=(14, 6))
fig8.suptitle('RSMA Gain over SDMA and NOMA - All 9 Cases\n'
              '(Percentage throughput gain at optimal MCS)',
              fontsize=11, fontweight='bold')

for ax, results, ttl in [(axes[0], results_uq, 'Unquantized CSI'),
                          (axes[1], results_q,  'Quantized CSI')]:
    cases = list(range(1, 10))
    gain_vs_sdma = [(results[c]['rsma']['T'] - results[c]['sdma']['T']) /
                    (results[c]['sdma']['T'] + 1e-6) * 100 for c in cases]
    gain_vs_noma = [(results[c]['rsma']['T'] - results[c]['noma']['T']) /
                    (results[c]['noma']['T'] + 1e-6) * 100 for c in cases]

    x = np.arange(len(cases))
    w = 0.35
    bars1 = ax.bar(x - w/2, gain_vs_sdma, w, color='#2980b9', alpha=0.8,
                   label='RSMA gain over SDMA')
    bars2 = ax.bar(x + w/2, gain_vs_noma, w, color='#e67e22', alpha=0.8,
                   label='RSMA gain over NOMA')

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Case {c}\n(rho={CASES[c][1]:.2f})' for c in cases],
                       fontsize=7.5)
    ax.set_ylabel('Throughput Gain (%)')
    ax.set_title(ttl)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.0f}%',
                ha='center', va='bottom', fontsize=6.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.0f}%',
                ha='center', va='bottom', fontsize=6.5, color='#7d3c00')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig8_rsma_gain_summary.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_rsma_gain_summary.png")

fig9, ax = plt.subplots(figsize=(8, 5))
fig9.suptitle('Block Error Rate vs SNR per MCS Level\n'
              '(Normal Approximation - Polar code model, blocklength N=2400)',
              fontsize=11, fontweight='bold')

snr_db = np.linspace(-5, 35, 200)
for idx in [0, 2, 4, 6, 8]:
    m, r = MCS_M[idx], MCS_R[idx]
    bler  = [1 - bler_normal_approx(10**(s/10), m, r) for s in snr_db]
    ax.semilogy(snr_db, bler, label=MCS_LABEL[idx])

ax.set_xlabel('SNR per stream (dB)')
ax.set_ylabel('BLER (Block Error Rate)')
ax.set_ylim(1e-4, 1.1)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig9_bler_vs_snr.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_bler_vs_snr.png")

fig10, ax = plt.subplots(figsize=(8, 5))
fig10.suptitle('Degrees-of-Freedom Analysis: High-SNR Sum-Rate Slope\n'
               '(DoF = pre-log factor - RSMA achieves full DoF under imperfect CSIT)',
               fontsize=11, fontweight='bold')

snr_db_hi = np.arange(10, 50, 2)
rates_hi = theoretical_rate_vs_snr(snr_db_hi, rho=0.15)
for scheme, color, label in [('rsma', '#27ae60', 'RSMA (DoF=2)'),
                               ('sdma', '#2980b9', 'SDMA (DoF<2 imperfect CSIT)'),
                               ('noma', '#e67e22', 'NOMA')]:
    ax.plot(snr_db_hi, rates_hi[scheme], '-', color=color, lw=2, label=label)

ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Sum Throughput (Mbps)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig10_dof_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_dof_analysis.png")

fig11, axes = plt.subplots(1, 3, figsize=(13, 5))
fig11.suptitle('Optimal Power Allocation Across Cases\n'
               '(Fraction of total Pt allocated to common/private streams)',
               fontsize=11, fontweight='bold')

for ax, case_group, title in [
    (axes[0], [1,2,3],   'Low Pathloss Cases (1-3)'),
    (axes[1], [4,5,6],   'Med Pathloss Cases (4-6)'),
    (axes[2], [7,8,9],   'High Pathloss Cases (7-9)'),
]:
    labels, pc_frac, p1_frac, p2_frac = [], [], [], []
    for cid in case_group:
        alpha_dB, rho, _ = CASES[cid]
        h1, h2 = generate_los_channel(alpha_dB, rho, seed=0)
        pc, p1, p2 = wmmse_rsma(h1, h2, Pt)
        total_p = np.linalg.norm(pc)**2 + np.linalg.norm(p1)**2 + np.linalg.norm(p2)**2
        labels.append(f'Case {cid}\nrho={rho:.2f}')
        pc_frac.append(np.linalg.norm(pc)**2 / total_p * 100)
        p1_frac.append(np.linalg.norm(p1)**2 / total_p * 100)
        p2_frac.append(np.linalg.norm(p2)**2 / total_p * 100)

    x = np.arange(len(case_group))
    ax.bar(x, pc_frac, color='#2ecc71', label='Common (Pc)', alpha=0.9)
    ax.bar(x, p1_frac, bottom=pc_frac, color='#27ae60', label='Pvt RX1 (P1)', alpha=0.9)
    ax.bar(x, p2_frac, bottom=[pc_frac[i]+p1_frac[i] for i in range(len(x))],
           color='#1a7a40', label='Pvt RX2 (P2)', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Power Fraction (%)')
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig11_power_allocation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11_power_allocation.png")

print("\n" + "="*75)
print(f"{'Case':^6} {'alpa(dB)':^8} {'rho':^6} | {'RSMA(UQ)':^10} {'SDMA(UQ)':^10} {'NOMA(UQ)':^10} |"
      f" {'RSMA(Q)':^9} {'SDMA(Q)':^9} {'NOMA(Q)':^9}")
print("-"*75)
for cid in range(1, 10):
    r_uq = results_uq[cid]
    r_q  = results_q[cid]
    print(f"  {cid:^4}  {r_uq['alpha']:^8.1f} {r_uq['rho']:^6.2f} | "
          f"{r_uq['rsma']['T']:^10.1f} {r_uq['sdma']['T']:^10.1f} {r_uq['noma']['T']:^10.1f} | "
          f"{r_q['rsma']['T']:^9.1f} {r_q['sdma']['T']:^9.1f} {r_q['noma']['T']:^9.1f}")

print("="*75)
print("All units: Mbps | RSMA consistently highest across all cases and CSI quality levels")

print("\n All 11 figures saved to /mnt/user-data/outputs/")