"""
Periodic SSH chain utilities
Used in periodic_chain.ipynb

Contains:
- SSH band structure
- correlation functions
- eta calculation
- concurrence and Bell observables
- critical line utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- project paths ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

FIG_ROOT = PROJECT_ROOT / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

FIG_DIR = FIG_ROOT / "periodic"
FIG_DIR.mkdir(parents=True, exist_ok=True)


KS = np.linspace(0, 2*np.pi, 8192, endpoint=False) # k-grid for continuum integrals (periodic grid; endpoint=False to exclude 2pi)
ETA_C = np.sqrt(2) - 1  # concurrence threshold: C=0 <-> eta = sqrt(2)-1
# ============================================================
# Basic SSH utilities
# ============================================================

def ssh_couplings(lam: float) -> tuple[float, float]:
    """t1 = 1 - lam,  t2 = 1 + lam"""
    return 1.0 - lam, 1.0 + lam

def ssh_dispersion_from_t1t2(k: np.ndarray, t1, t2) -> np.ndarray:
    """|h(k)| = sqrt(t1^2 + t2^2 + 2 t1 t2 cos k)"""
    return np.sqrt(t1**2 + t2**2 + 2.0*t1*t2*np.cos(k))

def ssh_energy_pm(k: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """E_-(k), E_+(k) using lam parametrization."""
    t1, t2 = ssh_couplings(lam)
    Ep = ssh_dispersion_from_t1t2(k, t1, t2)
    return -Ep, Ep

def fermi_dirac(E: np.ndarray, beta: float) -> np.ndarray:
    """Fermi–Dirac distribution: n(E) = 1/(exp(beta E)+1)."""
    return 1.0 / (np.exp(beta * E) + 1.0)

def k_grid_finiteN(N: int, bz: str = "0_2pi") -> np.ndarray:
    """
    Discrete k grid for a periodic chain of length N:
      k_n = 2π n / N , n=0,...,N-1

    bz="0_2pi" -> [0,2π)
    bz="pm_pi" -> [-π,π)
    """
    if bz == "0_2pi":
        return np.linspace(0.0, 2.0*np.pi, N, endpoint=False)
    elif bz == "pm_pi":
        return np.linspace(-np.pi, np.pi, N, endpoint=False)
    else:
        raise ValueError("bz must be '0_2pi' or 'pm_pi'")
    
def h_complex_from_lam(lam: float, k: np.ndarray) -> np.ndarray:
    """
    Bloch off-diagonal term: h(k) = t1 + t2 e^{-ik}.
    Uses our convention t1=1-lam, t2=1+lam (via ssh_couplings).
    """
    t1, t2 = ssh_couplings(lam)
    return t1 + t2 * np.exp(-1j * k)

def abs_h_from_lam(lam: float, k: np.ndarray) -> np.ndarray:
    return np.abs(h_complex_from_lam(lam, k))

def phi_k_from_lam(lam: float, k: np.ndarray) -> np.ndarray:
    # phase in (-0, 2pi]
    return np.angle(h_complex_from_lam(lam, k))

def eta_continuum(lam: float, beta=np.inf, r: int = 0, ks: np.ndarray = KS) -> float:
    """
    Continuum correlator amplitude:
      η_r(λ,β) = (1/2π) ∫ dk  tanh(β|h(k)|/2) cos(rk + φ(k))
        where r=m-n is the site separation, and φ(k) = arg(h(k)).
    beta = np.inf -> tanh(...) -> 1 (ground-state limit)
    """
    ph = phi_k_from_lam(lam, ks)
    if np.isinf(beta):
        w = 1.0
        return np.cos(r*ks + ph).mean()
    else:
        w = np.tanh(0.5 * beta * abs_h_from_lam(lam, ks))
        return (w * np.cos(r*ks + ph)).mean()

def eta_finiteN(lam: float, N: int, beta=np.inf, r: int = 0, bz: str = "0_2pi") -> float:
    """
    Finite-N correlator amplitude:
      η_r(λ,β;N) = (1/N) Σ_k  tanh(β|h(k)|/2) cos(rk + φ(k))

    beta=np.inf -> tanh -> 1 (T=0 limit)
    """
    ks = k_grid_finiteN(N, bz=bz)
    ph = phi_k_from_lam(lam, ks)

    if np.isinf(beta):
        return np.cos(r*ks + ph).mean()
    else:
        w = np.tanh(0.5 * beta * abs_h_from_lam(lam, ks))
        return (w * np.cos(r*ks + ph)).mean()
    
def concurrence_from_eta(eta: float) -> float:
    """
    For the X-state structure used in the report:
      C = max(0, 1/2 (eta^2 + 2|eta| - 1))
    This matches the Kim&Cho active branch at T=0 (eta>=0),
    and stays safe if eta changes sign.
    """
    return max(0.0, 0.5 * (eta**2 + 2.0*np.abs(eta) - 1.0))

def _crossing_at_boundary(x: np.ndarray, y: np.ndarray, j: int, eps: float):
    """
    Linear interpolation for crossing y=eps between indices j-1 and j.
    Assumes y[j-1] and y[j] are on different sides of eps.
    """
    x0, x1 = x[j-1], x[j]
    y0, y1 = y[j-1], y[j]
    if np.isclose(y1, y0):
        return x1
    return x0 + (eps - y0) * (x1 - x0) / (y1 - y0)

def nonzero_region_boundaries(
    x: np.ndarray, y: np.ndarray, eps: float = 1e-10
):
    """
    Finds the left and right boundary of the region where y > eps.
    Returns (x_left, x_right). If y never exceeds eps -> (None, None).

    Works for:
      - "turn-on" (0 -> >0) on the right side
      - "turn-off" (>0 -> 0) on the right side
      - any single contiguous nonzero region (which is our case)
    """
    mask = y > eps
    if not np.any(mask):
        return None, None

    idx = np.where(mask)[0]
    i_left = idx[0]
    i_right = idx[-1]

    # left boundary
    if i_left == 0:
        x_left = x[0]
    else:
        x_left = _crossing_at_boundary(x, y, i_left, eps)

    # right boundary
    if i_right == len(x) - 1:
        x_right = x[-1]
    else:
        x_right = _crossing_at_boundary(x, y, i_right + 1, eps)

    return x_left, x_right

def lambda_critical_for_beta(beta, lams, r, ks=KS, eps=1e-10):
    """
    For fixed beta, scan lambda grid and return the critical lambda boundary.
    r=0 -> C1: take RIGHT boundary (where C1 -> 0)
    r=1 -> C2: take LEFT boundary  (where C2 turns on)
    """
    eta = np.array([eta_continuum(l, beta=beta, r=r, ks=ks) for l in lams])
    C   = np.array([concurrence_from_eta(x) for x in eta])
    left, right = nonzero_region_boundaries(lams, C, eps=eps)
    return right if r == 0 else left

def bell_from_eta(eta: float) -> float:
    """<B> = 2*sqrt(2)*|eta|  (CHSH Bell operator expectation in your notation)"""
    return 2.0 * np.sqrt(2.0) * np.abs(eta)

def bell_continuum(lam: float, beta=np.inf, r: int = 0, ks: np.ndarray = KS) -> float:
    """Compute <B>(lambda,beta) using eta_continuum."""
    eta = eta_continuum(lam, beta=beta, r=r, ks=ks)
    return bell_from_eta(eta)

def lambda_critical_bell_for_beta(beta, lams, r, ks=KS, eps=1e-10):
    """
    Return critical lambda where <B>(lambda,beta,r)=2 (CHSH threshold).
    We detect violation region via y = <B>-2 > eps and use your existing boundary finder.

    Convention:
      r=0 -> take RIGHT boundary
      r=1 -> take LEFT boundary
    """
    B = np.array([bell_continuum(l, beta=beta, r=r, ks=ks) for l in lams])
    y = B - 2.0
    left, right = nonzero_region_boundaries(lams, y, eps=eps)
    return right if r == 0 else left

# ============================================================
# SSH: plotting functions
# ============================================================

def make_figure_ssh_bands(
    lams_line=(-0.8, -0.4, 0.0, 0.4, 0.8),
    lam_range=(-1.0, 1.0),
    nk=200,
    nlam=200,
    cmap="inferno",
    savepath=None,
    show=True,
):
    if savepath is None:
        savepath = FIG_DIR / "fig_01_ssh_bands.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    k = np.linspace(0.0, 2.0*np.pi, nk)
    lam_grid = np.linspace(lam_range[0], lam_range[1], nlam)
    K, L = np.meshgrid(k, lam_grid)

    t1 = 1.0 - L
    t2 = 1.0 + L
    E2D = ssh_dispersion_from_t1t2(K, t1, t2)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

    # Panel A
    ax1 = axes[0]
    for lam in lams_line:
        Em, Ep = ssh_energy_pm(k, lam)
        (line,) = ax1.plot(k, Ep, label=rf"$\lambda={lam}$")
        ax1.plot(k, Em, color=line.get_color())

    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel("Energy")
    ax1.set_title(r"SSH band structure for different $\lambda$")
    ax1.legend()

    xticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    # Panel B
    ax2 = axes[1]
    pcm = ax2.pcolormesh(K, L, E2D, shading="auto", cmap=cmap)
    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$\lambda$")
    ax2.set_title(r"Upper SSH energy band $E_+(k,\lambda)$")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    fig.colorbar(pcm, ax=ax2, label="Energy")

    print("Saving to:", savepath)
    fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes

def make_figure_upper_occupation_overlay(
    lam: float = 0.0,
    betas=(30, 5, 2, 1, 0.5),
    nk: int = 400,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Single-panel overlay of n_+(k) for multiple betas at fixed lambda.
    Uses n_+(k) = 1/(exp(beta E_+(k))+1), with E_+(k)=|h(k)|.
    """
    if savepath is None:
        savepath = FIG_DIR / f"fig_2_occ_overlay_lam{lam}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)


    k = np.linspace(0.0, 2.0*np.pi, nk, endpoint=False)

    t1, t2 = ssh_couplings(lam)
    E_plus = ssh_dispersion_from_t1t2(k, t1, t2)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.8))

    for beta in betas:
        n_plus = fermi_dirac(E_plus, beta)
        ax.plot(k, n_plus, label=rf"$\beta={beta}$")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$n_+(k)$")
    ax.set_title(rf"Upper band occupation $n_+(k)$ (fixed $\lambda={lam}$)")

    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.set_ylim(-0.02, 0.52)  # çünkü n_+ ∈ [0, 1/2] (mu=0 simetri)

    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax

def make_figure_upper_occupation_overlay_lams(
    beta: float = 2.0,
    lams=(0.0, 0.1, 0.4, 0.8),
    nk: int = 400,
    savepath: str | None = None,
    show: bool = True,
):
    if savepath is None:
        savepath = FIG_DIR / f"fig_3_occ_overlay_beta{beta}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)
    
    k = np.linspace(0.0, 2.0*np.pi, nk, endpoint=False)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.8))

    for lam in lams:
        t1, t2 = ssh_couplings(lam)
        E_plus = ssh_dispersion_from_t1t2(k, t1, t2)
        n_plus = fermi_dirac(E_plus, beta)
        ax.plot(k, n_plus, label=rf"$\lambda={lam}$")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$n_+(k)$")
    ax.set_title(rf"Upper band occupation $n_+(k)$ (fixed $\beta={beta}$)")

    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.set_ylim(-0.02, 0.52)

    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax

def make_figure_concurrence_vs_lambda(
    lam_min=-1.0,
    lam_max=1.0,
    num=2001,
    beta=np.inf,
    ks=KS,
    savepath=None,
    show=True,
    title=None,
    mark_lambda0=True,
):
    """
    Kim & Cho Fig.2 style plot of concurrence vs lambda, with:
      - C1(λ)  (r=0)
      - C2(λ)  (r=1)
      - dC2/dλ (right axis)
    beta=np.inf -> ground-state limit
    beta finite  -> thermal
    """

    if savepath is None:
        tag = "T0" if np.isinf(beta) else f"beta{beta:g}"
        savepath = FIG_DIR / f"fig_04_concurrence_{tag}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    lams = np.linspace(lam_min, lam_max, num)

    # compute eta and concurrence
    eta0 = np.array([eta_continuum(l, beta=beta, r=0, ks=ks) for l in lams])
    eta1 = np.array([eta_continuum(l, beta=beta, r=1, ks=ks) for l in lams])

    C1 = np.array([concurrence_from_eta(x) for x in eta0])
    C2 = np.array([concurrence_from_eta(x) for x in eta1])

    # derivative (clean + robust)
    dC2 = np.gradient(C2, lams)

    # ---- plot
    fig, axC = plt.subplots(figsize=(7.2, 4.6))
    axD = axC.twinx()

    ln1, = axC.plot(lams, C1, color="mediumorchid", lw=2, label=r"$C_1(\lambda)$")
    ln2, = axC.plot(lams, C2, color="seagreen",    lw=2, label=r"$C_2(\lambda)$")
    ln3, = axD.plot(lams, dC2, color="deepskyblue", lw=2, label=r"$\partial_\lambda C_2$")

    if mark_lambda0:
        axC.axvline(0.0, ls="--", lw=1)

    axC.set_xlabel(r"$\lambda$")
    axC.set_ylabel("Concurrence")
    axD.set_ylabel(r"$\partial_\lambda C_2$")

    tag = "T=0" if np.isinf(beta) else rf"$\beta={beta:g}$"
    axC.set_title(title or rf"Wootters concurrence ({tag})")

    # combined legend
    lines = [ln1, ln2, ln3]
    axC.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper left")

    plt.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()

    eps = 1e-10  

    c1_left, c1_right = nonzero_region_boundaries(lams, C1, eps=eps)
    c2_left, c2_right = nonzero_region_boundaries(lams, C2, eps=eps)

    # critical lambda values:
    lamc1 = c1_right
    lamc2 = c2_left
    print(f"[{tag}] lambda_c1 (C1 -> 0) ~ {lamc1}")
    print(f"[{tag}] lambda_c2 (C2 turns on) ~ {lamc2}")

    if lamc1 is not None:
        axC.axvline(lamc1, ls="--", color="gray", lw=1)
    if lamc2 is not None:
        axC.axvline(lamc2, ls="--", color="gray", lw=1)

    return fig, (axC, axD)

def make_figure_C1_overlay_betas(
    betas=(np.inf, 30, 10, 5, 2, 1),
    lam_min=-1.0,
    lam_max=1.0,
    num=800,
    ks=KS,
    eps=1e-10,
    savepath=None,
    show=True,
    title=None,
):
    """
    Overlay plot of C1(λ) for multiple beta values on a single axis.
    Includes printed critical lambda where C1 drops to ~0 (right boundary).
    """
    if savepath is None:
        savepath = FIG_DIR / "fig_05_C1_overlay_betas.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    lams = np.linspace(lam_min, lam_max, num)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for beta in betas:
        eta0 = np.array([eta_continuum(l, beta=beta, r=0, ks=ks) for l in lams])
        C1 = np.array([concurrence_from_eta(x) for x in eta0])

        # critical lambda (where C1 goes to ~0): use nonzero region right boundary
        c1_left, c1_right = nonzero_region_boundaries(lams, C1, eps=eps)
        tag = "∞" if np.isinf(beta) else f"{beta:g}"
        print(f"[C1 overlay] beta={tag:>3}  lambda_c1 (C1->0) ~ {c1_right}")

        ax.plot(lams, C1, lw=2, label=rf"$\beta={tag}$")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$C_1(\lambda)$")
    ax.set_xlim(lam_min, lam_max)

    ax.set_title(title or r"$C_1(\lambda)$ for different temperatures")
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax

def make_figure_concurrence_finiteN(
    lam_lo=-0.05,
    lam_hi=0.05,
    num=1201,
    Ns=(50, 51),
    beta=np.inf,
    r=1,
    bz="0_2pi",
    savepath=None,
    show=True,
    title=None,
):
    """
    Plot C_r(λ) for several finite N values.
    r=0 -> C1-like, r=1 -> C2-like (depending on your mapping).
    """
    if savepath is None:
        tag = "T0" if np.isinf(beta) else f"beta{beta:g}"
        savepath = FIG_DIR / f"fig_06_finiteN_Cr_r{r}_{tag}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    lams = np.linspace(lam_lo, lam_hi, num)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for N in Ns:
        eta = np.array([eta_finiteN(l, N=N, beta=beta, r=r, bz=bz) for l in lams])
        C = np.array([concurrence_from_eta(x) for x in eta])
        tagN = "T=0" if np.isinf(beta) else rf"$\beta={beta:g}$"
        ax.plot(lams, C, lw=2, label=rf"$N={N}$ ({tagN})")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(rf"$C_{r}(\lambda)$")
    ax.set_title(title or "Finite-$N$ concurrence")
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax

def make_figure_lambda_critical_vs_beta(
    betas=None,
    lam_min=-1,
    lam_max=1,
    nlam=600,
    eps=1e-10,
    ks=KS,
    savepath=None,
    show=True,
    print_values=True,
    save_csv_path=None,   
):
    if betas is None:
        betas = np.r_[np.geomspace(0.2, 30.0, 100), np.inf]

    lams = np.linspace(lam_min, lam_max, nlam)

    lamc1 = []
    lamc2 = []
    for beta in betas:
        lamc1.append(lambda_critical_for_beta(beta, lams, r=0, ks=ks, eps=eps))
        lamc2.append(lambda_critical_for_beta(beta, lams, r=1, ks=ks, eps=eps))

    lamc1 = np.array(lamc1, dtype=float)
    lamc2 = np.array(lamc2, dtype=float)

    # ---- PRINT TABLE (NEW)
    if print_values:
        print("\nCritical lambdas from concurrence boundary detection")
        print(f"eps = {eps:g}, ks = {ks}, lambda-grid = [{lam_min}, {lam_max}] with nlam={nlam}")
        print("--------------------------------------------------------------")
        print(f"{'beta':>10}   {'lambda_c1 (C1->0)':>18}   {'lambda_c2 (C2 on)':>18}")
        for b, c1, c2 in zip(betas, lamc1, lamc2):
            bstr = "inf" if np.isinf(b) else f"{b:10.6g}"
            c1str = "None" if np.isnan(c1) else f"{c1:18.10f}"
            c2str = "None" if np.isnan(c2) else f"{c2:18.10f}"
            print(f"{bstr:>10}   {c1str}   {c2str}")

    # ---- SAVE CSV 
    if save_csv_path is not None:
        beta_out = np.array(betas, dtype=float)
        header = "beta,lambda_c1,lambda_c2"
        data = np.column_stack([beta_out, lamc1, lamc2])
        np.savetxt(save_csv_path, data, delimiter=",", header=header, comments="")
        if print_values:
            print("\nSaved CSV to:", save_csv_path)

    # x-axis: finite betas for log plot
    finite = np.isfinite(betas)
    betas_f = np.array(betas)[finite]
    lamc1_f = lamc1[finite]
    lamc2_f = lamc2[finite]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(betas_f, lamc1_f, lw=2, label=r"$\lambda_{c1}(\beta)$  (C1$\to$0)")
    ax.plot(betas_f, lamc2_f, lw=2, label=r"$\lambda_{c2}(\beta)$  (C2 turns on)")


    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\lambda_c$")
    ax.set_title(r"Critical $\lambda_c(\beta)$ from concurrence")
    ax.legend(frameon=False)

    plt.tight_layout()

    if savepath is None:
        savepath = FIG_DIR / "fig_07_lambda_critical_vs_beta.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    print("Saving figure to:", savepath)

    fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, (betas, lamc1, lamc2)

def make_figure_bell_vs_lambda(
    lam_min=-1.0,
    lam_max=1.0,
    num=400,
    betas=np.inf,  
    r_values=(0, 1),
    ks=KS,
    savepath=None,
    show=True,
    title=None,
    mark_lambda0=True,
):
    """
    Plot <B>(lambda) for one or multiple beta values.

    Parameters
    ----------
    betas : float or list of float
        e.g.
        betas = np.inf
        betas = [0.5, 1, 2, 5, np.inf]

    r_values : tuple
        e.g. (0,), (1,), or (0,1)
    """

    
    if np.isscalar(betas):
        betas = [betas]

    if savepath is None:
        savepath = FIG_DIR / "fig_08_bell_vs_lambda_multi_beta.pdf"
    else:
        savepath = Path(savepath)

    savepath.parent.mkdir(parents=True, exist_ok=True)

    lams = np.linspace(lam_min, lam_max, num)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for beta in betas:

        beta_label = "∞" if np.isinf(beta) else f"{beta:g}"

        for r in r_values:

            B_vals = np.array([
                bell_continuum(l, beta=beta, r=r, ks=ks)
                for l in lams
            ])

            ax.plot(
                lams,
                B_vals,
                lw=2,
                label=rf"$\langle B\rangle$  ($r={r}$, $\beta={beta_label}$)"
            )

    # CHSH bound
    ax.axhline(2.0, ls="--", lw=1, color="black", label=r"CHSH bound (=2)")

    if mark_lambda0:
        ax.axvline(0.0, ls="--", lw=1)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle B\rangle$")
    ax.set_xlim(lam_min, lam_max)

    if title is None:
        title = r"Bell expectation $\langle B\rangle$ vs $\lambda$"

    ax.set_title(title)

    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

def make_figure_lambda_critical_bell_vs_beta(
    betas,
    lam_min=-1.0,
    lam_max=1.0,
    nlam=600,
    eps=1e-10,
    ks=KS,
    savepath=None,
    show=True,
    title=None,
    save_csv_path=None,   # <-- NEW
):
    """
    Plot lambda_critical(beta) extracted from <B>=2 threshold,
    for r=0 and r=1.

    If save_csv_path is provided, saves:
        beta, lambda_B_r0, lambda_B_r1
    """

    lams = np.linspace(lam_min, lam_max, nlam)

    lamB0 = []
    lamB1 = []

    for beta in betas:
        lamB0.append(lambda_critical_bell_for_beta(beta, lams, r=0, ks=ks, eps=eps))
        lamB1.append(lambda_critical_bell_for_beta(beta, lams, r=1, ks=ks, eps=eps))

    lamB0 = np.array(lamB0, dtype=float)
    lamB1 = np.array(lamB1, dtype=float)

    betas = np.array(betas, dtype=float)
    finite = np.isfinite(betas)

    # ---------------------------
    # SAVE CSV (NEW PART)
    # ---------------------------
    if save_csv_path is not None:
        data = np.column_stack([betas, lamB0, lamB1])
        header = "beta,lambda_B_r0,lambda_B_r1"
        np.savetxt(save_csv_path, data, delimiter=",",
                   header=header, comments="")
        print("Saved Bell critical data to:", save_csv_path)

    # ---------------------------
    # Plot
    # ---------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.plot(betas[finite], lamB0[finite],
            lw=2, label=r"$\lambda^{(B)}_{\mathrm{crit}}(\beta)$  ($r=0$)")
    ax.plot(betas[finite], lamB1[finite],
            lw=2, label=r"$\lambda^{(B)}_{\mathrm{crit}}(\beta)$  ($r=1$)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\lambda^{(B)}_{\mathrm{crit}}$")
    ax.set_title(title or r"Critical line from Bell threshold $\langle B\rangle=2$")
    ax.legend(frameon=False)

    plt.tight_layout()

    if savepath is None:
        savepath = FIG_DIR / "fig_09_lambda_critical_bell_vs_beta.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    print("Saving figure to:", savepath)

    fig.savefig(savepath, bbox_inches="tight")


    if show:
        plt.show()

    return fig, ax, (betas, lamB0, lamB1)

def plot_entanglement_phase_diagram_like_curves(
    csv_conc_path,
    csv_bell_path,
    lam_min=-1.0,
    lam_max=1.0,
    beta_plot_min=None,
    beta_plot_max=None,
    savepath=None,
    show=True,
    title=r"Entanglement Phase Diagram ($r=0$)",
):
    # --- load concurrence CSV
    conc = np.genfromtxt(csv_conc_path, delimiter=",", names=True)
    beta_c = conc["beta"].astype(float)
    lam_c1 = conc["lambda_c1"].astype(float)

    mc = np.isfinite(beta_c) & np.isfinite(lam_c1)
    beta_c, lam_c1 = beta_c[mc], lam_c1[mc]
    oc = np.argsort(beta_c)
    beta_c, lam_c1 = beta_c[oc], lam_c1[oc]

    # --- load bell CSV
    bell = np.genfromtxt(csv_bell_path, delimiter=",", names=True)
    beta_b = bell["beta"].astype(float)
    lam_B0 = bell["lambda_B_r0"].astype(float)

    mb = np.isfinite(beta_b) & np.isfinite(lam_B0)
    beta_b, lam_B0 = beta_b[mb], lam_B0[mb]
    ob = np.argsort(beta_b)
    beta_b, lam_B0 = beta_b[ob], lam_B0[ob]

    # --- master beta grid
    beta = beta_c.copy()
    lc1 = lam_c1.copy()

    # --- interpolate bell only on its own domain
    lB = np.full_like(beta, np.nan, dtype=float)
    in_range = (beta >= beta_b.min()) & (beta <= beta_b.max())
    lB[in_range] = np.interp(beta[in_range], beta_b, lam_B0)

    # --- plot limits
    if beta_plot_min is None:
        beta_plot_min = beta.min()
    if beta_plot_max is None:
        beta_plot_max = beta.max()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # ==================================================
    # Make the default phase GREY everywhere
    # ==================================================
    ax.set_facecolor("lightgrey")

    # --- Yellow region: entangled but local
    # Bell not defined yet  -> from lam_min up to lc1
    m_noB = np.isnan(lB)
    ax.fill_between(
        beta[m_noB],
        lam_min,
        lc1[m_noB],
        color="khaki",
        alpha=0.7,
        zorder=2,
    )

    # Bell defined -> between lB and lc1
    m_yesB = np.isfinite(lB)
    ax.fill_between(
        beta[m_yesB],
        lB[m_yesB],
        lc1[m_yesB],
        where=(lc1[m_yesB] >= lB[m_yesB]),
        interpolate=True,
        color="khaki",
        alpha=0.7,
        zorder=2,
    )

    # --- Blue region: Bell-violating
    ax.fill_between(
        beta[m_yesB],
        lam_min,
        lB[m_yesB],
        color="lightblue",
        alpha=0.6,
        zorder=2,
    )

    # --- Curves
    ax.plot(beta, lc1, color="red", lw=2,
            label=r"$\lambda_{c1}(\beta)$ (C1$\to0$)", zorder=3)
    ax.plot(beta[m_yesB], lB[m_yesB], color="blue", lw=2,
            label=r"$\lambda_{B}(\beta)$ ($\langle B\rangle\to2$)", zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(beta_plot_min, beta_plot_max)
    ax.set_ylim(lam_min, lam_max)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(title)
    ax.legend(frameon=False)

    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

#Asymptotic behavior of lambda critical:

def plot_lambda_critical_asymptotics(
    csv_C_path,
    csv_B_path=None,
    beta_min_plot=0.05,
    lam_min=-1,
    lam_max=1,
    show=True,
    savepath=None
):


    # --------------------
    # Load concurrence CSV
    # --------------------
    conc = np.genfromtxt(csv_C_path, delimiter=",", names=True)

    beta = conc["beta"]
    lam_c1 = conc["lambda_c1"]

    m = np.isfinite(beta) & np.isfinite(lam_c1)
    beta = beta[m]
    lam_c1 = lam_c1[m]

    order = np.argsort(beta)
    beta = beta[order]
    lam_c1 = lam_c1[order]

    # --------------------
    # Load Bell CSV
    # --------------------
    beta_b = None
    lam_b = None

    if csv_B_path is not None:

        bell = np.genfromtxt(csv_B_path, delimiter=",", names=True)

        if "lambda_B_r0" in bell.dtype.names:
            lam_b = bell["lambda_B_r0"]
        else:
            lam_b = bell["lambda_B"]

        beta_b = bell["beta"]

        m = np.isfinite(beta_b) & np.isfinite(lam_b)
        beta_b = beta_b[m]
        lam_b = lam_b[m]

    # --------------------
    # Asymptotic curve
    # --------------------
    beta_dense = np.geomspace(beta_min_plot, np.max(beta), 800)

    lam_asym = 1 - 2*(np.sqrt(2)-1)/beta_dense

    # --------------------
    # Plot
    # --------------------
    fig= plt.figure(figsize=(10,6))

    plt.plot(beta, lam_c1,
             color="red", lw=2,
             label=r"$\lambda_{c1}(\beta)$ (numerical, $C_1 \to 0$)")

    plt.plot(beta_dense, lam_asym,
             "--", color="black", lw=2,
             label=r"$\lambda^{\mathrm{asym}}_{c1}(\beta)=1-\frac{2(\sqrt{2}-1)}{\beta}$")

    if beta_b is not None:
        plt.plot(beta_b, lam_b,
                 color="blue", lw=2,
                 label=r"$\lambda_B(\beta)$ (numerical, Bell threshold)")

    # onset marker
    plt.axvline(ETA_C, ls=":", color="gray",
                label=rf"marker $\beta \approx {ETA_C:.3f}$")

    plt.xscale("log")

    plt.xlim(beta_min_plot, np.max(beta))
    plt.ylim(lam_min, lam_max)

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\lambda$")

    plt.title(r"Critical $\lambda(\beta)$ with closed-form low-$\beta$ asymptotic overlay")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    if savepath is None:
        savepath = FIG_DIR / "fig_11_lambda_critical_asymptotics.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    print("Saving figure to:", savepath)

    fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
    
def plot_lambda_critical_fit_asymptotics(
    csv_path,
    lambda_col="lambda_c1",
    observable_name="Concurrence",
    low_beta_max=0.8,
    low_deg=3,
    high_beta_min=8.0,
    tail_pts=8,
    y_tol=1e-8,
    r=1,                   # SSH bond choice: r=0 or r=1
    savepath=None,
):
    # ---------- LOAD DATA ----------
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    beta = data["beta"].astype(float)
    lam  = data[lambda_col].astype(float)

    m = np.isfinite(beta) & np.isfinite(lam)
    beta, lam = beta[m], lam[m]
    o = np.argsort(beta)
    beta, lam = beta[o], lam[o]

    # ---------- LOW-BETA polynomial fit ----------
    mlow = beta <= low_beta_max
    if np.sum(mlow) < low_deg + 1:
        raise ValueError(
            f"Not enough low-beta points for degree-{low_deg} polynomial fit. "
            f"Need at least {low_deg+1}, got {np.sum(mlow)}."
        )

    coeff = np.polyfit(beta[mlow], lam[mlow], low_deg)
    b_low = np.linspace(beta[mlow].min(), beta[mlow].max(), 250)
    lam_low = np.polyval(coeff, b_low)

    # ---------- HIGH-BETA exponential fit ----------
    mhi = beta >= high_beta_min
    if np.sum(mhi) < 3:
        raise ValueError(
            f"Not enough high-beta points for exponential fit. Got {np.sum(mhi)} points."
        )

    b_hi_data = beta[mhi]
    lam_hi_data = lam[mhi]

    if len(lam_hi_data) >= tail_pts:
        lam_inf = np.median(lam_hi_data[-tail_pts:])
    else:
        lam_inf = np.median(lam_hi_data)

    y = lam_hi_data - lam_inf
    keep = np.abs(y) > y_tol

    if np.sum(keep) >= 3:
        b_fit = b_hi_data[keep]
        y_fit = y[keep]

        logy = np.log(np.abs(y_fit))
        p = np.polyfit(b_fit, logy, 1)

        Delta = -p[0]
        A_abs = np.exp(p[1])
        sign = np.sign(y_fit[0])
        A = sign * A_abs

        b_high = np.linspace(b_hi_data.min(), b_hi_data.max(), 250)
        lam_high = lam_inf + A * np.exp(-Delta * b_high)
    else:
        b_high = np.linspace(b_hi_data.min(), b_hi_data.max(), 250)
        lam_high = np.full_like(b_high, lam_inf)
        A = 0.0
        Delta = np.inf


    # ---------- PLOT ----------
    fig = plt.figure(figsize=(7.2, 4.8))

    plt.plot(beta, lam, lw=2.4, label=rf"numeric $\lambda_c(\beta)$")
    plt.plot(
        b_low, lam_low, "--", lw=2,
        label=rf"low-$\beta$ polynomial fit (deg={low_deg})"
    )

    plt.plot(
        b_high, lam_high,
        linestyle=":", color="crimson", lw=3,
        label=rf"high-$\beta$: $\lambda_\infty + A e^{{-\Delta\beta}}$"
    )

    plt.xscale("log")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\lambda_c$")
    plt.title(rf"{observable_name}: $\lambda_c(\beta)$ with asymptotics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is None:
        savepath = FIG_DIR / "fig_12_lambda_critical_asymptotics.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    print("Saving figure to:", savepath)

    fig.savefig(savepath, bbox_inches="tight")

    plt.show()

    # ---------- PRINT SUMMARY ----------
    print(f"{observable_name} --- Low-beta fit summary:")
    print(f"  beta <= {low_beta_max}")
    print(f"  degree = {low_deg}")
    print("  lambda_c(beta) ≈ " + " + ".join(
        [f"({c:+.6g})*beta^{p}" for p, c in zip(range(low_deg, -1, -1), coeff)]
    ))

    a = coeff[::-1]
    print("  (a0,a1,a2,a3...) =", ", ".join([f"{x:.6g}" for x in a]))
    print(f"  points used: {mlow.sum()} / {len(beta)}")

    print(f"{observable_name} --- High-beta fit summary:")
    print(f"  beta >= {high_beta_min}")
    print(f"  lambda_inf ≈ {lam_inf:.10f}")
    print(f"  A ≈ {A:.3e}")
    print(f"  Delta ≈ {Delta if np.isfinite(Delta) else 'inf'}")
    print(f"  points used in log-fit: {int(np.sum(keep))} / {len(y)}")

def plot_lambda_bell_with_asymptotics(
    csv_path,
    lambda_col="lambda_B_r1",
    low_beta_max=0.8,
    low_deg=3,
    high_beta_min=8.0,
    tail_pts=8,
    y_tol=1e-8,
    out_pdf=None,
):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    beta = data["beta"].astype(float)
    lam  = data[lambda_col].astype(float)

    m = np.isfinite(beta) & np.isfinite(lam)
    beta, lam = beta[m], lam[m]

    if len(beta) == 0:
        raise ValueError(f"{lambda_col} için hiç finite veri yok.")

    o = np.argsort(beta)
    beta, lam = beta[o], lam[o]

    plt.figure(figsize=(7,4.5))
    plt.plot(beta, lam, lw=2, label=rf"numeric {lambda_col}")

    # ---------- LOW-BETA polynomial fit ----------
    mlow = beta <= low_beta_max
    if np.sum(mlow) >= low_deg + 1:
        coeff = np.polyfit(beta[mlow], lam[mlow], low_deg)
        b_low = np.linspace(beta[mlow].min(), beta[mlow].max(), 250)
        lam_low = np.polyval(coeff, b_low)
        plt.plot(b_low, lam_low, "--", lw=2, label=rf"low-$\beta$ poly (deg={low_deg})")

        print("Low-beta fit summary:")
        print(f"  beta <= {low_beta_max}")
        print(f"  degree = {low_deg}")
        print("  lambda_B(beta) ≈ " + " + ".join(
            [f"({c:+.6g})*beta^{p}" for p, c in zip(range(low_deg, -1, -1), coeff)]
        ))
        a = coeff[::-1]
        print("  (a0,a1,a2,a3...) =", ", ".join([f"{x:.6g}" for x in a]))
        print(f"  points used: {mlow.sum()} / {len(beta)}")
    else:
        print("Low-beta fit skipped:")
        print(f"  beta <= {low_beta_max} bölgesinde yeterli finite veri yok.")
        print(f"  points used: {mlow.sum()} / {len(beta)}")

    # ---------- HIGH-BETA exponential fit ----------
    mhi = beta >= high_beta_min
    if np.sum(mhi) >= 3:
        b_hi_data = beta[mhi]
        lam_hi_data = lam[mhi]

        if len(lam_hi_data) >= tail_pts:
            lam_inf = np.median(lam_hi_data[-tail_pts:])
        else:
            lam_inf = np.median(lam_hi_data)

        y = lam_hi_data - lam_inf
        keep = np.abs(y) > y_tol

        if np.sum(keep) >= 3:
            b_fit = b_hi_data[keep]
            y_fit = y[keep]

            logy = np.log(np.abs(y_fit))
            p = np.polyfit(b_fit, logy, 1)
            Delta = -p[0]
            A_abs = np.exp(p[1])
            sign = np.sign(y_fit[0])
            A = sign * A_abs

            b_high = np.linspace(b_hi_data.min(), b_hi_data.max(), 250)
            lam_high = lam_inf + A * np.exp(-Delta * b_high)

            plt.plot(
                b_high, lam_high,
                linestyle=":", color="crimson", lw=3,
                label=rf"high-$\beta$: $\lambda_\infty + A e^{{-\Delta\beta}}$"
            )

            print("High-beta fit summary:")
            print(f"  beta >= {high_beta_min}")
            print(f"  lambda_inf ≈ {lam_inf:.10f}")
            print(f"  A ≈ {A:.3e}")
            print(f"  Delta ≈ {Delta:.6g}")
            print(f"  points used in log-fit: {int(np.sum(keep))} / {len(y)}")
        else:
            print("High-beta fit skipped: y = lambda-lambda_inf çok küçük.")
    else:
        print("High-beta fit skipped:")
        print(f"  beta >= {high_beta_min} bölgesinde yeterli veri yok.")

    plt.xscale("log")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\lambda_B$")
    plt.title(rf"${lambda_col}(\beta)$ with asymptotics")
    plt.grid(True)
    plt.legend()

    if out_pdf is not None:
        plt.savefig(out_pdf, bbox_inches="tight")
        print(f"Figure saved to: {out_pdf}")

    plt.show()

