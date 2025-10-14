import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import toeplitz, eigh
from pathlib import Path
import json

# Candidate variable names inside the .mat file
PREF_VAR_NAMES = ("x", "xb", "signal", "data", "X", "y")

def find_matfile(here: Path, user_path: str | None) -> Path:
    """Auto-detect x.mat / xb.mat if not specified."""
    if user_path:
        p = (here / user_path) if not Path(user_path).is_absolute() else Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"--matfile specified but not found: {p}")
        return p
    for name in ["x.mat", "xb.mat", "X.mat"]:
        p = here / name
        if p.exists():
            return p
    raise FileNotFoundError("No MAT file found. Place x.mat or xb.mat next to this script, or pass --matfile.")

def load_vector_from_mat(path: Path) -> np.ndarray:
    """Read a 1D float vector from a .mat file, tolerant of shapes & cell arrays."""
    d = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    pick = None
    for k in PREF_VAR_NAMES:
        if k in d:
            pick = k
            break
    if pick is None:
        keys = [k for k in d.keys() if not k.startswith("__")]
        if not keys:
            raise RuntimeError(f"No numeric variables found in {path}")
        pick = keys[0]
    arr = d[pick]

    # flatten and clean
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr, dtype=object)
    if getattr(arr, "dtype", None) == np.object_:
        flat = [it for it in np.ravel(arr) if isinstance(it, np.ndarray)]
        if flat:
            arr = flat[0]
        else:
            arr = np.array(arr, dtype=float)
    x = np.asarray(arr, dtype=np.float64).squeeze()
    if x.ndim > 1:
        x = x.ravel()
    # mean-center
    x = x - np.nanmean(x)
    if not np.isfinite(x).all():
        x = x[np.isfinite(x)]
    x = np.ascontiguousarray(x)
    if x.size < 2:
        raise ValueError(f"Vector from {path} too short after cleaning (size={x.size})")
    return x

def sample_autocorr(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64).ravel()
    r = np.correlate(x, x, mode='full')
    mid = len(r) // 2
    return r, mid

def minvar_fir_from_Rx(x: np.ndarray, N: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    r, mid = sample_autocorr(x)
    r_nonneg = r[mid:mid+N]
    Rx = toeplitz(r_nonneg)
    vals, vecs = eigh(Rx)  # ascending
    h = vecs[:, 0]
    h = h / np.linalg.norm(h)
    y = np.convolve(x, h, mode='same')
    vy = float(np.var(y))

    np.save(outdir / f"q3_h_N{N}.npy", h)
    with open(outdir / f"q3_metrics_N{N}.json", "w") as f:
        json.dump({"N": int(N), "min_eigval": float(vals[0]), "var_y": vy}, f, indent=2)

    # Plot impulse response (no use_line_collection)
    plt.figure()
    markerline, stemlines, baseline = plt.stem(np.arange(N), h)
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.xlabel('n'); plt.ylabel('h[n]')
    plt.title(f'Q3: Minimum-Variance FIR (N={N})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outdir / f"q3_impulse_N{N}.png", dpi=160, bbox_inches='tight')
    plt.close()

    # Plot magnitude spectrum
    Lfft = 4096
    H = np.fft.rfft(h, n=Lfft)
    w = np.linspace(0, np.pi, H.shape[0])
    plt.figure()
    plt.plot(w, np.abs(H))
    plt.xlabel('ω (rad/sample)'); plt.ylabel('|H(e^{jω})|')
    plt.title(f'Q3: |H(e^{{jω}})| (N={N})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outdir / f"q3_spectrum_N{N}.png", dpi=160, bbox_inches='tight')
    plt.close()

    return vy, float(vals[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matfile", type=str, default=None, help="Path to x.mat / xb.mat")
    parser.add_argument("--N", nargs="+", type=int, default=[8, 12, 16, 24], help="FIR lengths to evaluate")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    outroot = here / "outputs" / "q3"
    outroot.mkdir(parents=True, exist_ok=True)

    mat_path = find_matfile(here, args.matfile)
    x = load_vector_from_mat(mat_path)
    print(f"[INFO] Using MAT file: {mat_path.name}")
    print(f"[INFO] x: dtype={x.dtype}, len={x.size}, mean≈{float(np.mean(x)):.3g}, std≈{float(np.std(x)):.3g}")

    # Autocorrelation plot
    r, mid = sample_autocorr(x)
    center = min(100, mid)
    lags = np.arange(-center, center+1)
    plt.figure()
    plt.plot(lags, r[mid-center:mid+center+1])
    plt.xlabel('Lag'); plt.ylabel('r_x[l]')
    plt.title('Q3: Sample Autocorrelation (central lags)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outroot / "q3_autocorr.png", dpi=160, bbox_inches='tight')
    plt.close()

    # Process different N
    results = []
    for N in args.N:
        vy, mineig = minvar_fir_from_Rx(x, N, outroot / f"N{N}")
        results.append((N, vy, mineig))

    # Variance vs N plot
    Ns = [t[0] for t in results]
    Vars = [t[1] for t in results]
    plt.figure()
    plt.plot(Ns, Vars, marker='o')
    plt.xlabel('FIR length N'); plt.ylabel('var(y)')
    plt.title('Q3: Output Variance vs FIR Length')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outroot / "q3_var_vs_N.png", dpi=160, bbox_inches='tight')
    plt.close()

    # Save summary
    with open(outroot / "q3_summary.json", "w") as f:
        json.dump({"results": [{"N": int(N), "var_y": float(v), "min_eigval": float(me)} for N, v, me in results]}, f, indent=2)

    print("=== Q3 Minimum-Variance FIR ===")
    for N, v, me in results:
        print(f"N={N:>2d} : var(y) = {v:.6g} , min_eig(Rx) = {me:.6g}")

if __name__ == "__main__":
    main()