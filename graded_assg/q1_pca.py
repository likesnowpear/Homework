import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import eigh
from pathlib import Path
import json

def center_columns(X):
    return X - np.mean(X, axis=0, keepdims=True)

def fro_error(A, B):
    return np.linalg.norm(A - B, 'fro')**2

def main():
    here = Path(__file__).resolve().parent
    outdir = here / "outputs" / "q1"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load X.mat ----
    mat_path = here / "X.mat"
    if not mat_path.exists():
        raise FileNotFoundError("X.mat not found. Put X.mat next to this script.")

    d = loadmat(str(mat_path))
    # try common keys; otherwise pick the first non-metadata key
    keys = [k for k in d.keys() if not k.startswith("__")]
    if not keys:
        raise RuntimeError("No numeric variables found in X.mat")
    if "X" in d:
        X = d["X"]
    else:
        X = d[keys[0]]
        print(f"[INFO] Using variable '{keys[0]}' from X.mat")

    # ---- PCA pipeline ----
    Xc = center_columns(X)               
    C = np.cov(Xc, rowvar=False)

    # eigen-decomposition (symmetric)
    vals, vecs = eigh(C)                 
    idx = np.argsort(vals)[::-1]         
    lam = vals[idx]
    V   = vecs[:, idx]

    errors = []
    varexp = []
    for r in range(1, V.shape[1]+1):
        Vr = V[:, :r]
        Xhat = Xc @ Vr @ Vr.T            
        errors.append(fro_error(Xc, Xhat))
        varexp.append(np.sum(lam[:r]) / np.sum(lam))

    # ---- Plots ----
    # (1) Reconstruction error vs r
    plt.figure()
    plt.plot(range(1, V.shape[1]+1), errors, marker='o')
    plt.xlabel('r (number of principal components)')
    plt.ylabel('Frobenius reconstruction error')
    plt.title('Q1: Reconstruction Error vs r')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outdir / "q1_error_vs_r.png", dpi=160, bbox_inches='tight')
    plt.close()

    # (2) Cumulative variance explained
    plt.figure()
    plt.plot(range(1, V.shape[1]+1), varexp, marker='o')
    plt.xlabel('r (number of principal components)')
    plt.ylabel('Cumulative variance explained')
    plt.title('Q1: Cumulative Variance Explained vs r')
    plt.ylim(0, 1.01)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outdir / "q1_cumvar_vs_r.png", dpi=160, bbox_inches='tight')
    plt.close()

    # (3) Scree plot of eigenvalues
    plt.figure()
    plt.plot(range(1, V.shape[1]+1), lam, marker='o')
    plt.xlabel('Component index')
    plt.ylabel('Eigenvalue')
    plt.title('Q1: Eigenvalue (Scree) Plot')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outdir / "q1_scree.png", dpi=160, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        "eigenvalues_desc": lam.tolist(),
        "errors_Fro": errors,
        "cumulative_variance": varexp
    }
    with open(outdir / "q1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Console summary for quick reporting
    print("=== Q1 PCA Summary ===")
    print("Eigenvalues (desc):", np.array(lam))
    print("Frobenius errors  :", np.array(errors))
    print("Cumulative var exp:", np.array(varexp))
    # Simple rule-of-thumb choice of r (first r hitting >= 0.95 variance)
    r95 = next((i+1 for i,v in enumerate(varexp) if v >= 0.95), V.shape[1])
    print(f"Suggested r (>=95% variance): {r95}")

if __name__ == "__main__":
    main()
