import numpy as np, json
from scipy.io import loadmat
from scipy.linalg import eigh

def load_x():
    d = loadmat("xb.mat", squeeze_me=True)
    for k in ("x", "xb", "signal", "data", "X", "y"):
        if k in d:
            x = np.asarray(d[k], dtype=float).ravel()
            break
    else:
        ks = [k for k in d.keys() if not k.startswith("__")]
        x = np.asarray(d[ks[0]], dtype=float).ravel()
    x = x[np.isfinite(x)]
    return x - x.mean()

def build_A(x, N):
    L = len(x)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            m = max(i, j)
            if m < L:
                M = L - m
                A[i, j] = np.dot(x[i:i+M], x[j:j+M])
    return (A + A.T) / 2

def main():
    x = load_x()
    results = []
    for N in [8, 12, 16, 24]:
        A = build_A(x, N)
        w, _ = eigh(A)
        results.append({"N": N, "lambda_min": float(w[0])})
    with open("q3_summary.json", "w") as f:
        json.dump({"results": results}, f, indent=2)
    print("q3_summary.json saved")

if __name__ == "__main__":
    main()