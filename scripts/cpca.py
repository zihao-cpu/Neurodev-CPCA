import numpy as np
from utils import load_concat_file
from scipy.signal import hilbert
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from joblib import Parallel, delayed
import fbpca

def complex_svd(pts, n_comp=10, seed=0, ref=None, normalized=True):
    # (Number of Time points x Number of Regions)
    
    anlytic_pts = hilbert(pts, axis=0)
    
    U, S, Vh = fbpca.pca(anlytic_pts, k=n_comp, n_iter=20, l=n_comp+10)
    """
    perm = slice(n_comp -1 + 10, n_comp - 1, -1) # svds returns ascending singular value order
    U, S, Vh = svds(anlytic_pts, k=n_comp + 10, random_state=seed)
    U = U[:, perm]
    S = S[perm]
    Vh = Vh[perm]
    """
    if ref is None:
        return U, S, Vh
    else:
        if normalized: # Calculate transform matrix based on right signular vector
            _u, _w, _vt = svd(ref.dot(Vh.conj().T).conj().T, full_matrices=False)
        else: # Calculate transform matrix based on PCA score
            _u, _w, _vt = svd(ref.dot(S * Vh.conj().T).conj().T, full_matrices=False)
        R = _u.dot(_vt)
        US_rot = (U * S) @ R
        Vh_rot = R.conj().T @ Vh
        S_rot = np.sqrt(np.sum(US_rot * US_rot.conj(), axis=0).real)
        U_rot = US_rot / S_rot
        return U_rot, S_rot, Vh_rot

    
def reconstruct(u, vh, n_bin=32):
    n_tps, n_comp = u.shape
    angle_u = np.mod(np.angle(u), 2 * np.pi)
    bins = np.linspace(0, 2 * np.pi, n_bin + 1)

    # Get the bin index for each element in angle_u
    bin_idx = np.digitize(angle_u, bins) - 1  # (n_tps, n_comp)

    # Initialize sum and count arrays
    sum_phase_u = np.zeros((n_bin, n_comp), dtype=np.complex128)
    count_phase_u = np.zeros((n_bin, n_comp), dtype=int)

    # Vectorized summing for each bin and component
    for comp in range(n_comp):
        np.add.at(sum_phase_u[:, comp], bin_idx[:, comp], u[:, comp])
        np.add.at(count_phase_u[:, comp], bin_idx[:, comp], 1)

    # Compute the mean of each bin
    mean_phase_u = np.divide(sum_phase_u, count_phase_u, out=np.zeros_like(sum_phase_u), where=count_phase_u != 0)

    # Compute phase_map
    phase_map = mean_phase_u[:, :, np.newaxis] * vh[np.newaxis, :, :]
    phase_map = np.transpose(phase_map, (1, 0, 2))
    return phase_map

def reconstruct_multiple(list_u, list_vh, n_bin=32, n_cpu=8):
    n_sub = len(list_u)
    list_phase_map = np.asarray(Parallel(n_jobs=n_cpu)(delayed(reconstruct)(list_u[sub], list_vh[sub], n_bin=n_bin) for sub in range(n_sub)))
    return list_phase_map
    
    

def cpca(pts, n_comp=10, seed=0, ref=None, normalized=False, n_bin=32, svd_results=False):
    U, S, Vh = complex_svd(pts, n_comp, seed, ref, normalized)
    patterns = reconstruct(U * S, Vh, n_bin)
    if svd_results:
        return U, S, Vh, patterns
    
    else:
        return patterns
    
    
def cpca_multiple(list_pts, n_comp=10, seed=0, ref=None, normalized=False, n_bin=32, n_cpu=8):
    n_sub = len(list_pts)
    list_phase_map = np.asarray(Parallel(n_jobs=n_cpu)(delayed(cpca)(list_pts[sub], n_comp, seed, ref, normalized, n_bin) for sub in range(n_sub)))
    return list_phase_map