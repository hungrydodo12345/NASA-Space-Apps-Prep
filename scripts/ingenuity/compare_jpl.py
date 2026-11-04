"""Compare my flight-26 mosaic with JPL's by image content: SIFT + RANSAC similarity between a JPL tile (2 cm/px, vertical projection) and my mosaic, then the geometric
residual of the matched points after the best similarity (my geometry against JPL's, in centimetres), the scale ratio and the rotation between them.
Usage: python compare_jpl.py <scratch_dir> <my_mosaic_npz> <jpl_tile_index> [my_scale_mpp]"""
import sys, json, numpy as np, cv2
S, MOS, K = sys.argv[1], sys.argv[2], int(sys.argv[3]); MY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0290
j = np.load(f"{S}/jplmos/tile{K}.npy"); v = j != 0; lo, hi = np.percentile(j[v], [1, 99.5]); J = (np.clip((j - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
m = np.load(MOS); M = m["mosaic"]; Vm = np.isfinite(M); lo, hi = np.percentile(M[Vm], [1, 99.5]); My = (np.clip((np.nan_to_num(M, nan=lo) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
f = 0.02 / MY; Jr = cv2.resize(J, None, fx=f, fy=f, interpolation=cv2.INTER_AREA); vr = cv2.resize(v.astype(np.uint8), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) * 255              # JPL tile at my nominal scale
cl = cv2.createCLAHE(3.0, (8, 8)); sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.005)
k1, d1 = sift.detectAndCompute(cl.apply(My), (Vm * 255).astype(np.uint8)); k2, d2 = sift.detectAndCompute(cl.apply(Jr), cv2.erode(vr, np.ones((15, 15), np.uint8)))
print(f"tile {K}: features mine {len(k1)}, JPL {len(k2)}")
mt = [a for a, b in (z for z in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(z) == 2) if a.distance < 0.85 * b.distance]; print("ratio-test matches:", len(mt))
if len(mt) >= 8:
    p1 = np.float32([k1[x.queryIdx].pt for x in mt]); p2 = np.float32([k2[x.trainIdx].pt for x in mt]); A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=10000)
    n = int(inl.sum()) if inl is not None else 0; print("similarity inliers:", n)
    if A is not None and n >= 8:
        s = float(np.hypot(A[0, 0], A[1, 0])); rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))); q = inl.ravel() > 0; res = np.linalg.norm((p1[q] @ A[:, :2].T + A[:, 2]) - p2[q], axis=1) * MY * 100
        print(f"scale ratio (JPL tile px per my px, after nominal rescale): {s:.4f}  -> my 0.0290 m/px corresponds to {MY / s:.4f} m/px in JPL's frame; rotation between the mosaics {rot:.1f} deg; residual of matched points: median {np.median(res):.1f} cm, 90th {np.percentile(res, 90):.1f} cm (over {q.sum()} points)")
        np.save(f"{S}/jplmos/match_tile{K}.npy", A)
