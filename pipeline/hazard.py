"""Rock-hazard maps with confidence (docs/METHODS.md Section 8).

Input: calibrated rocks [(y, x, D_cm, P)] on the 6.25 cm grid (pipeline/confidence.score), the base-trust mask, and
optionally a slope map (degrees, same grid).
For each map cell (default 1 m):
  * p_hazard = 1 - prod_i (1 - P_i) over rocks with D_i >= d_hazard in the cell: the probability that the cell holds at
    least one hazard rock, treating rock calls as independent;
  * expected_rocks = sum_i P_i (rocks >= d_hazard);
  * status: 'unknown' where less than half the cell is trusted base (calibrated uncertainty above the limit);
    'hazard' where p_hazard >= p_alarm or slope >= slope_max; otherwise 'clear'.
Size bands below ~60 cm are calibrated only in simulation (Section 7); the CSV records d_hazard so a user can see
which band drives the map.
"""
import numpy as np


def cells(shape, rocks, trust, px_m=0.0625, cell_m=1.0, d_hazard=40, p_alarm=0.5, slope=None, slope_max=15.0):
    H, W = shape; k = int(round(cell_m / px_m)); ny, nx = H // k, W // k
    logq = np.zeros((ny, nx)); exp = np.zeros((ny, nx)); nmax = np.zeros((ny, nx)); dmax = np.zeros((ny, nx))
    for y, x, D, p in rocks:
        if D < d_hazard: continue
        cy, cx = int(y) // k, int(x) // k
        if 0 <= cy < ny and 0 <= cx < nx:
            logq[cy, cx] += np.log1p(-min(p, 1 - 1e-9)); exp[cy, cx] += p
            if p > nmax[cy, cx]: nmax[cy, cx] = p; dmax[cy, cx] = D
    ph = 1 - np.exp(logq)
    tr = trust[:ny * k, :nx * k].reshape(ny, k, nx, k).mean((1, 3))
    sl = slope[:ny * k, :nx * k].reshape(ny, k, nx, k).max((1, 3)) if slope is not None else np.zeros((ny, nx))
    status = np.where(tr < 0.5, "unknown", np.where((ph >= p_alarm) | (sl >= slope_max), "hazard", "clear"))
    return dict(p_hazard=ph, expected_rocks=exp, max_p=nmax, max_p_diameter_cm=dmax, trusted_share=tr, slope_max_deg=sl,
                status=status, cell_m=cell_m, d_hazard_cm=d_hazard, p_alarm=p_alarm)


def to_csv(hz, path):
    ny, nx = hz["p_hazard"].shape
    with open(path, "w") as f:
        f.write("cell_x_m,cell_y_m,p_hazard,expected_rocks,max_p,max_p_diameter_cm,trusted_share,slope_max_deg,status\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{i * hz['cell_m']:.2f},{j * hz['cell_m']:.2f},{hz['p_hazard'][j, i]:.3f},{hz['expected_rocks'][j, i]:.3f},"
                        f"{hz['max_p'][j, i]:.3f},{int(hz['max_p_diameter_cm'][j, i])},{hz['trusted_share'][j, i]:.2f},{hz['slope_max_deg'][j, i]:.1f},{hz['status'][j, i]}\n")


def summary(hz):
    s = hz["status"]; n = s.size
    return dict(cells=int(n), hazard_pct=100 * float((s == "hazard").mean()), clear_pct=100 * float((s == "clear").mean()),
                unknown_pct=100 * float((s == "unknown").mean()), expected_hazard_rocks=float(hz["expected_rocks"].sum()),
                cell_m=hz["cell_m"], d_hazard_cm=hz["d_hazard_cm"], p_alarm=hz["p_alarm"])
