"""Camera model of Ingenuity's colour camera (HELI_RTE, 4208 x 3120) from the CAHVORE parameters in the image metadata.
C, A, H, V, O, R, E, then type and linearity. pixel -> ray: CAHV gives the distorted ray; CAHVOR distortion is inverted by fixed-point iteration
(lambda' = lambda + mu (lambda - zeta O), zeta = lambda.O, tau = |lambda|^2 / zeta^2 - 1, mu = R0 + R1 tau + R2 tau^2). The E (entrance-pupil) terms are ~1e-3 and ignored.
Rays are returned in a camera frame (x right, y down, z along the axis A) as normalised pinhole coordinates (x/z, y/z)."""
import json, numpy as np
def parse(s):
    v = [tuple(float(x) for x in t.strip("()").split(",")) for t in s.split(";")[:7]]; return [np.array(x) for x in v]
class Rte:
    def __init__(self, model_str):
        self.C, self.A, self.H, self.V, self.O, self.R, self.E = parse(model_str)
        self.hs = np.linalg.norm(np.cross(self.A, self.H)); self.vs = np.linalg.norm(np.cross(self.A, self.V)); self.hc = self.A @ self.H; self.vc = self.A @ self.V
        self.Hp = (self.H - self.hc * self.A) / self.hs; self.Vp = (self.V - self.vc * self.A) / self.vs          # unit image axes (right, down) in the model frame
    def rays(self, px):
        """px: (n, 2) full-resolution pixel coords -> (n, 2) normalised coordinates (x/z, y/z) in the (Hp, Vp, A) frame, distortion removed"""
        x, y = px[:, 0], px[:, 1]; d = (self.H[None] - x[:, None] * self.A[None]); e = (self.V[None] - y[:, None] * self.A[None]); lam_p = np.cross(d, e)
        lam_p *= np.sign(lam_p @ self.A)[:, None]; lam_p /= np.linalg.norm(lam_p, axis=1)[:, None]; lam = lam_p.copy()
        for _ in range(200):                                              # damped fixed-point iteration (undamped diverges in the corners, R1 = 0.47)
            z = lam @ self.O; tau = (lam * lam).sum(1) / z ** 2 - 1; mu = self.R[0] + self.R[1] * tau + self.R[2] * tau ** 2
            new = lam_p - mu[:, None] * (lam - z[:, None] * self.O[None]); lam = 0.6 * lam + 0.4 * new; lam /= np.linalg.norm(lam, axis=1)[:, None]
        zz = lam @ self.A; return np.stack([(lam @ self.Hp) / zz, (lam @ self.Vp) / zz], 1)
if __name__ == "__main__":
    it = json.load(open(__import__("sys").argv[1])); r = [i for i in it if i["camera"]["instrument"] == "HELI_RTE"][0]; m = Rte(r["camera"]["camera_model_component_list"])
    print("fx %.1f fy %.1f cx %.1f cy %.1f   angle between axis A and optical axis O: %.2f deg" % (m.hs, m.vs, m.hc, m.vc, np.degrees(np.arccos(np.clip(m.A @ m.O, -1, 1)))))
    for name, p in (("centre", (2104, 1560)), ("left edge", (0, 1560)), ("right edge", (4208, 1560)), ("top edge", (2104, 0)), ("corner", (0, 0))): q = m.rays(np.array([p], float))[0]; print(f"  {name:10s} -> normalised ({q[0]:+.3f}, {q[1]:+.3f})  angle off axis {np.degrees(np.arctan(np.hypot(*q))):.1f} deg   pinhole would be ({(p[0]-m.hc)/m.hs:+.3f}, {(p[1]-m.vc)/m.vs:+.3f})")
