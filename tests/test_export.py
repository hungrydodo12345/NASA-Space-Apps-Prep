import numpy as np
from pipeline import export


def test_pack_roundtrip_within_quantisation(tmp_path):
    rng = np.random.default_rng(0)
    emin, emax = -2629.2463, -2466.8210        # the prototype's LOCAL_ELEV_MIN/MAX
    h = emin + rng.random((64, 64)) * (emax - emin)
    p = tmp_path / "dem.png"
    export.write_png(p, export.pack_elevation(h, emin, emax))
    back = export.unpack_elevation(export.read_png(p), emin, emax)
    step = (emax - emin) / 4095
    assert np.abs(back - h).max() <= step / 2 + 1e-9
    print(f"quantisation step {step:.4f} m, max err {np.abs(back-h).max():.4f} m")


def test_fill_gaps_flags_inferred():
    h = np.full((32, 32), 5.0); valid = np.ones_like(h, bool); valid[10:14, 10:14] = False
    filled, inferred = export.fill_gaps(np.where(valid, h, np.nan), valid)
    assert inferred.sum() == 16 and np.allclose(filled, 5.0)


def test_out_of_range_is_clipped_not_wrapped():
    emin, emax = 0.0, 100.0
    rgb = export.pack_elevation(np.array([[-50.0, 150.0]]), emin, emax)
    back = export.unpack_elevation(rgb, emin, emax)
    assert back[0, 0] == emin and back[0, 1] == emax
