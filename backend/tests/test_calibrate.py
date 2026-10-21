from app.data.calibrate import calibrate


def test_calibrate_offline_synthetic():
    out = calibrate("AAPL", allow_network=False)
    assert out["source"] == "synthetic"
    assert out["suggested_init_mid"] == 1000
    assert out["daily_vol"] > 0
