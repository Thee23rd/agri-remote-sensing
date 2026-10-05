from fastapi import HTTPException

from app.main import InferRequest, SimulateRequest, crops, index, infer_stages, simulate_season


def test_crops_and_a_simulated_season_round_trip():
    listed = crops()
    assert {item["id"] for item in listed["crops"]} == {"maize", "wheat", "soybean"}

    season = simulate_season(
        SimulateRequest(crop="maize", seed=9, cloud_prob=0.2, interval_days=5)
    )
    assert season["clear_passes"] >= 8
    assert season["observations"][0]["simulator_stage"] == 0

    result = infer_stages(
        InferRequest(crop="maize", observations=season["observations"])
    )
    assert len(result["timeline"]) == season["clear_passes"]
    assert result["model_label"] == "Textbook prior"
    assert 0.0 <= result["agreement"] <= 1.0


def test_rejects_an_index_outside_the_physical_range():
    try:
        infer_stages(
            InferRequest(
                crop="maize",
                observations=[{"date": "2025-11-20", "ndvi": 4.2}],
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "ndvi" in exc.detail
    else:
        raise AssertionError("an index of 4.2 should be rejected")


def test_home_page_is_the_monitoring_board():
    response = index()
    assert str(response.path).endswith("index.html")
    text = open(response.path, encoding="utf-8").read()
    assert "Agriculture" in text
    assert "hidden states" in text
