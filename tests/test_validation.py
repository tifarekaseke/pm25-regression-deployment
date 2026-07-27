from fastapi.testclient import TestClient

from summative.API.prediction import app

client = TestClient(app)


def valid_payload() -> dict:
    return {
        "month": 7,
        "hour": 14,
        "pm10": 120,
        "so2": 12,
        "no2": 45,
        "co": 900,
        "o3": 70,
        "temperature": 28.5,
        "pressure": 1004,
        "dew_point": 17.2,
        "rainfall": 0,
        "wind_speed": 2.4,
        "wind_direction": "SE",
        "station": "Aotizhongxin",
    }


def test_out_of_range_month_is_rejected() -> None:
    payload = valid_payload()
    payload["month"] = 20
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_wrong_datatype_is_rejected() -> None:
    payload = valid_payload()
    payload["hour"] = "afternoon"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_health_endpoint_exists() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
