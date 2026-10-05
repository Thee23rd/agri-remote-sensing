"""Local monitoring app: spectral passes in, growth stage out."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agri_rs.service import calibrate, infer, list_crops, simulate

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Agriculture & Remote Sensing", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class SimulateRequest(BaseModel):
    crop: str = "maize"
    planting_date: Optional[str] = None
    interval_days: int = 5
    cloud_prob: float = 0.18
    seed: Optional[int] = None


class InferRequest(BaseModel):
    crop: str = "maize"
    observations: list = Field(default_factory=list)
    parameters: Optional[dict] = None


class CalibrateRequest(BaseModel):
    crop: str = "maize"
    observations: list = Field(default_factory=list)
    source: str = "simulator"
    n_fields: int = 28
    seed: int = 7


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/crops")
def crops() -> dict:
    return {"crops": list_crops()}


@app.post("/api/simulate")
def simulate_season(body: SimulateRequest) -> dict:
    return _call(
        simulate,
        body.crop,
        planting_date=body.planting_date,
        interval_days=body.interval_days,
        cloud_prob=body.cloud_prob,
        seed=body.seed,
    )


@app.post("/api/infer")
def infer_stages(body: InferRequest) -> dict:
    return _call(infer, body.crop, body.observations, parameters=body.parameters)


@app.post("/api/calibrate")
def calibrate_model(body: CalibrateRequest) -> dict:
    return _call(
        calibrate,
        body.crop,
        body.observations,
        source=body.source,
        n_fields=body.n_fields,
        seed=body.seed,
    )


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        raise HTTPException(status_code=404, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
