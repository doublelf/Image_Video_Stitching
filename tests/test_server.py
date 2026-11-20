from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from hstitch.server import app as server_module
from hstitch.server.jobs import JobStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A test client whose jobs live in a temporary directory."""
    store = JobStore(tmp_path / "jobs", workers=1)
    monkeypatch.setattr(server_module, "_store", store)
    with TestClient(server_module.app) as test_client:
        yield test_client
    store.shutdown()


def _wait(client, job_id, timeout: float = 60.0):
    deadline = time.time() + timeout
    state = {}
    while time.time() < deadline:
        response = client.get(f"/api/v1/stitch/videos/{job_id}")
        assert response.status_code == 200
        state = response.json()
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(0.2)
    raise AssertionError(f"the job did not finish: {state}")


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "hstitch" in response.text


def test_upload_and_download(client, video_pair):
    left, right = video_pair
    with left.open("rb") as first, right.open("rb") as second:
        response = client.post(
            "/api/v1/stitch/videos",
            files=[
                ("files", ("left.mp4", first, "video/mp4")),
                ("files", ("right.mp4", second, "video/mp4")),
            ],
        )
    assert response.status_code == 200
    job = response.json()
    assert job["status"] in ("pending", "running")
    assert job["job_id"]

    state = _wait(client, job["job_id"])
    assert state["status"] == "done", state.get("error")
    assert state["report"]["frames"] > 0
    assert state["report"]["size"][0] > 0

    result = client.get(f"/api/v1/stitch/videos/{job['job_id']}/result")
    assert result.status_code == 200
    assert result.headers["content-type"] == "video/mp4"
    assert len(result.content) > 1000


def test_a_single_file_is_rejected(client, video_pair):
    left, _ = video_pair
    with left.open("rb") as first:
        response = client.post(
            "/api/v1/stitch/videos",
            files=[("files", ("left.mp4", first, "video/mp4"))],
        )
    assert response.status_code == 400


def test_an_unknown_job_is_reported(client):
    assert client.get("/api/v1/stitch/videos/nope").status_code == 404
    assert client.get("/api/v1/stitch/videos/nope/result").status_code == 404


def test_the_concat_mode_can_be_requested(client, video_pair):
    left, right = video_pair
    with left.open("rb") as first, right.open("rb") as second:
        response = client.post(
            "/api/v1/stitch/videos",
            files=[
                ("files", ("left.mp4", first, "video/mp4")),
                ("files", ("right.mp4", second, "video/mp4")),
            ],
            data={"mode": "concat", "sync": "index"},
        )
    assert response.status_code == 200
    state = _wait(client, response.json()["job_id"])
    assert state["status"] == "done", state.get("error")
    assert state["report"]["mode"] == "concat"
    assert state["report"]["size"][0] == 660
