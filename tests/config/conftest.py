"""A real temporary workspace and a valid candidate to mutate, for the WP1 config suites."""

import copy

import pytest

DIGEST = "sha256:" + "ab" * 32

MODEL_REF = {"id": "line-clearance-cam-01", "version": "2026.08.20", "digest": DIGEST}


@pytest.fixture
def workspace(tmp_path):
    """Real directories, because the validator checks that completion targets are usable."""
    for relative in ("spool/cam-01", "state", "models", "staging", "processed", "failed",
                     "inspection"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def global_config(workspace):
    """A `component.global` object that satisfies every rule."""
    return {
        "paths": {
            "stateDb": str(workspace / "state" / "state.db"),
            "modelCache": str(workspace / "models"),
            "staging": str(workspace / "staging"),
        },
        "runtime": {
            "providers": ["CPUExecutionProvider"],
            "requiredProvider": "CPUExecutionProvider",
            "allowCpuOnly": True,
        },
        "models": [
            dict(
                MODEL_REF,
                uri="s3://approved-models/line-clearance-cam-01/2026.08.20.tar.gz",
                credentials={"$secret": "model-source/approved-models"},
            )
        ],
    }


@pytest.fixture
def spool_route(workspace):
    """A camera-bound spool route that archives its inputs."""
    return {
        "id": "clearance-cam-01",
        "priority": 100,
        "source": {
            "kind": "spool",
            "root": str(workspace / "spool" / "cam-01"),
            "readiness": {"mode": "cameraSidecar"},
            "camera": {"component": "camera-adapter", "instance": "cam-01"},
        },
        "modelRef": dict(MODEL_REF),
        "outputs": {
            "writeResultSidecar": True,
            "decisionSignals": [{"id": "line-clearance/pass", "value": "$.decision.pass"}],
        },
        "completion": {
            "onSuccess": "archive",
            "archiveDir": str(workspace / "processed"),
            "failedDir": str(workspace / "failed"),
        },
    }


@pytest.fixture
def trigger_route(workspace):
    """A subscription route that deletes what it inspects."""
    return {
        "id": "adhoc-inspect",
        "source": {
            "kind": "trigger",
            "subscribe": ["ecv1/+/inspection-ui/+/app/inspect/request"],
            "fileRoot": str(workspace / "inspection"),
            "inlineStaging": str(workspace / "staging" / "adhoc"),
        },
        "modelRef": dict(MODEL_REF),
        "outputs": {"writeResultSidecar": False, "decisionSignals": []},
        "completion": {"onSuccess": "delete"},
    }


@pytest.fixture
def candidate(global_config, spool_route, trigger_route):
    """A whole candidate document that the validator accepts unchanged."""
    return {
        "component": {
            "token": "image-processor",
            "global": copy.deepcopy(global_config),
            "instances": [copy.deepcopy(spool_route), copy.deepcopy(trigger_route)],
        }
    }
