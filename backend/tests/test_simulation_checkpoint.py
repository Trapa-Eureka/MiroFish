"""
模拟检查点（Checkpoint）测试

覆盖 app/services/simulation_checkpoint.py 的构建/保存/加载，以及
_monitor_simulation 在轮次前进时写入检查点、并在失败终态记录"到达轮次 +
明确的续跑限制说明"的端到端行为。
"""

import json

import pytest

from app import create_app
from app.services import simulation_checkpoint
from app.services.simulation_checkpoint import (
    RESUME_LIMITATION_NOTE,
    Checkpoint,
    build_checkpoint_from_state,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from app.services import simulation_runner as runner_module
from app.services.simulation_runner import RunnerStatus, SimulationRunState, SimulationRunner


class TestCheckpointDataStructure:
    def test_checkpoint_defaults_are_not_resumable(self):
        cp = Checkpoint(simulation_id="sim_test1234")
        assert cp.resumable is False
        assert cp.resume_limitation == RESUME_LIMITATION_NOTE

    def test_to_dict_shape(self):
        cp = Checkpoint(
            simulation_id="sim_test1234",
            twitter_round=5,
            reddit_round=3,
            total_rounds=10,
            twitter_action_count=42,
            reddit_action_count=17,
            runner_status="running",
        )
        data = cp.to_dict()
        assert data["simulation_id"] == "sim_test1234"
        assert data["twitter_round"] == 5
        assert data["reddit_round"] == 3
        assert data["resumable"] is False
        assert "checkpointed_at" in data


class TestBuildCheckpointFromState:
    def test_reflects_run_state_fields(self):
        state = SimulationRunState(
            simulation_id="sim_test1234",
            runner_status=RunnerStatus.RUNNING,
            total_rounds=100,
            twitter_current_round=30,
            reddit_current_round=28,
            twitter_actions_count=120,
            reddit_actions_count=95,
        )
        cp = build_checkpoint_from_state(state)
        assert cp.simulation_id == "sim_test1234"
        assert cp.twitter_round == 30
        assert cp.reddit_round == 28
        assert cp.total_rounds == 100
        assert cp.twitter_action_count == 120
        assert cp.reddit_action_count == 95
        assert cp.runner_status == "running"


class TestSaveAndLoadCheckpoint:
    def test_round_trip(self, tmp_path):
        cp = Checkpoint(simulation_id="sim_test1234", twitter_round=7, total_rounds=50)
        save_checkpoint(str(tmp_path), cp)

        assert (tmp_path / "checkpoint.json").exists()
        loaded = load_checkpoint(str(tmp_path))
        assert loaded["simulation_id"] == "sim_test1234"
        assert loaded["twitter_round"] == 7
        assert loaded["resumable"] is False
        assert loaded["resume_limitation"] == RESUME_LIMITATION_NOTE

    def test_missing_checkpoint_returns_none(self, tmp_path):
        assert load_checkpoint(str(tmp_path)) is None

    def test_checkpoint_path(self, tmp_path):
        assert checkpoint_path(str(tmp_path)) == str(tmp_path / "checkpoint.json")

    def test_overwrite_replaces_content(self, tmp_path):
        save_checkpoint(str(tmp_path), Checkpoint(simulation_id="sim_1", twitter_round=1))
        save_checkpoint(str(tmp_path), Checkpoint(simulation_id="sim_1", twitter_round=2))
        loaded = load_checkpoint(str(tmp_path))
        assert loaded["twitter_round"] == 2


class TestMonitorSimulationWritesCheckpoints:
    """
    端到端验证 _monitor_simulation：轮次前进时写入检查点，失败终态时写入
    最终检查点并在 state.error 中附上明确的"不可续跑，需从 round 0 重开"说明。
    """

    def _write_round_end(self, log_path, round_num, simulated_hours=1):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event_type": "round_end",
                        "round": round_num,
                        "simulated_hours": simulated_hours,
                    }
                )
                + "\n"
            )

    def test_checkpoint_written_during_run_and_on_failure(self, tmp_path, monkeypatch):
        simulation_id = "sim_checkpoint1"
        sim_dir = tmp_path / simulation_id
        twitter_log = sim_dir / "twitter" / "actions.jsonl"
        self._write_round_end(twitter_log, round_num=5)

        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.RUNNING,
            total_rounds=100,
            twitter_running=True,
        )

        monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
        monkeypatch.setattr(SimulationRunner, "_run_states", {simulation_id: state})
        monkeypatch.setattr(SimulationRunner, "_processes", {})
        monkeypatch.setattr(SimulationRunner, "_manual_stop_requests", set())
        monkeypatch.setattr(SimulationRunner, "_graph_memory_enabled", {})
        monkeypatch.setattr(runner_module.time, "sleep", lambda _: None)
        monkeypatch.setattr(
            SimulationRunner, "_sync_simulation_status", classmethod(lambda *a, **k: None)
        )

        poll_results = iter([None, 1])  # one loop iteration, then process exits (code 1 = failure)

        class FakeProcess:
            pid = 999
            returncode = 1

            def poll(self):
                return next(poll_results, 1)

        SimulationRunner._processes[simulation_id] = FakeProcess()

        SimulationRunner._monitor_simulation(simulation_id, locale="zh")

        # A checkpoint was written mid-loop, reflecting the round_end event.
        mid_run_checkpoint = load_checkpoint(str(sim_dir))
        assert mid_run_checkpoint is not None
        assert mid_run_checkpoint["twitter_round"] == 5

        # Final state: FAILED, with an honest, non-resumable explanation.
        final_state = SimulationRunner._run_states[simulation_id]
        assert final_state.runner_status == RunnerStatus.FAILED
        assert "round 0" in final_state.error
        assert "重新开始" in final_state.error
        assert "twitter=5" in final_state.error

        final_checkpoint = load_checkpoint(str(sim_dir))
        assert final_checkpoint["runner_status"] == "failed"
        assert final_checkpoint["resumable"] is False
        assert final_checkpoint["resume_limitation"] == RESUME_LIMITATION_NOTE


class TestCheckpointApiEndpoint:
    def test_get_checkpoint_returns_persisted_checkpoint(self, tmp_path, monkeypatch):
        simulation_id = "sim_apitest1234"
        monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
        sim_dir = tmp_path / simulation_id
        sim_dir.mkdir()
        save_checkpoint(
            str(sim_dir),
            Checkpoint(simulation_id=simulation_id, twitter_round=12, total_rounds=50),
        )

        app = create_app()
        app.config.update(TESTING=True)
        client = app.test_client()

        response = client.get(f"/api/simulation/{simulation_id}/checkpoint")
        assert response.status_code == 200
        assert response.json["data"]["twitter_round"] == 12
        assert response.json["data"]["resumable"] is False

    def test_get_checkpoint_404_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))

        app = create_app()
        app.config.update(TESTING=True)
        client = app.test_client()

        response = client.get("/api/simulation/sim_nope12345/checkpoint")
        assert response.status_code == 404
