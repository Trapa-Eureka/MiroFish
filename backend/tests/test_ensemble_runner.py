"""
集成模拟（Ensemble Simulation）测试

覆盖 app/services/ensemble_runner.py：
- 输入校验（run_count 边界、来源模拟不存在/未准备好）
- 成功路径：N 个成员被独立启动，人设/配置文件被原样拷贝，owner_id/
  random_seed 被正确继承
- 单个成员启动失败不应该让整个集成失败（记录在该成员的 start_error 上）
- 聚合统计只在所有成员到达终态后现算：运行中 -> completed=0 的聚合为 None；
  全部完成 -> 中位数/方差/置信区间/结果频率/敏感度都被正确计算；
  全部失败 -> 整体状态为 failed 且没有聚合
- stop_ensemble 只停止仍处于非终态的成员，跳过启动失败/已终态的成员
"""

import json
import os

import pytest

from app.services import ensemble_runner as ensemble_runner_module
from app.services.ensemble_runner import (
    MAX_ENSEMBLE_RUN_COUNT,
    MIN_ENSEMBLE_RUN_COUNT,
    EnsembleRunner,
)
from app.services import simulation_runner as runner_module
from app.services.simulation_runner import RunnerStatus, SimulationRunner
from app.services.simulation_manager import SimulationManager, SimulationState, SimulationStatus


class FakeProcess:
    pid = 4242

    def poll(self):
        return None


class NoOpThread:
    """Replaces threading.Thread so _monitor_simulation never actually runs;
    the member's run_state stays exactly as start_simulation itself left it
    (RUNNING), which is all these tests need."""

    def __init__(self, **_kwargs):
        pass

    def start(self):
        pass


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    simulations_dir = tmp_path / "simulations"
    scripts_dir = tmp_path / "scripts"
    ensembles_dir = tmp_path / "ensembles"
    simulations_dir.mkdir()
    scripts_dir.mkdir()

    for script_name in (
        "run_parallel_simulation.py",
        "run_twitter_simulation.py",
        "run_reddit_simulation.py",
    ):
        (scripts_dir / script_name).write_text("pass\n", encoding="utf-8")

    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulations_dir))
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(simulations_dir))
    monkeypatch.setattr(SimulationRunner, "SCRIPTS_DIR", str(scripts_dir))
    monkeypatch.setattr(EnsembleRunner, "ENSEMBLE_DATA_DIR", str(ensembles_dir))

    monkeypatch.setattr(SimulationRunner, "_run_states", {})
    monkeypatch.setattr(SimulationRunner, "_processes", {})
    monkeypatch.setattr(SimulationRunner, "_action_queues", {})
    monkeypatch.setattr(SimulationRunner, "_monitor_threads", {})
    monkeypatch.setattr(SimulationRunner, "_stdout_files", {})
    monkeypatch.setattr(SimulationRunner, "_stderr_files", {})
    monkeypatch.setattr(SimulationRunner, "_graph_memory_enabled", {})
    monkeypatch.setattr(SimulationRunner, "_manual_stop_requests", set())

    monkeypatch.setattr(runner_module.subprocess, "Popen", lambda *_a, **_k: FakeProcess())
    monkeypatch.setattr(runner_module.threading, "Thread", NoOpThread)
    monkeypatch.setattr(
        SimulationRunner, "_sync_simulation_status", classmethod(lambda *a, **k: None)
    )

    yield


def _make_source_simulation(
    simulation_id="sim_source12345",
    owner_id="user-1",
    enable_reddit=True,
    enable_twitter=False,
    random_seed=777,
):
    manager = SimulationManager()
    state = SimulationState(
        simulation_id=simulation_id,
        project_id="proj_test1234",
        graph_id="graph-test-1",
        enable_twitter=enable_twitter,
        enable_reddit=enable_reddit,
        status=SimulationStatus.READY,
        profiles_generated=True,
        config_generated=True,
        owner_id=owner_id,
        random_seed=random_seed,
    )
    manager._save_simulation_state(state)

    sim_dir = manager._get_simulation_dir(simulation_id)
    with open(os.path.join(sim_dir, "simulation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"time_config": {"total_simulation_hours": 1, "minutes_per_round": 60}}, f)

    if enable_reddit:
        with open(os.path.join(sim_dir, "reddit_profiles.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": 1, "username": "alice"}], f)
    if enable_twitter:
        with open(os.path.join(sim_dir, "twitter_profiles.csv"), "w", encoding="utf-8") as f:
            f.write("user_id,username\n1,alice\n")

    return state


class TestStartEnsembleValidation:
    def test_run_count_below_minimum_rejected(self):
        _make_source_simulation()
        with pytest.raises(ValueError, match="run_count"):
            EnsembleRunner.start_ensemble("sim_source12345", run_count=MIN_ENSEMBLE_RUN_COUNT - 1)

    def test_run_count_above_maximum_rejected(self):
        _make_source_simulation()
        with pytest.raises(ValueError, match="run_count"):
            EnsembleRunner.start_ensemble("sim_source12345", run_count=MAX_ENSEMBLE_RUN_COUNT + 1)

    def test_non_integer_run_count_rejected(self):
        _make_source_simulation()
        with pytest.raises(ValueError, match="run_count"):
            EnsembleRunner.start_ensemble("sim_source12345", run_count=3.5)

    def test_bool_run_count_rejected(self):
        # isinstance(True, int) is True in Python; must be explicitly excluded.
        _make_source_simulation()
        with pytest.raises(ValueError, match="run_count"):
            EnsembleRunner.start_ensemble("sim_source12345", run_count=True)

    def test_missing_source_simulation_rejected(self):
        with pytest.raises(ValueError, match="不存在"):
            EnsembleRunner.start_ensemble("sim_doesnotexist", run_count=3)

    def test_unprepared_source_simulation_rejected(self):
        manager = SimulationManager()
        state = SimulationState(
            simulation_id="sim_unprepared1",
            project_id="proj_test1234",
            graph_id="graph-test-1",
            status=SimulationStatus.CREATED,
            profiles_generated=False,
            config_generated=False,
        )
        manager._save_simulation_state(state)

        with pytest.raises(ValueError, match="准备"):
            EnsembleRunner.start_ensemble("sim_unprepared1", run_count=3)


class TestStartEnsembleHappyPath:
    def test_members_are_independently_started_from_shared_source(self):
        source = _make_source_simulation()
        manager = SimulationManager()

        record = EnsembleRunner.start_ensemble(
            "sim_source12345", run_count=3, platform="reddit"
        )

        assert record.run_count == 3
        assert len(record.members) == 3
        assert all(m.start_error is None for m in record.members)

        member_ids = [m.simulation_id for m in record.members]
        assert len(set(member_ids)) == 3  # all distinct

        for member in record.members:
            member_state = manager.get_simulation(member.simulation_id)
            assert member_state is not None
            assert member_state.status == SimulationStatus.READY
            assert member_state.config_generated is True
            assert member_state.profiles_generated is True
            assert member_state.owner_id == source.owner_id
            assert member_state.random_seed == source.random_seed

            member_dir = manager._get_simulation_dir(member.simulation_id)
            with open(os.path.join(member_dir, "reddit_profiles.json"), encoding="utf-8") as f:
                assert json.load(f) == [{"id": 1, "username": "alice"}]

            run_state = SimulationRunner.get_run_state(member.simulation_id)
            assert run_state is not None
            assert run_state.runner_status == RunnerStatus.RUNNING

        # The ensemble record itself is persisted and reloadable.
        reloaded = EnsembleRunner.get_ensemble_record(record.ensemble_id)
        assert reloaded is not None
        assert [m.simulation_id for m in reloaded.members] == member_ids

    def test_platform_defaults_to_source_default_platform(self):
        _make_source_simulation(enable_reddit=True, enable_twitter=False)
        record = EnsembleRunner.start_ensemble("sim_source12345", run_count=2)
        assert record.platform == "reddit"

    def test_twitter_only_source_only_copies_csv_profiles(self):
        _make_source_simulation(
            simulation_id="sim_twsource123",
            enable_reddit=False,
            enable_twitter=True,
        )
        manager = SimulationManager()

        record = EnsembleRunner.start_ensemble("sim_twsource123", run_count=2)

        for member in record.members:
            member_dir = manager._get_simulation_dir(member.simulation_id)
            assert os.path.exists(os.path.join(member_dir, "twitter_profiles.csv"))
            assert not os.path.exists(os.path.join(member_dir, "reddit_profiles.json"))


class TestPartialMemberStartFailure:
    def test_one_member_failing_to_start_does_not_abort_the_others(self, monkeypatch):
        _make_source_simulation()

        call_count = {"n": 0}

        def flaky_popen(*_a, **_k):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise OSError("simulated launch failure")
            return FakeProcess()

        monkeypatch.setattr(runner_module.subprocess, "Popen", flaky_popen)

        record = EnsembleRunner.start_ensemble("sim_source12345", run_count=3)

        errors = [m.start_error for m in record.members]
        assert errors[0] is None
        assert errors[1] is not None and "simulated launch failure" in errors[1]
        assert errors[2] is None

        summary = EnsembleRunner.get_ensemble_summary(record.ensemble_id)
        assert summary["status"] == "running"  # members 0 and 2 are still RUNNING
        assert summary["member_counts"]["failed"] == 1
        assert summary["member_counts"]["running"] == 2
        assert summary["aggregate"] is None


class TestEnsembleMembersRunNonInteractively:
    def test_members_are_started_with_no_wait(self, monkeypatch):
        # Ensemble members have no human operator to send interview/close
        # commands; without --no-wait the child process would sit in the
        # post-rounds command loop forever and the member would never reach
        # COMPLETED, so the ensemble could never aggregate.
        _make_source_simulation()

        captured_cmds = []
        real_popen = runner_module.subprocess.Popen

        def capturing_popen(cmd, *args, **kwargs):
            captured_cmds.append(cmd)
            return FakeProcess()

        monkeypatch.setattr(runner_module.subprocess, "Popen", capturing_popen)

        EnsembleRunner.start_ensemble("sim_source12345", run_count=2)

        assert len(captured_cmds) == 2
        assert all("--no-wait" in cmd for cmd in captured_cmds)


class TestEnsemblePlatformMustMatchPreparedProfiles:
    def test_twitter_platform_rejected_when_source_has_no_twitter_profiles(self):
        _make_source_simulation(enable_reddit=True, enable_twitter=False)
        with pytest.raises(ValueError, match="Twitter"):
            EnsembleRunner.start_ensemble(
                "sim_source12345", run_count=2, platform="twitter"
            )

    def test_reddit_platform_rejected_when_source_has_no_reddit_profiles(self):
        _make_source_simulation(
            simulation_id="sim_tw_only1234", enable_reddit=False, enable_twitter=True
        )
        with pytest.raises(ValueError, match="Reddit"):
            EnsembleRunner.start_ensemble(
                "sim_tw_only1234", run_count=2, platform="reddit"
            )

    def test_parallel_platform_rejected_when_source_is_single_platform(self):
        _make_source_simulation(enable_reddit=True, enable_twitter=False)
        with pytest.raises(ValueError, match="Twitter"):
            EnsembleRunner.start_ensemble(
                "sim_source12345", run_count=2, platform="parallel"
            )

    def test_matching_platform_is_accepted(self):
        _make_source_simulation(enable_reddit=True, enable_twitter=False)
        record = EnsembleRunner.start_ensemble(
            "sim_source12345", run_count=2, platform="reddit"
        )
        assert record.platform == "reddit"


class TestEnsembleRecordPersistedBeforeMembersLaunch:
    def test_ensemble_is_discoverable_even_if_launch_loop_never_completes(self, monkeypatch):
        # Simulate a crash partway through the launch loop (e.g. the process
        # is killed after member 0 starts but before member 1 is attempted).
        # The provisional record written before the loop must already make
        # the ensemble (and its deterministic member ids) discoverable, so a
        # crash never leaves orphaned simulation subprocesses with no
        # ensemble record pointing at them.
        _make_source_simulation()

        original_start = SimulationRunner.start_simulation
        call_count = {"n": 0}

        def crash_after_first_member(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise KeyboardInterrupt("simulated worker crash mid-loop")
            return original_start(*args, **kwargs)

        monkeypatch.setattr(
            SimulationRunner, "start_simulation", classmethod(
                lambda cls, *a, **k: crash_after_first_member(*a, **k)
            )
        )

        captured_ensemble_id = {}
        real_write = ensemble_runner_module.atomic_write_json

        def spying_write(path, data):
            if "ensemble_id" in data and data.get("members"):
                captured_ensemble_id["id"] = data["ensemble_id"]
            real_write(path, data)

        monkeypatch.setattr(ensemble_runner_module, "atomic_write_json", spying_write)

        with pytest.raises(KeyboardInterrupt):
            EnsembleRunner.start_ensemble("sim_source12345", run_count=3)

        ensemble_id = captured_ensemble_id["id"]
        reloaded = EnsembleRunner.get_ensemble_record(ensemble_id)
        assert reloaded is not None
        assert len(reloaded.members) == 3  # full deterministic member list, not just member 0

    def test_final_write_failure_does_not_abort_already_started_members(self, monkeypatch):
        # If the final (detailed start_error) write fails, already-launched
        # member processes must not be torn down just because bookkeeping
        # failed — they are real, possibly-costly running simulations.
        _make_source_simulation()

        real_write = ensemble_runner_module.atomic_write_json
        write_count = {"n": 0}

        def flaky_final_write(path, data):
            write_count["n"] += 1
            if write_count["n"] == 2:  # first call is the provisional write
                raise OSError("simulated disk failure on final write")
            real_write(path, data)

        monkeypatch.setattr(ensemble_runner_module, "atomic_write_json", flaky_final_write)

        record = EnsembleRunner.start_ensemble("sim_source12345", run_count=2)

        for member in record.members:
            run_state = SimulationRunner.get_run_state(member.simulation_id)
            assert run_state is not None
            assert run_state.runner_status == RunnerStatus.RUNNING

        reloaded = EnsembleRunner.get_ensemble_record(record.ensemble_id)
        assert reloaded is not None
        assert len(reloaded.members) == 2


def _complete_member(member_id, twitter_actions=0, reddit_actions=0, rounds_reached=0):
    state = SimulationRunner.get_run_state(member_id)
    state.runner_status = RunnerStatus.COMPLETED
    state.twitter_actions_count = twitter_actions
    state.reddit_actions_count = reddit_actions
    state.current_round = rounds_reached
    SimulationRunner._save_run_state(state)


def _fail_member(member_id):
    state = SimulationRunner.get_run_state(member_id)
    state.runner_status = RunnerStatus.FAILED
    state.error = "simulated failure"
    SimulationRunner._save_run_state(state)


def _write_action_log(member_id, platform, action_types):
    manager = SimulationManager()
    member_dir = manager._get_simulation_dir(member_id)
    platform_dir = os.path.join(member_dir, platform)
    os.makedirs(platform_dir, exist_ok=True)
    with open(os.path.join(platform_dir, "actions.jsonl"), "w", encoding="utf-8") as f:
        for action_type in action_types:
            f.write(json.dumps({"action_type": action_type, "round": 1}) + "\n")
        f.write(json.dumps({"event_type": "round_end", "round": 1}) + "\n")


class TestEnsembleSummaryAggregation:
    def test_status_is_running_until_every_member_reaches_a_terminal_state(self):
        _make_source_simulation()
        record = EnsembleRunner.start_ensemble("sim_source12345", run_count=3)
        member_ids = [m.simulation_id for m in record.members]

        _complete_member(member_ids[0], twitter_actions=0, reddit_actions=5, rounds_reached=2)
        # member_ids[1] and [2] stay RUNNING.

        summary = EnsembleRunner.get_ensemble_summary(record.ensemble_id)
        assert summary["status"] == "running"
        assert summary["aggregate"] is None

    def test_status_is_failed_when_no_member_completes(self):
        _make_source_simulation()
        record = EnsembleRunner.start_ensemble("sim_source12345", run_count=2)
        member_ids = [m.simulation_id for m in record.members]

        for member_id in member_ids:
            _fail_member(member_id)

        summary = EnsembleRunner.get_ensemble_summary(record.ensemble_id)
        assert summary["status"] == "failed"
        assert summary["member_counts"]["failed"] == 2
        assert summary["aggregate"] is None

    def test_aggregate_computed_once_all_members_terminal(self):
        _make_source_simulation()
        record = EnsembleRunner.start_ensemble("sim_source12345", run_count=3)
        member_ids = [m.simulation_id for m in record.members]

        _complete_member(member_ids[0], reddit_actions=10, rounds_reached=4)
        _complete_member(member_ids[1], reddit_actions=20, rounds_reached=6)
        _fail_member(member_ids[2])

        _write_action_log(member_ids[0], "reddit", ["CREATE_POST", "LIKE_POST"])
        _write_action_log(member_ids[1], "reddit", ["CREATE_POST", "CREATE_POST"])

        summary = EnsembleRunner.get_ensemble_summary(record.ensemble_id)
        assert summary["status"] == "completed"
        assert summary["member_counts"] == {
            "total": 3, "completed": 2, "failed": 1, "stopped": 0, "running": 0,
        }

        aggregate = summary["aggregate"]
        assert aggregate["completed_run_count"] == 2

        reddit_metric = aggregate["metrics"]["reddit_actions_count"]
        assert reddit_metric["count"] == 2
        assert reddit_metric["mean"] == 15
        assert reddit_metric["median"] == 15
        assert reddit_metric["min"] == 10
        assert reddit_metric["max"] == 20
        assert reddit_metric["confidence_interval_95"] is not None
        # Jackknife sensitivity is undefined (None) below 3 samples.
        assert reddit_metric["sensitivity"] is None

        rounds_metric = aggregate["metrics"]["rounds_reached"]
        assert rounds_metric["mean"] == 5

        freq = aggregate["outcome_frequency"]
        assert freq["CREATE_POST"]["run_occurrence_count"] == 2
        assert freq["CREATE_POST"]["run_occurrence_rate"] == 1.0
        assert freq["CREATE_POST"]["total_count"] == 3  # 1 + 2
        assert freq["LIKE_POST"]["run_occurrence_count"] == 1
        assert freq["LIKE_POST"]["run_occurrence_rate"] == 0.5
        assert freq["LIKE_POST"]["total_count"] == 1

    def test_sensitivity_reported_with_three_or_more_completed_members(self):
        _make_source_simulation()
        record = EnsembleRunner.start_ensemble("sim_source12345", run_count=3)
        member_ids = [m.simulation_id for m in record.members]

        _complete_member(member_ids[0], reddit_actions=10)
        _complete_member(member_ids[1], reddit_actions=20)
        _complete_member(member_ids[2], reddit_actions=90)  # outlier

        summary = EnsembleRunner.get_ensemble_summary(record.ensemble_id)
        sensitivity = summary["aggregate"]["metrics"]["reddit_actions_count"]["sensitivity"]
        assert sensitivity is not None
        assert sensitivity > 0


class TestStopEnsemble:
    def test_stop_only_targets_non_terminal_members_and_skips_start_failures(self, monkeypatch):
        _make_source_simulation()

        call_count = {"n": 0}

        def flaky_popen(*_a, **_k):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise OSError("simulated launch failure")
            return FakeProcess()

        monkeypatch.setattr(runner_module.subprocess, "Popen", flaky_popen)

        record = EnsembleRunner.start_ensemble("sim_source12345", run_count=3)
        member_ids = [m.simulation_id for m in record.members]

        # member 0: still running. member 1: already completed (must be
        # skipped). member 2: never started at all (start_error set, must be
        # skipped -- calling stop_simulation on it would fail since it has no
        # run_state).
        _complete_member(member_ids[1], reddit_actions=1)

        stop_calls = []

        def fake_stop_simulation(cls, simulation_id):
            stop_calls.append(simulation_id)
            state = SimulationRunner.get_run_state(simulation_id)
            state.runner_status = RunnerStatus.STOPPED
            SimulationRunner._save_run_state(state)
            return state

        monkeypatch.setattr(
            SimulationRunner, "stop_simulation", classmethod(fake_stop_simulation)
        )

        results = EnsembleRunner.stop_ensemble(record.ensemble_id)

        assert stop_calls == [member_ids[0]]
        assert results[member_ids[0]]["success"] is True

    def test_stop_missing_ensemble_raises(self):
        with pytest.raises(ValueError, match="不存在"):
            EnsembleRunner.stop_ensemble("ens_doesnotexist")
