"""
集成模拟（Ensemble Simulation）框架

一次模拟运行不应该被当成一个概率估计——LLM 采样带来的随机性可能让同一份
配置、同一批 Agent Profile 在不同运行之间得出明显不同的结果。这个模块
支持把"同一个场景"独立重复运行 N 次，并把这 N 次运行的结果聚合成一份
分布报告（中位数/方差/置信区间/结果频率/敏感度），取代"模拟预测了 X"这种
单次叙事式结论，改为"在这些假设下，100 次运行里有 64 次出现了 X"。

设计上的关键约束（继承自本项目已有的架构限制，而不是这个模块引入的新限制）：

1. 不做"真正的续跑"（见 simulation_checkpoint.py 的说明）：OASIS/camel-ai
   不支持序列化 Agent 记忆，所以集成的每个成员（member）都必须是一次完全
   独立、从零开始的全新模拟，而不是同一次运行的多个"分支"。

2. 同一场景 = 同一份已准备好的人设（profiles）与配置（simulation_config.json）。
   集成不会为每个成员重新调用 LLM 生成人设/配置——那样引入的差异来自
   "输入不同"，而不是这个功能想要测量的"给定完全相同的输入，LLM 在模拟
   执行过程中的采样随机性能造成多大差异"。成员之间的唯一差异来源就是
   模拟执行过程本身的 LLM 采样，这也是可复现性清单里 `deterministic: false`
   的那部分随机性（模拟准备阶段的非 LLM 随机性，例如人设兜底默认值的
   random_seed，则是完全相同的，因为人设文件是原样拷贝，不是重新生成）。

3. 每个成员本质上就是一次普通的、独立的 SimulationRunner.start_simulation()
   调用，运行在自己的子进程里，写入自己的 run_state.json/checkpoint.json。
   集成本身不维护一个需要在进程重启后恢复的后台监控线程——集成的状态
   （运行中/已完成/失败、聚合统计）在每次被查询时，直接从每个成员当前
   已持久化的 run_state.json 现场推导，而不是缓存一份可能过期的"集成状态"。
   这个设计直接复用了 TASK 7 中 GET .../checkpoint 的教训：与其维护两份
   可能不同步的状态（成员状态 + 集成状态快照），不如让集成状态永远是
   成员状态的一个纯函数。代价是每次查询一个仍在运行的大集成时都要重新
   读取全部成员的状态文件；在集成规模（数十个成员）下可以接受。

4. 因此，集成本身只有一份在创建时写入、之后不再修改的记录文件
   （ensemble.json：source_simulation_id、成员列表、启动期失败信息等），
   不需要状态机/CAS——它不是一个会被并发更新的活跃状态，聚合结果永远是
   现算的。
"""

import json
import os
import shutil
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger
from ..utils.id_validation import validate_ensemble_id, safe_join
from ..utils.state_machine import atomic_write_json
from .simulation_manager import SimulationManager, SimulationState, SimulationStatus
from .simulation_runner import SimulationRunner, RunnerStatus

logger = get_logger('mirofish.ensemble')

ENSEMBLE_FILENAME = "ensemble.json"

# 单次集成允许的成员数量上限。每个成员都是一次完整的、由 LLM 驱动的模拟
# 子进程，成本与运行中的模拟运行时间成正比；这里做一个保守的硬上限，
# 避免一次请求意外地把服务器拖入几十上百个并发子进程。
MAX_ENSEMBLE_RUN_COUNT = 50
MIN_ENSEMBLE_RUN_COUNT = 2

# 已到达终态、不会再变化的 runner_status 取值。
_TERMINAL_RUNNER_STATUSES = {
    RunnerStatus.COMPLETED.value,
    RunnerStatus.FAILED.value,
    RunnerStatus.STOPPED.value,
}


@dataclass
class EnsembleMemberRecord:
    """一个集成成员在创建时的静态记录（不随运行变化）。"""
    simulation_id: str
    index: int
    # 若非 None，说明这个成员在 start_simulation 调用本身就失败了（例如
    # 拷贝人设/配置文件失败），从未真正进入 SimulationRunner 的状态机——
    # 这种情况下 run_state.json 可能根本不存在，必须单独记录，不能指望
    # 从成员的 run_state 里读到这个错误。
    start_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnsembleMemberRecord":
        return cls(
            simulation_id=data["simulation_id"],
            index=data["index"],
            start_error=data.get("start_error"),
        )


@dataclass
class EnsembleRecord:
    """一次集成的不可变创建记录，创建后只读，聚合状态永远现算。"""
    ensemble_id: str
    source_simulation_id: str
    project_id: str
    graph_id: str
    platform: str
    max_rounds: Optional[int]
    run_count: int
    members: List[EnsembleMemberRecord]
    created_at: str
    owner_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ensemble_id": self.ensemble_id,
            "source_simulation_id": self.source_simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "platform": self.platform,
            "max_rounds": self.max_rounds,
            "run_count": self.run_count,
            "members": [m.to_dict() for m in self.members],
            "created_at": self.created_at,
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnsembleRecord":
        return cls(
            ensemble_id=data["ensemble_id"],
            source_simulation_id=data["source_simulation_id"],
            project_id=data.get("project_id", ""),
            graph_id=data.get("graph_id", ""),
            platform=data.get("platform", "parallel"),
            max_rounds=data.get("max_rounds"),
            run_count=data.get("run_count", 0),
            members=[EnsembleMemberRecord.from_dict(m) for m in data.get("members", [])],
            created_at=data.get("created_at", datetime.now().isoformat()),
            owner_id=data.get("owner_id"),
        )


class EnsembleRunner:
    """
    集成模拟的编排入口。

    与 SimulationRunner/SimulationManager 保持相同的“目录即存储”风格：
    每个集成在 uploads/ensembles/<ensemble_id>/ensemble.json 下保存一份
    创建时的静态记录；每个成员本身就是 uploads/simulations/<member_id> 下
    一个普通的、完全由 SimulationManager/SimulationRunner 管理的模拟。
    """

    ENSEMBLE_DATA_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../uploads/ensembles'
    )

    @classmethod
    def _get_ensemble_dir(cls, ensemble_id: str) -> str:
        validate_ensemble_id(ensemble_id)
        return safe_join(cls.ENSEMBLE_DATA_DIR, ensemble_id)

    @classmethod
    def _ensemble_path(cls, ensemble_id: str) -> str:
        return os.path.join(cls._get_ensemble_dir(ensemble_id), ENSEMBLE_FILENAME)

    @classmethod
    def get_ensemble_record(cls, ensemble_id: str) -> Optional[EnsembleRecord]:
        path = cls._ensemble_path(ensemble_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return EnsembleRecord.from_dict(json.load(f))
        except Exception:
            logger.exception(f"读取集成记录失败: ensemble_id={ensemble_id}")
            return None

    @classmethod
    def list_ensembles(cls, project_id: Optional[str] = None) -> List[EnsembleRecord]:
        """列出所有集成记录（可选按 project_id 过滤）。"""
        records: List[EnsembleRecord] = []
        if not os.path.exists(cls.ENSEMBLE_DATA_DIR):
            return records
        for ensemble_id in os.listdir(cls.ENSEMBLE_DATA_DIR):
            if ensemble_id.startswith('.'):
                continue
            try:
                record = cls.get_ensemble_record(ensemble_id)
            except Exception:
                continue
            if record is None:
                continue
            if project_id is None or record.project_id == project_id:
                records.append(record)
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    # ────────────────────────── 启动 ──────────────────────────

    @classmethod
    def start_ensemble(
        cls,
        source_simulation_id: str,
        run_count: int,
        platform: Optional[str] = None,
        max_rounds: Optional[int] = None,
    ) -> EnsembleRecord:
        """
        以某个已准备好的模拟为"场景来源"，独立启动 run_count 次重复运行。

        Args:
            source_simulation_id: 已完成 /prepare 的模拟ID，提供人设与配置。
            run_count: 成员数量（独立重复运行次数）。
            platform: twitter/reddit/parallel，默认沿用来源模拟的启用平台。
            max_rounds: 可选，透传给每个成员的 start_simulation。

        Returns:
            EnsembleRecord（创建时的静态记录；实时状态请用 get_ensemble_summary）。
        """
        if not isinstance(run_count, int) or isinstance(run_count, bool):
            raise ValueError("run_count 必须是整数")
        if run_count < MIN_ENSEMBLE_RUN_COUNT or run_count > MAX_ENSEMBLE_RUN_COUNT:
            raise ValueError(
                f"run_count 必须在 {MIN_ENSEMBLE_RUN_COUNT} 到 "
                f"{MAX_ENSEMBLE_RUN_COUNT} 之间"
            )

        manager = SimulationManager()
        source_state = manager.get_simulation(source_simulation_id)
        if source_state is None:
            raise ValueError(f"来源模拟不存在: {source_simulation_id}")
        if not (source_state.config_generated and source_state.profiles_generated):
            raise ValueError(
                f"来源模拟尚未完成准备（需要先调用 /prepare）: {source_simulation_id}"
            )

        source_dir = manager._get_simulation_dir(source_simulation_id)
        source_config_path = os.path.join(source_dir, "simulation_config.json")
        if not os.path.exists(source_config_path):
            raise ValueError(f"来源模拟缺少 simulation_config.json: {source_simulation_id}")

        resolved_platform = platform or source_state.get_default_platform()
        if resolved_platform not in ("twitter", "reddit", "parallel"):
            raise ValueError(f"不支持的平台: {resolved_platform}")
        # 请求的 platform 必须是来源模拟真正准备过人设的平台，否则拷贝到
        # 成员目录的输入文件里会缺少该平台的 profiles，对应的运行脚本会在
        # 找不到 profile 文件时直接以退出码 0 正常结束——这会被误判为一次
        # "正常完成但没有任何动作"的运行，污染聚合统计而不是暴露成配置错误。
        if resolved_platform in ("twitter", "parallel") and not source_state.enable_twitter:
            raise ValueError(
                f"来源模拟未启用 Twitter，无法以 platform={resolved_platform} 启动集成: "
                f"{source_simulation_id}"
            )
        if resolved_platform in ("reddit", "parallel") and not source_state.enable_reddit:
            raise ValueError(
                f"来源模拟未启用 Reddit，无法以 platform={resolved_platform} 启动集成: "
                f"{source_simulation_id}"
            )

        ensemble_id = f"ens_{uuid.uuid4().hex[:12]}"
        ensemble_dir = cls._get_ensemble_dir(ensemble_id)
        os.makedirs(ensemble_dir, exist_ok=True)

        # 需要拷贝到每个成员目录的静态输入文件：配置 + 人设（只拷贝该场景
        # 实际启用的平台对应的人设文件）。
        files_to_copy = ["simulation_config.json"]
        if source_state.enable_reddit:
            files_to_copy.append("reddit_profiles.json")
        if source_state.enable_twitter:
            files_to_copy.append("twitter_profiles.csv")

        # 先以“成员列表已确定、尚未启动”的状态落盘一份临时记录：member_id
        # 是从 ensemble_id + index 确定性推导出来的，不需要等任何一个
        # start_simulation 调用返回就能确定完整列表。这样即使worker在下面
        # 的启动循环中途崩溃，或者循环结束后的最终写入本身失败，集成也已经
        # 是可被 /list、summary、/stop 发现的——不会出现"若干模拟子进程已经
        # 在跑，但没有任何集成记录知道它们存在"的情况。
        provisional_members = [
            EnsembleMemberRecord(simulation_id=f"{ensemble_id}_m{index}", index=index)
            for index in range(run_count)
        ]
        provisional_record = EnsembleRecord(
            ensemble_id=ensemble_id,
            source_simulation_id=source_simulation_id,
            project_id=source_state.project_id,
            graph_id=source_state.graph_id,
            platform=resolved_platform,
            max_rounds=max_rounds,
            run_count=run_count,
            members=provisional_members,
            created_at=datetime.now().isoformat(),
            owner_id=source_state.owner_id,
        )
        atomic_write_json(cls._ensemble_path(ensemble_id), provisional_record.to_dict())

        members: List[EnsembleMemberRecord] = []
        for index in range(run_count):
            member_id = f"{ensemble_id}_m{index}"
            start_error: Optional[str] = None
            try:
                member_dir = manager._get_simulation_dir(member_id)
                for filename in files_to_copy:
                    src_path = os.path.join(source_dir, filename)
                    if not os.path.exists(src_path):
                        # enable_reddit/enable_twitter 为 True 但对应文件缺失，
                        # 说明来源模拟本身处于不一致状态——让这个成员启动
                        # 失败并记录原因，而不是启动一个缺人设的模拟。
                        raise FileNotFoundError(f"来源模拟缺少 {filename}")
                    shutil.copyfile(src_path, os.path.join(member_dir, filename))

                member_state = SimulationState(
                    simulation_id=member_id,
                    project_id=source_state.project_id,
                    graph_id=source_state.graph_id,
                    enable_twitter=source_state.enable_twitter,
                    enable_reddit=source_state.enable_reddit,
                    status=SimulationStatus.READY,
                    entities_count=source_state.entities_count,
                    profiles_count=source_state.profiles_count,
                    entity_types=list(source_state.entity_types),
                    profiles_generated=True,
                    config_generated=True,
                    config_reasoning=source_state.config_reasoning,
                    owner_id=source_state.owner_id,
                    # 人设文件是原样拷贝的，不是重新生成的，所以沿用同一个
                    # random_seed 是诚实的——它就是这份被拷贝的人设实际使用
                    # 过的种子，而不是一个凭空分配、从未真正生效过的新种子。
                    random_seed=source_state.random_seed,
                )
                manager._save_simulation_state(member_state)

                # graph_id=None：集成成员不写入 Zep 图谱记忆——N 个成员并发
                # 把各自的模拟活动写回同一个图谱会相互践踏且没有明确语义
                # （"图谱应该记住哪一次运行？"）。这是一个有意识的范围限制，
                # 而不是遗漏；需要图谱记忆更新的场景应该用单次模拟运行。
                SimulationRunner.start_simulation(
                    simulation_id=member_id,
                    platform=resolved_platform,
                    max_rounds=max_rounds,
                    enable_graph_memory_update=False,
                    graph_id=None,
                    # 集成成员没有人工操作者去调用 interview/close：不传
                    # no_wait 的话，子进程会在跑完所有轮次后停留在等待
                    # 命令的状态，SimulationRunner 永远看不到进程退出，
                    # 这个成员也就永远不会变成 COMPLETED，集成也就永远
                    # 凑不齐聚合所需的终态成员。
                    no_wait=True,
                )
            except Exception as error:
                logger.exception(
                    f"集成成员启动失败: ensemble_id={ensemble_id}, member_id={member_id}"
                )
                start_error = str(error)

            members.append(EnsembleMemberRecord(
                simulation_id=member_id,
                index=index,
                start_error=start_error,
            ))

        record = EnsembleRecord(
            ensemble_id=ensemble_id,
            source_simulation_id=source_simulation_id,
            project_id=source_state.project_id,
            graph_id=source_state.graph_id,
            platform=resolved_platform,
            max_rounds=max_rounds,
            run_count=run_count,
            members=members,
            created_at=provisional_record.created_at,
            owner_id=source_state.owner_id,
        )
        try:
            atomic_write_json(cls._ensemble_path(ensemble_id), record.to_dict())
        except Exception:
            # 最终这份带有各成员 start_error 详情的记录写入失败了，但成员
            # 已经在跑（或已经失败）——上面的临时记录已经让这个集成对
            # /list、summary、/stop 可见，只是 start_error 字段会暂时停留
            # 在"未知"（run_state 不存在时，summary 会把它当作 starting
            # 处理）。这里选择不回滚/终止已经启动的成员：它们是真实、可能
            # 已产生 LLM 调用成本的进程，"记录写入失败"不应该反过来杀掉
            # 已经在花钱运行的模拟。
            logger.exception(
                f"写入集成最终记录失败（成员已启动，临时记录仍然可见）: "
                f"ensemble_id={ensemble_id}"
            )

        if all(m.start_error is not None for m in members):
            logger.error(
                f"集成的所有成员均启动失败: ensemble_id={ensemble_id}"
            )

        return record

    # ────────────────────────── 停止 ──────────────────────────

    @classmethod
    def stop_ensemble(cls, ensemble_id: str) -> Dict[str, Any]:
        """尽力停止集成中所有仍处于非终态的成员，返回逐成员的停止结果。"""
        record = cls.get_ensemble_record(ensemble_id)
        if record is None:
            raise ValueError(f"集成不存在: {ensemble_id}")

        results: Dict[str, Any] = {}
        for member in record.members:
            if member.start_error is not None:
                continue
            run_state = SimulationRunner.get_run_state(member.simulation_id)
            if run_state is None or run_state.runner_status.value in _TERMINAL_RUNNER_STATUSES:
                continue
            try:
                stopped = SimulationRunner.stop_simulation(member.simulation_id)
                results[member.simulation_id] = {"success": True, "runner_status": stopped.runner_status.value}
            except Exception as error:
                logger.exception(
                    f"停止集成成员失败: ensemble_id={ensemble_id}, member_id={member.simulation_id}"
                )
                results[member.simulation_id] = {"success": False, "error": str(error)}
        return results

    # ────────────────────────── 聚合 ──────────────────────────

    @classmethod
    def get_ensemble_summary(cls, ensemble_id: str) -> Dict[str, Any]:
        """
        现算集成的整体状态与（若已全部到达终态）聚合统计。

        绝不信任任何缓存的"集成状态"字段——每个成员的 runner_status 都是
        现场从其 run_state.json 读出的，聚合统计也是每次调用都重新计算。
        这个函数是幂等的纯读取，可以被随时高频调用而不会破坏任何状态。
        """
        record = cls.get_ensemble_record(ensemble_id)
        if record is None:
            raise ValueError(f"集成不存在: {ensemble_id}")

        member_infos = []
        completed_member_ids = []
        non_terminal_count = 0
        completed_count = 0
        failed_count = 0
        stopped_count = 0

        for member in record.members:
            if member.start_error is not None:
                status = RunnerStatus.FAILED.value
                info = {
                    "simulation_id": member.simulation_id,
                    "index": member.index,
                    "runner_status": status,
                    "error": member.start_error,
                }
                failed_count += 1
            else:
                run_state = SimulationRunner.get_run_state(member.simulation_id)
                if run_state is None:
                    # start_simulation 成功发起但还没有任何 run_state 落盘
                    # 是不应该发生的（start_simulation 在返回前必然已经
                    # 保存过至少一次 run_state），但防御性地把它当作仍在
                    # 启动中处理，而不是让整个聚合请求抛异常。
                    status = RunnerStatus.STARTING.value
                    info = {
                        "simulation_id": member.simulation_id,
                        "index": member.index,
                        "runner_status": status,
                        "error": None,
                    }
                    non_terminal_count += 1
                else:
                    status = run_state.runner_status.value
                    info = {
                        "simulation_id": member.simulation_id,
                        "index": member.index,
                        "runner_status": status,
                        "error": run_state.error,
                        "current_round": run_state.current_round,
                        "total_rounds": run_state.total_rounds,
                        "twitter_actions_count": run_state.twitter_actions_count,
                        "reddit_actions_count": run_state.reddit_actions_count,
                        "total_actions_count": (
                            run_state.twitter_actions_count + run_state.reddit_actions_count
                        ),
                    }
                    if status == RunnerStatus.COMPLETED.value:
                        completed_count += 1
                        completed_member_ids.append(member.simulation_id)
                    elif status == RunnerStatus.FAILED.value:
                        failed_count += 1
                    elif status == RunnerStatus.STOPPED.value:
                        stopped_count += 1
                    else:
                        non_terminal_count += 1

            member_infos.append(info)

        is_complete = non_terminal_count == 0
        if not is_complete:
            ensemble_status = "running"
        elif completed_count == 0:
            ensemble_status = "failed"
        else:
            ensemble_status = "completed"

        aggregate = None
        if is_complete and completed_count > 0:
            aggregate = cls._compute_aggregate(record, completed_member_ids)

        return {
            "ensemble_id": record.ensemble_id,
            "source_simulation_id": record.source_simulation_id,
            "platform": record.platform,
            "max_rounds": record.max_rounds,
            "run_count": record.run_count,
            "created_at": record.created_at,
            "status": ensemble_status,
            "member_counts": {
                "total": len(record.members),
                "completed": completed_count,
                "failed": failed_count,
                "stopped": stopped_count,
                "running": non_terminal_count,
            },
            "members": member_infos,
            "aggregate": aggregate,
        }

    @classmethod
    def _compute_aggregate(
        cls, record: EnsembleRecord, completed_member_ids: List[str]
    ) -> Dict[str, Any]:
        manager = SimulationManager()
        numeric_samples: Dict[str, List[float]] = {
            "total_actions_count": [],
            "twitter_actions_count": [],
            "reddit_actions_count": [],
            "rounds_reached": [],
        }
        action_type_run_occurrences: Dict[str, int] = {}
        action_type_total_counts: Dict[str, int] = {}

        for member_id in completed_member_ids:
            run_state = SimulationRunner.get_run_state(member_id)
            if run_state is None:
                continue
            numeric_samples["total_actions_count"].append(
                run_state.twitter_actions_count + run_state.reddit_actions_count
            )
            numeric_samples["twitter_actions_count"].append(run_state.twitter_actions_count)
            numeric_samples["reddit_actions_count"].append(run_state.reddit_actions_count)
            numeric_samples["rounds_reached"].append(run_state.current_round)

            member_dir = manager._get_simulation_dir(member_id)
            action_types_seen_this_run = cls._tally_action_types(
                member_dir, action_type_total_counts
            )
            for action_type in action_types_seen_this_run:
                action_type_run_occurrences[action_type] = (
                    action_type_run_occurrences.get(action_type, 0) + 1
                )

        run_count = len(completed_member_ids)
        metrics = {
            name: cls._summarize_numeric(values)
            for name, values in numeric_samples.items()
        }

        outcome_frequency = {
            action_type: {
                "run_occurrence_count": action_type_run_occurrences.get(action_type, 0),
                "run_occurrence_rate": round(
                    action_type_run_occurrences.get(action_type, 0) / run_count, 4
                ),
                "total_count": total,
            }
            for action_type, total in sorted(action_type_total_counts.items())
        }

        return {
            "completed_run_count": run_count,
            "metrics": metrics,
            "outcome_frequency": outcome_frequency,
        }

    @staticmethod
    def _tally_action_types(
        member_dir: str, running_totals: Dict[str, int]
    ) -> set:
        """
        统计一个成员目录下 twitter/reddit 的 actions.jsonl 里出现过的
        action_type，把每种类型的总出现次数累加进 running_totals，并
        返回这次运行里"至少出现过一次"的 action_type 集合（用于统计
        跨运行的出现频率）。
        """
        seen_this_run = set()
        for platform_dir in ("twitter", "reddit"):
            log_path = os.path.join(member_dir, platform_dir, "actions.jsonl")
            if not os.path.exists(log_path):
                continue
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if "event_type" in entry:
                            continue
                        action_type = entry.get("action_type")
                        if not action_type:
                            continue
                        running_totals[action_type] = running_totals.get(action_type, 0) + 1
                        seen_this_run.add(action_type)
            except Exception:
                logger.exception(f"读取动作日志失败（聚合统计）: {log_path}")
        return seen_this_run

    @staticmethod
    def _summarize_numeric(values: List[float]) -> Dict[str, Any]:
        """
        对一组样本值计算描述性统计与一个基于正态近似的 95% 置信区间。

        注意：置信区间用的是正态近似（mean ± 1.96 * stderr），不是针对
        小样本的严格方法（如 t 分布）——集成的典型规模是几到几十次运行，
        这个区间应该被理解为"粗略的不确定性范围"，而不是统计学意义上
        严谨的区间估计。敏感度用留一法（jackknife）：依次去掉每一个样本，
        看均值最多偏移多少，衡量结果对单次运行的依赖程度；样本数小于3时
        没有意义，返回 None。
        """
        n = len(values)
        if n == 0:
            return {
                "count": 0, "mean": None, "median": None, "stdev": None,
                "variance": None, "min": None, "max": None,
                "confidence_interval_95": None, "sensitivity": None,
            }

        mean = statistics.mean(values)
        median = statistics.median(values)
        stdev = statistics.stdev(values) if n >= 2 else 0.0
        variance = stdev ** 2

        confidence_interval_95 = None
        if n >= 2:
            stderr = stdev / (n ** 0.5)
            margin = 1.96 * stderr
            confidence_interval_95 = [mean - margin, mean + margin]

        sensitivity = None
        if n >= 3:
            loo_means = []
            for i in range(n):
                remaining = values[:i] + values[i + 1:]
                loo_means.append(statistics.mean(remaining))
            sensitivity = max(abs(loo_mean - mean) for loo_mean in loo_means)

        return {
            "count": n,
            "mean": mean,
            "median": median,
            "stdev": stdev,
            "variance": variance,
            "min": min(values),
            "max": max(values),
            "confidence_interval_95": confidence_interval_95,
            "sensitivity": sensitivity,
        }
