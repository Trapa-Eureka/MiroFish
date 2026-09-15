"""
模拟检查点（Checkpoint）

记录一次模拟运行已确认到达的轮次与动作水位线，使得运行该模拟的进程被
杀死/崩溃后，运维者和前端能够准确知道"跑到哪一轮了"，而不是面对一个
含糊的 FAILED 状态无从下手。

重要限制——这不是真正的"从检查点恢复执行"：
OASIS/camel-ai 目前把每个 Agent 的对话记忆（camel 的 ChatHistoryMemory）
完全保存在进程内存中（InMemoryKeyValueStorage），没有任何序列化/反序列化
能力；AgentGraph、OasisEnv 也都不暴露任何 save/load/state-dict 接口
（详见 backend/.venv 中 camel-oasis 包源码，整个包搜不到一处
checkpoint/resume/save_state/load_state）。因此一旦运行该模拟的子进程
被杀死，所有 Agent 的认知状态——它们在对话中已经"记住"、已经形成的
上下文——就永久丢失，无法从 round 31 真正接着跑下去，必须从 round 0
重新开始一次全新的模拟。

这个模块做的是诚实地记录"模拟到底跑到哪儿了"，而不是假装提供一个当前
架构做不到的"续跑"能力。真正的续跑需要 OASIS/camel-ai 自身先支持
Agent 记忆与环境状态的序列化。
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from ..utils.logger import get_logger
from ..utils.state_machine import atomic_write_json

logger = get_logger('mirofish.checkpoint')

CHECKPOINT_FILENAME = "checkpoint.json"

RESUME_LIMITATION_NOTE = (
    "OASIS/camel-ai 中每个 Agent 的对话记忆仅保存在进程内存中，没有任何"
    "序列化能力，AgentGraph/OasisEnv 也不提供 save/load 接口。因此该检查点"
    "只能告诉你模拟运行到了哪一轮，不能被用来真正恢复执行——"
    "必须从 round 0 重新开始一次全新的模拟。"
)


@dataclass
class Checkpoint:
    simulation_id: str
    twitter_round: int = 0
    reddit_round: int = 0
    total_rounds: int = 0
    twitter_action_count: int = 0
    reddit_action_count: int = 0
    runner_status: str = "unknown"
    checkpointed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # 目前恒为 False：详见模块docstring。字段保留是为了让 API/UI 不需要
    # 靠字符串匹配 resume_limitation 就能判断"这份检查点能不能用来续跑"，
    # 以及为将来 OASIS 若支持状态序列化时的扩展留出位置。
    resumable: bool = False
    resume_limitation: str = RESUME_LIMITATION_NOTE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def checkpoint_path(sim_dir: str) -> str:
    return os.path.join(sim_dir, CHECKPOINT_FILENAME)


def save_checkpoint(sim_dir: str, checkpoint: Checkpoint) -> None:
    """原子性地将检查点写入该模拟目录。"""
    atomic_write_json(checkpoint_path(sim_dir), checkpoint.to_dict())


def load_checkpoint(sim_dir: str) -> Optional[Dict[str, Any]]:
    """读取该模拟目录下已持久化的检查点；不存在则返回 None。"""
    path = checkpoint_path(sim_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        logger.exception(f"读取检查点失败: {path}")
        return None


def build_checkpoint_from_state(state) -> Checkpoint:
    """从 SimulationRunState 构建一份检查点快照。"""
    return Checkpoint(
        simulation_id=state.simulation_id,
        twitter_round=state.twitter_current_round,
        reddit_round=state.reddit_current_round,
        total_rounds=state.total_rounds,
        twitter_action_count=state.twitter_actions_count,
        reddit_action_count=state.reddit_actions_count,
        runner_status=state.runner_status.value,
    )
