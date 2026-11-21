"""Goal-conditioned recurrent crew (farmer + hands), independent of manager.

WorkerGrowActor/grow_step_reward remain legacy experiments for compatibility;
the new crew runner does not call their reward or independent-unit policies.
"""
from worker_grow.contracts import ProductionTask, TaskBook, TaskStatus
from worker_grow.crew import FeudalCrew, CrewState, propose_crew_turn, attach_manager_market
from worker_grow.issues import Issue, IssueBook
from worker_grow.manager import ManagerActor, propose_manager_contracts, propose_manager_market
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.encoding import OwnFarmObservation, encode_own_farm
from worker_grow.recipes import (
    ANIMAL_UNIT_ACTIONS,
    CROP_UNIT_ACTIONS,
    GROW_OPS,
    required_unit_actions,
)
from worker_grow.reward import REWARD_SCALE, grow_step_reward
from worker_grow.policy import WorkerGrowActor, sample_worker

__all__ = (
    "ProductionTask", "TaskBook", "TaskStatus", "FeudalCrew", "CrewState",
    "propose_crew_turn", "attach_manager_market", "CrewDeliveryTracker",
    "Issue", "IssueBook", "ManagerActor",
    "propose_manager_contracts", "propose_manager_market",
    "ANIMAL_UNIT_ACTIONS",
    "CROP_UNIT_ACTIONS",
    "GROW_OPS",
    "OwnFarmObservation",
    "REWARD_SCALE",
    "WorkerGrowActor",
    "encode_own_farm",
    "grow_step_reward",
    "required_unit_actions",
    "sample_worker",
)
