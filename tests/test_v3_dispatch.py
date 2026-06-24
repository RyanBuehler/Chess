def test_factory_routes_emergent_chained_to_vector_player(tmp_path):
    from chessrl.config.config import RunConfig, GoalConfig, NetworkConfig, EvalConfig, TrainingConfig
    from chessrl.evaluation.daemon import _default_agent_factory
    from chessrl.evaluation.players import VectorGoalMCTSPlayer
    from chessrl.model.network import PolicyValueNet
    from chessrl.training.trainer import Trainer

    ncfg = NetworkConfig(blocks=2, filters=16, goal_cond="vector")
    tr = Trainer(PolicyValueNet(ncfg, goal_conditioned=True), TrainingConfig(device="cpu"),
                 run_dir=str(tmp_path))
    ckpt = tr.save_checkpoint()
    run_cfg = RunConfig(network=ncfg, goal=GoalConfig(goal_mode="emergent_chained"))
    agent = _default_agent_factory("v3@0", ckpt, run_cfg, EvalConfig(agent_simulations=8))
    assert isinstance(agent, VectorGoalMCTSPlayer)


def test_winvalue_chain_credit_from_record_credits_distinct_clusters():
    import numpy as np
    from chessrl.training.parallel_loop import update_winvalue_chain_from_record
    from chessrl.goals.winvalue import WinValueEstimator

    class _Rec:
        # White (plies 0,2) explores clusters 1 then 3; Black (plies 1,3) cluster 5.
        protagonist = None
        outcomes = np.array([1, -1, 1, -1], np.int64)   # side-to-move; White wins
        active_cluster = np.array([1, 5, 3, 5], np.int64)
        explore = np.array([1, 1, 1, 1], np.int64)
        def __len__(self): return 4
        def has_cluster_goals(self): return True

    est = WinValueEstimator()
    update_winvalue_chain_from_record(est, _Rec())
    assert est.attempts(1) == 1 and est.attempts(3) == 1   # White's pursued clusters credited
    assert est.attempts(5) == 1                            # Black's cluster credited once (distinct)
