from chessrl.goals.winvalue import WinValueEstimator


def test_credit_chain_credits_each_distinct_cluster():
    est = WinValueEstimator()
    est.credit_chain([2, 5, 2, 7], won=True)   # distinct: 2,5,7
    assert est.attempts(2) == 1 and est.attempts(5) == 1 and est.attempts(7) == 1
    assert est.attempts(3) == 0


def test_credit_chain_skips_terminal_ids():
    est = WinValueEstimator()
    est.credit_chain([-1, 4, -1], won=False)
    assert est.attempts(4) == 1 and est.attempts(-1) == 0
