"""
S6_setC_leads2_3.py  --  Set C: leads 2-3, selected model per timescale + XGBoost, 5 seeds. Resumable.
Run after S4 has finished (it refuses to start on an unfinished Set B).
First prints which models the selection rule picked.
"""

if __name__ == "__main__":      # this guard is needed on Windows (parallel CEEMDAN)
    import sys, run10_seed_runs
    sys.argv = ["run10_seed_runs.py", "--set", "C"]
    run10_seed_runs.main()
