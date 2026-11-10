"""
S4_setB_lead1_seeds.py  --  Set B: lead 1, all 5 models x 3 timescales x 5 seeds. Resumable.
Finished runs are skipped, so it can be stopped and run again.
"""

if __name__ == "__main__":      # this guard is needed on Windows (parallel CEEMDAN)
    import sys, run10_seed_runs
    sys.argv = ["run10_seed_runs.py", "--set", "B"]
    run10_seed_runs.main()
