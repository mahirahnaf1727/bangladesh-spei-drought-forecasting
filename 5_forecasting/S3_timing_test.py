"""
S3_timing_test.py  --  Timing + determinism test: 5 short runs and 1 repeat.
Includes the one-time CEEMDAN decomposition for SPEI-12 (a few minutes).
"""

if __name__ == "__main__":      # this guard is needed on Windows (parallel CEEMDAN)
    import sys, run10_seed_runs
    sys.argv = ["run10_seed_runs.py", "--set", "timing"]
    run10_seed_runs.main()
