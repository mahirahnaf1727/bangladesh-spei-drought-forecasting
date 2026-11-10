"""
S1_build_arrays.py  --  Build the feature matrices and model arrays (main pipeline).
Expect "All checks passed." twice. About 2 minutes.
"""

if __name__ == "__main__":      # this guard is needed on Windows (parallel CEEMDAN)
    import run1_build_matrices, run2_data_prep
    run1_build_matrices.main()
    run2_data_prep.main()
