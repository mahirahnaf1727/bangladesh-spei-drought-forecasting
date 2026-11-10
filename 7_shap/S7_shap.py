"""
S7_shap.py  --  SHAP for the selected models (lead 1, all 5 seeds). Resumable.
Run after S4 has finished (it refuses to start on an unfinished Set B).
"""

if __name__ == "__main__":      # this guard is needed on Windows (parallel CEEMDAN)
    import run12_shap
    run12_shap.main()
