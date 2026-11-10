"""
S2_spei_refit.py  --  Refit SPEI on 1980-2018 and compare (fix F1). Needed by S5.
Expect "[self-check] PASSED". Under a minute.
"""

if __name__ == "__main__":      # this guard is needed on Windows (parallel CEEMDAN)
    import p1_spei_refit
    p1_spei_refit.main()
