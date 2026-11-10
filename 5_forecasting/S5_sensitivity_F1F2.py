"""
S5_sensitivity_F1F2.py  --  F1 + F2 sensitivity: strict pipeline, Bi-LSTM SPEI-6 lead 1, 5 seeds.
Run after S4. Prints the verdict of the rule fixed in advance.
"""

if __name__ == "__main__":      # this guard is needed on Windows (parallel CEEMDAN)
    import run11_sensitivity
    run11_sensitivity.main()
