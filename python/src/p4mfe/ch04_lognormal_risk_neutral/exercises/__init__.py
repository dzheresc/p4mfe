"""Chapter 4 exercises (section 4.9), mirroring ``cpp/ch04_lognormal_risk_neutral/exercises``.

One module per group of exercises, with the same names as the C++ headers:

- ``ex01_02_normal_averages``          1-2   sum of independent normals; mean and variance of an average
- ``ex03_06_binomial_calibration``     3-6   binomial and lognormal moments; CRR and p = 1/2 trees
- ``ex07_08_confidence_roulette``      7-8   price confidence intervals; American roulette
- ``ex09_10_itm_delta``                9-10  put in-the-money probabilities; ATM call Delta vs 0.5
- ``ex11_put_integration``             11    put as a finite integral, Midpoint and Simpson vs Black-Scholes
- ``ex12_13_power_payoffs``            12-13 squared call ((S-K)^+)^2; power call (S^alpha - K)^+
- ``ex14_15_normal_model_inverse``     14-15 Bachelier call and put; payoffs 1/S(T) and (1/K - 1/S(T))^+

Full derivations are in the C++ headers and in ``cpp/ch04_lognormal_risk_neutral/exercises/README.md``.
Run ``python -m p4mfe.ch04_lognormal_risk_neutral.exercises`` to print all answers.
"""
