p4mfe_add_module(ch08_multivariable_barriers
  DEPS p4mfe::common p4mfe::ch01_calculus_options p4mfe::ch02_integration_bonds
       p4mfe::ch03_probability_black_scholes p4mfe::ch04_lognormal_risk_neutral
       p4mfe::ch05_newton_bootstrapping p4mfe::ch07_finite_differences_bs_pde)
add_subdirectory(exercises)
