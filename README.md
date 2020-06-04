# Multigrid-Poisson-Solver
The final project of the class Computational Astrophysics in NTU.

We modified the mg basic code from CME 342: Parallel Methods in Numerical Analysis at Stanford University (Thomas D. Economon)
- Change the residual algorithm
- Extend to V-W cycle
- Change the ouput data path
- Change the Poisson problem : 
  doc_ref: Richardson Cascadic Multigrid Method for 2D Poisson Equation Based on a Fourth Order Compact Scheme, example 5 
- Applied OpenMP (odd-ever separation)
- Applied MPI
- Extract and modify the SOR part to be the individual code to impletment (count one iteration for each calculation)

Others
- Add `plot_gif.py` file to visualize the exact solution, solution animation with iteraction, error
