# Multigrid-Poisson-Solver
The final project of the class Computational Astrophysics in NTU.

We modified the MG basic code from *CME 342: Parallel Methods in Numerical Analysis at Stanford University (Thomas D. Economon)*
- Change the residual algorithm
- Extend to V-W cycle
- Change the ouput data path
- Change the Poisson problem : 
  *Richardson Cascadic Multigrid Method for 2D Poisson Equation Based on a Fourth Order Compact Scheme, example 5 * @`doc_ref`
- Applied OpenMP (odd-even separation)
- Applied MPI    (half separation)
- Extract and modify the SOR part to be the individual code (one iteration for one count)

Others
- Add `plot_gif.py` file to visualize the exact solution, animation of solution with iteraction, error

## Test 
`mg_omp.cpp`: compile.sh, can set back to original MG or change the thread
`mg_mpi.cpp`: mpi.sh, only used 2 rank
`mg_sor.cpp`: compile.sh
