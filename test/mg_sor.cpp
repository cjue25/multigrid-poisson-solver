/*Modified: cjue1325 2020.06.07*/   

/******************************************************************************/
/* File: multigrid_poisson.cpp                                                */
/* ---------------------------                                                */
/* Author: Thomas D. Economon                                                 */
/* Created: 2014.02.17                                                        */
/*                                                                            */
/* A program that solves the poisson equation with a                          */
/* known source term (and solution) on a uniform 2D mesh                      */
/* in serial using multigrid with a number of available smoothers.            */
/* This serial code forms the starting point for a final                      */
/* project in CME 342: Parallel Methods in Numerical Analysis                 */
/* at Stanford University.                                                    */
/******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string.h>
#include <omp.h>

using namespace std;

//! Enumerated types for available smoothers.
enum SMOOTHER_T {
  JACOBI       = 0,
  GAUSS_SEIDEL = 1,
  SOR          = 2
};

//! Definition of the finest grid level for code readability.
const int FINE_MESH = 0;


/******************************************************************************/
/* Set the variable parameters for the simulation here. Note that             */
/* the number of nodes and levels of multigrid can also be set                */
/* at the command line.                                                       */
/******************************************************************************/

//! Number of nodes in the i & j directions
#define NUM_NODES 17

//! Maximum number of multigrid cycles
#define MG_CYCLES 100000

//! Flag for disabling multigrid (Disable MG = 1, Use MG = 0)
#define DISABLE_MG 1

//! Number of smoothing sweeps at each stage of the multigrid
#define NUM_SWEEP 1

//! Choice of iterative smoother (see SMOOTHER_T above)
#define SMOOTHER 2

//! Flag controlling whether to write Tecplot mesh/solution files (Yes=1,No=0)
#define VISUALIZE 0

//! Iteration frequency with which to print to console and write output files
#define FREQUENCY 1

//! Convergence criteria in terms of orders reduction in the L2 norm
#define TOLERANCE 12.0

#define pow_tol -10.0

#define RELAX 1.7

/******************************************************************************/
/* Function prototypes. All necessary functions are contained in this file.   */
/******************************************************************************/

//! Writes some information about the calculation to the console
void write_settings(int n_nodes, int n_levels, int n_mgcycles);

//! Dynamically allocates arrays needed for the life of the solver
int allocate_arrays(double ****phi, double ***phi_exact,
                    double ****f, double ****x, double ****y, double ****aux,
                    int n_nodes);

//! Creates the fine mesh structure
void generate_fine_mesh(double **x, double **y, int n_nodes,
                        bool visualize);

//! Automatically generates the coarse mesh levels recursively
void coarsen_mesh(double ***x, double ***y, int n_nodes,
                  int n_levels, int level, bool visualize);

//! Initializes the values for phi and the forcing term on the fine mesh
void intitialize_solution(double **phi, double **phi_exact, double **f,
                          double **x, double **y, int n_nodes);

//! Recursive function for completing a multigrid V-cycle
void multigrid_cycle(double ***phi, double ***f, double ***aux, int n_nodes,
                     int n_sweeps, int n_levels, int level);

//! Smooth the linear system using the Jacobi method
void smooth_jacobi(double **phi, double **f, double **aux,
                   int n_nodes, int n_sweeps);

//! Smooth the linear system using the Gauss-Seidel method
void smooth_gauss_seidel(double **phi, double **f, double **aux,
                         int n_nodes, int n_sweeps);

//! Smooth the linear system using the SOR method
double smooth_sor(double **phi, double **f,double **residual, int n_nodes, int n_sweeps);

//! Weighted restriction of the residual from a fine mesh to a coarser level
void restrict_weighted(double ***phi, double ***f, double ***aux, int n_nodes,
                       int level);

//! Weighted prolongation of the correction from a coarse mesh to a finer level
void prolongate_weighted(double ***phi, double ***aux, int n_nodes, int level);

//! Residual calculation routine (used by the output and multigrid)
double compute_residual(double **phi, double **f, double **residual,
                        int n_nodes);

//! Routine for writing Tecplot files of the nested mesh levels
void write_mesh(double **x, double **y, int n_nodes, int level);

//! Routine for writing solution information to Tecplot files
void write_output(double **phi, double **phi_exact, double **x, double **y,
                  int n_nodes, int iter, double residual, bool visualize);

//! Deallocates the arrays used throughout the life of the solver
void deallocate_arrays(double ***phi, double **phi_exact, double ***f,
                       double ***x, double ***y, double ***aux, int n_nodes,
                       int n_levels);


/******************************************************************************/
/* Main function driving the high-level solver execution.                     */
/******************************************************************************/

int main(int argc, char* argv[]) {
  
  //! Local variables and settings (defined above) for the poisson problem
  
  bool visualize = VISUALIZE, stop_calc = false;
  int freq = FREQUENCY, n_mgcycles = MG_CYCLES, n_levels, n_sweeps = NUM_SWEEP;
  int i_mgcycles = 0, n_nodes = NUM_NODES, mg_levels = 0;
  double tolerance = TOLERANCE, residual_0, residual,residual_old;
  
  //! Check if we have specified a number of nodes or multigrid levels
  //! on the command line. Other command line inputs could be added here...
  
  if (argc == 2){ n_nodes = atoi(argv[1]); }
  if (argc == 3){ n_nodes = atoi(argv[1]); mg_levels = atoi(argv[2]); }
  
  //! Pointers to arrays that we need throughout the solver
  
  double ***phi, **phi_exact, ***f, ***x, ***y, ***aux;
  
  //! Allocate memory for the grid and solution arrays
  
  n_levels = allocate_arrays(&phi, &phi_exact, &f, &x, &y, &aux, n_nodes);
  
  //! Override the number of multigrid levels if input on the command line
  //! or if the user has disabled multigrid using the input options
  
  if (mg_levels != 0) n_levels = mg_levels;
  if (DISABLE_MG) n_levels = 1;
  
  //! Print some initial information to the console
  
  write_settings(n_nodes, n_levels, n_mgcycles);
  
  //! Create the fine mesh and store the x and y coordinates
  
  generate_fine_mesh(x[FINE_MESH], y[FINE_MESH], n_nodes, visualize);
  
  
  //! Initialize the approximate and exact solutions on all mesh levels
  
  intitialize_solution(phi[FINE_MESH], phi_exact, f[FINE_MESH],
                       x[FINE_MESH], y[FINE_MESH], n_nodes);
  
  residual_0 = smooth_sor(phi[FINE_MESH], f[FINE_MESH], aux[FINE_MESH], n_nodes, n_sweeps);


  //! Write the initial residual and solution files if requested
  write_output(phi[FINE_MESH], phi_exact, x[FINE_MESH],
               y[FINE_MESH], n_nodes, i_mgcycles, residual_0, visualize);



  //! Main solver loop over the prescribed number of multigrid cycles
  
  for (i_mgcycles = 1; i_mgcycles <= n_mgcycles; i_mgcycles++) {
    
       
    //! Check the solution residual and for convergence on the fine mesh
   residual = smooth_sor(phi[FINE_MESH], f[FINE_MESH], aux[FINE_MESH], n_nodes, n_sweeps);
    
   // if (residual < pow(10,pow_tol)) stop_calc = true;
    //if (log10(residual_0)-log10(residual) > tolerance) stop_calc = true;
    if (abs(residual-residual_old) < pow(10,pow_tol)) stop_calc = true;
   residual_old=residual;
   
   // printf("r0:%f - r:%f - tol %f",log10(residual_0),log10(residual),tolerance);
    //! Depending on the cycle number, write a solution file if requested
    
    if ((i_mgcycles%freq == 0) || stop_calc)
      write_output(phi[FINE_MESH], phi_exact, x[FINE_MESH], y[FINE_MESH],
                   n_nodes, i_mgcycles, residual, visualize);
    
    //! Stop the simulation if the convergence criteria is reached
    
    if (stop_calc) break;
    
  }
  
  //! Free all memory used by the solver
  
  deallocate_arrays(phi, phi_exact, f, x, y, aux, n_nodes, n_levels);
  
  //! Print final message to console
  
  if (stop_calc) {
    printf("\nConverged %3.1f orders of magnitude...\n", tolerance);
    printf("#============================================#\n\n");
  } else printf("\n#============================================#\n\n");
  
  return 0;
}

void write_settings(int n_nodes, int n_levels, int n_mgcycles) {
  
  //! Compute the global mesh spacing
  
  double h = 1.0/((double)n_nodes-1.0);
  
  //! Print the settings to the console for our current simulation.
  
  printf("\n");
  printf("#============================================#\n");
  printf("#                                            #\n");
  printf("# 2D, cartesian, multigrid Poisson solver.   #\n");
  printf("#                                            #\n");
  printf("#   Grid Size: %4d  x%4d                   #\n", n_nodes, n_nodes );
  printf("#   Mesh Levels: %6d                      #\n", n_levels);
  printf("#   MG Cycles: %8d                      #\n", n_mgcycles);
  printf("#   Mesh Spacing: %8e               #\n", h);
  printf("#                                            #\n");
  printf("#============================================#\n\n");
  
}

int allocate_arrays(double ****phi, double ***phi_exact, double ****f,
                    double ****x, double ****y, double ****aux, int n_nodes) {
  
  //! Perform allocation using temporary pointers
  
  int i_max, j_max;
  double ***phi_temp, ***aux_temp, ***f_temp, ***x_temp, ***y_temp;
  double **phi_exact_temp;
  
  //! Automatically compute the number of levels for the v-cycle.
  
  bool coarsen = true; int n_levels = 1; int nodes = n_nodes;
  while (coarsen) {
    if (((nodes-1)%2 == 0) && ((nodes-1)/2 + 1 >= 5)) {
      nodes = (nodes-1)/2 + 1;
      n_levels++;
    } else {
      coarsen = false;
    }
  }
  
  //! Allocate arrays first for the coordinates and solutions on all
  //! multigrid levels. This will be the first index.
  
  x_temp   = new double**[n_levels];
  y_temp   = new double**[n_levels];
  phi_temp = new double**[n_levels];
  aux_temp = new double**[n_levels];
  f_temp   = new double**[n_levels];
  
  //! Allocate space for the arrays in the i & j dimensions.
  
  nodes = n_nodes;
  for (int i_level = 0; i_level < n_levels; i_level++) {
    
    //! Allocate space for the i dimension
    
    x_temp[i_level]   = new double*[nodes];
    y_temp[i_level]   = new double*[nodes];
    phi_temp[i_level] = new double*[nodes];
    aux_temp[i_level] = new double*[nodes];
    f_temp[i_level]   = new double*[nodes];
    
    //! Allocate the exact solution only on the fine mesh for visualization
    
    if (i_level == 0) phi_exact_temp = new double*[nodes];
    
    //! Allocate space for the j dimension
    
    for (int i = 0; i < nodes; i++) {
      x_temp[i_level][i]   = new double[nodes];
      y_temp[i_level][i]   = new double[nodes];
      phi_temp[i_level][i] = new double[nodes];
      aux_temp[i_level][i] = new double[nodes];
      f_temp[i_level][i]   = new double[nodes];
      
      //! Allocate the exact solution only on the fine mesh for visualization
      
      if (i_level == 0) phi_exact_temp[i] = new double[nodes];
    }
    
    //! Compute number of nodes on the next coarse level
    
    nodes = (nodes-1)/2 + 1;
    
  }
  
  //! Set the pointers to the correct memory for use outside the function
  
  *x         = x_temp;
  *y         = y_temp;
  *phi       = phi_temp;
  *aux       = aux_temp;
  *f         = f_temp;
  *phi_exact = phi_exact_temp;
  
  //! Return the number of levels for the multigrid
  
  return n_levels;
  
}

void generate_fine_mesh(double **x, double **y, int n_nodes, bool visualize) {
  
  //! Initialize the x & y coordinate arrays. Here, we are
  //! assuming a 2D, cartesian domain over [0,1] X [0,1] with
  //! uniform spacing.
  
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      x[i][j] = (double)i/(double)(n_nodes-1);
      y[i][j] = (double)j/(double)(n_nodes-1);
    }
  }
  
  //! Visualize the mesh to aid in debugging
  
  printf("Created fine cartesian grid (%d x %d)...\n", n_nodes, n_nodes);
  if (visualize) write_mesh(x, y, n_nodes, FINE_MESH);
  
}



void intitialize_solution(double **phi, double **phi_exact, double **f,
                          double **x, double **y, int n_nodes) {
  float fx,fy;
  //! Initialize the solution, exact solution, and source term arrays.
  //! We set an initial condition of zero everywhere except on the
  //! boundaries where we impose the exact solution (Dirichlet condition).
  //! Note that we call this routine only on the fine mesh.
  
  for (int i = 0; i < n_nodes; i++) {
  
    for (int j = 0; j < n_nodes; j++) {
    fx=x[i][j];
    fy=y[i][j];
      if (i == 0 || j == 0 || i == n_nodes-1 || j == n_nodes-1)
        //phi[i][j] = exp(x[i][j])*exp(-2.0*y[i][j]);
        phi[i][j] =log(1+sin(M_PI*x[i][j]*x[i][j]))*(cos(sin(x[i][j]))-1)*sin(M_PI*y[i][j]);
        //phi[i][j]=exp(fx-fy)*fx*(1-fx)*fy*(1-fy);
      else
        phi[i][j] = 0.0;
      //phi_exact[i][j] = exp(x[i][j])*exp(-2.0*y[i][j]);
      phi_exact[i][j]=log(1+sin(M_PI*x[i][j]*x[i][j]))*(cos(sin(x[i][j]))-1)*sin(M_PI*y[i][j]);
      //phi_exact[i][j]=exp(fx-fy)*fx*(1-fx)*fy*(1-fy);
      
      //! Initialize the source term at all nodes. This chosen forcing
      //! for which we have an exact solution to the Poisson equation.
      //! Note that this is fixed during the solution process.
      
      //f[i][j] = -5.0*exp(x[i][j])*exp(-2.0*y[i][j]);
      
      f[i][j] = M_PI*M_PI *sin(M_PI*fy) * log (sin(M_PI*fx*fx)+1) * (cos(sin(fx))-1)\
      - sin(sin(fx))*sin(M_PI*fy)*log(sin(M_PI*fx*fx)+1)*sin(fx)\
      + cos(sin(fx))*sin(M_PI*fy)*log(sin(M_PI*fx*fx)+1)*cos(fx)*cos(fx)\
      - (2*M_PI*sin(M_PI*fy)*cos(M_PI*fx*fx)* (cos(sin(fx))-1)) / (sin(M_PI*fx*fx)+1) \
      + (4*pow(M_PI,2)*fx*fx*sin(M_PI*fy)*pow(cos(M_PI*fx*fx),2)*(cos(sin(fx))-1)) / pow((sin(M_PI*fx*fx)+1),2) \
      + (4*pow(M_PI,2)*fx*fx*sin(M_PI*fy)*sin(M_PI*fx*fx)*(cos(sin(fx))-1)) / (sin(M_PI*fx*fx)+1) \
      + (4*M_PI*sin(sin(fx))*sin(M_PI*fy)*cos(fx)*cos(M_PI*fx*fx)) / (sin(M_PI*fx*fx)+1);
      
      //f[i][j] = - (2*fx*(fy-1)*(fy-2*fx+fx*fy+2)*exp(fx-fy));
      
    }
  }
  
  //! Print some information to the console and exit
  
  printf("Successfully initialized solution...\n");
  
}


double smooth_sor(double **phi, double **f,double **residual, int n_nodes, int n_sweeps) {
   
  double norm = 0.0;
  double h2 = pow(1.0/((double)n_nodes-1.0),2.0);
  double relax = RELAX;
  int mv_node;

for (int iter = 0; iter < n_sweeps; iter++) {
  

	// odd	
      for (int i = 1; i < n_nodes-1; i++) {
      if (i%2!=0){mv_node=1;}
      else if (i%2==0){mv_node=2;}

      for (int j = mv_node; j < n_nodes-1; j+=2) {
        phi[i][j] = (1.0 - relax)*phi[i][j] + relax*(phi[i][j-1] + phi[i-1][j] +
                                                     phi[i+1][j] + phi[i][j+1] +
                                                     h2*f[i][j])/4.0;

      }
    }

      for (int i = 1; i < n_nodes-1; i++) {
      if (i%2!=0){mv_node=2;}
      else if (i%2==0){mv_node=1;}

      for (int j = mv_node; j < n_nodes-1; j+=2) {
        phi[i][j] = (1.0 - relax)*phi[i][j] + relax*(phi[i][j-1] + phi[i-1][j] +
                                                     phi[i+1][j] + phi[i][j+1] +
                                                     h2*f[i][j])/4.0;

      }
    }

}//iter loop

  for (int i=1;i<(n_nodes-1);i++){
      for (int j=1;j<(n_nodes-1);j++){
      residual[i][j] = f[i][j] + (phi[i][j-1] + phi[i-1][j] +
                                phi[i+1][j] + phi[i][j+1] - 4.0*phi[i][j])/h2;
       norm += residual[i][j]*residual[i][j];                         
	//norm+=abs(residual[i][j]/phi[i][j])/pow(n_nodes-1,2);		    
      }
      }
      norm = sqrt(norm);
      return norm;
}//function



double compute_residual(double **phi, double **f, double **residual,
                        int n_nodes) {
  
  //! Compute the residual at each node and then take an L2 norm.
  //! Form R(phi) = 0 where R(phi) is the discrete laplacian + the source.
  
  double norm;
  double h2 = pow(1.0/((double)n_nodes-1.0),2.0);
  
  norm = 0.0;
  for (int i = 1; i < n_nodes-1; i++) {
    for (int j = 1; j < n_nodes-1; j++) {
        residual[i][j] = f[i][j] + (phi[i][j-1] + phi[i-1][j] +
                                phi[i+1][j] + phi[i][j+1] - 4.0*phi[i][j])/h2;
      //residual[i][j] = (h2*f[i][j] + (phi[i][j-1] + phi[i-1][j] +
      //                            phi[i+1][j] + phi[i][j+1] - 4.0*phi[i][j]))/phi[i][j]/pow(n_nodes-1.0,2.0);
                                
                                  
      norm += residual[i][j]*residual[i][j];
    }
  }
  norm = sqrt(norm);
  
  //! Return the value of the total residual magnitude, and note that
  //! the value of the residual at each node is stored in aux.
  
  return norm;
}

void write_mesh(double **x, double **y, int n_nodes, int level) {
  
  //! Write tecplot files of the mesh levels for visualization
  
  char cstr[200], buffer[50];
  strcpy(cstr, "./mesh/mesh");
  sprintf(buffer, "_%d.dat", level);
  strcat(cstr,buffer);
  ofstream sol_file;
  sol_file.precision(15);
  sol_file.open(cstr, ios::out);
  sol_file << "TITLE = \"Visualization of Laplacian smoothing\"" << endl;
  sol_file << "VARIABLES = \"x\", \"y\"" << endl;
  sol_file << "ZONE STRANDID=" << level+1 << ", SOLUTIONTIME=" << level+1;
  sol_file << ", DATAPACKING=BLOCK I=" << n_nodes << ", J=" << n_nodes << endl;
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      sol_file << x[i][j] << " ";
    }
    sol_file << endl;
  }
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      sol_file << y[j][j]  << " ";
    }
    sol_file << endl;
  }
  sol_file.close();
  
  // Set status flag to success and exit
  
  printf("Successfully wrote mesh file for level = %d...\n", level);
  
}

void write_output(double **phi, double **phi_exact, double **x, double **y,
                  int n_nodes, int iter, double residual, bool visualize) {
  
  //! Write tecplot files at each iteration for visualization. The below
  //! is the Tecplot ascii format. Check if files were requested first.
  
  double error = 0.0, error_L2 = 0.0, center_sol = 0.0;
  int center = n_nodes/2+1;
  char cstr[200], buffer[50];
  
  if (visualize) {
    strcpy(cstr, "./sol/solution");
    sprintf(buffer, "_%d.dat", iter+1);
    strcat(cstr,buffer);
    ofstream sol_file;
    sol_file.precision(15);
    sol_file.open(cstr, ios::out);
    sol_file << "TITLE = \"Visualization of Laplacian smoothing\"" << endl;
    sol_file << "VARIABLES = \"x\", \"y\", \"phi\", \"phi_exact\", \"error\"";
    sol_file << endl;
    sol_file << "ZONE STRANDID=" << iter+1 << ", SOLUTIONTIME=" << iter+1;
    sol_file << ", DATAPACKING=BLOCK I=" << n_nodes << ", J=" << n_nodes;
    sol_file << endl;
    for (int i = 0; i < n_nodes; i++) {
      for (int j = 0; j < n_nodes; j++) {
        sol_file << x[i][j] << " ";
      }
      sol_file << endl;
    }
    for (int i = 0; i < n_nodes; i++) {
      for (int j = 0; j < n_nodes; j++) {
        sol_file << y[j][j]  << " ";
      }
      sol_file << endl;
    }
    for (int i = 0; i < n_nodes; i++) {
      for (int j = 0; j < n_nodes; j++) {
        sol_file << phi[i][j]  << " ";
        if (i == center && j == center) center_sol = phi[i][j];
      }
      sol_file << endl;
    }
    for (int i = 0; i < n_nodes; i++) {
      for (int j = 0; j < n_nodes; j++) {
        sol_file << phi_exact[i][j]  << " ";
      }
      sol_file << endl;
    }
    for (int i = 0; i < n_nodes; i++) {
      for (int j = 0; j < n_nodes; j++) {
        error = phi_exact[i][j]-phi[i][j];
        sol_file << fabs(error) << " ";
      }
      sol_file << endl;
    }
    sol_file.close();
  }
  
  //! Compute the L2 norm of the error for reporting
  
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      error = phi_exact[i][j]-phi[i][j];
      error_L2 += error*error;
    }
  }
  error_L2 = sqrt(error_L2);
  
  //! Print the log10 of the residual & error to the console
  
  if (iter == 0) { printf("\n");
    printf("  Iteration         Residual        L2 Error \n");
    printf("  ------------------------------------------ \n");
    
  }
  printf("    %6d     %13e   %13e \n", iter, residual, error_L2);
  
}

void deallocate_arrays(double ***phi, double **phi_exact, double ***f,
                       double ***x, double ***y, double ***aux, int n_nodes,
                       int n_levels) {
  
  //! Deallocation of all dynamic memory in the program.
  
  int nodes = n_nodes;
  for (int i_level = 0; i_level < n_levels; i_level++) {
    
    //! Delete j dimension
    
    for (int i = 0; i < nodes; i++) {
      delete [] x[i_level][i];
      delete [] y[i_level][i];
      delete [] phi[i_level][i];
      delete [] aux[i_level][i];
      delete [] f[i_level][i];
      if (i_level == 0) delete [] phi_exact[i];
    }
    
    //! Delete i dimension
    
    delete [] x[i_level];
    delete [] y[i_level];
    delete [] phi[i_level];
    delete [] aux[i_level];
    delete [] f[i_level];
    if (i_level == 0) delete [] phi_exact;
    
    //! Compute number of nodes on the next coarse level
    
    nodes = (nodes-1)/2 + 1;
    
  }
  
  //! Delete levels dimension
  
  delete [] x;
  delete [] y;
  delete [] phi;
  delete [] aux;
  delete [] f;
  
}
