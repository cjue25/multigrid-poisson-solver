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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string.h>
#include <mpi.h>

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
#define NUM_NODES 256

//! Maximum number of multigrid cycles
#define MG_CYCLES 10000

//! Flag for disabling multigrid (Disable MG = 1, Use MG = 0)
#define DISABLE_MG 0

//! Number of smoothing sweeps at each stage of the multigrid
#define NUM_SWEEP 3

//! Choice of iterative smoother (see SMOOTHER_T above)
#define SMOOTHER 2

//! Flag controlling whether to write Tecplot mesh/solution files (Yes=1,No=0)
#define VISUALIZE 0

//! Iteration frequency with which to print to console and write output files
#define FREQUENCY 1

//! Convergence criteria in terms of orders reduction in the L2 norm
#define TOLERANCE 12.0

#define pow_tol -10

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
void smooth_sor(double **phi, double **f, double **aux, int n_nodes,
                int n_sweeps);

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
int NRank, MyRank;
const int RootRank  = 0;
int main(int argc, char *argv[]) {
    // initialize MPI

   MPI_Init( &argc, &argv );
   MPI_Comm_rank( MPI_COMM_WORLD, &MyRank );
   MPI_Comm_size( MPI_COMM_WORLD, &NRank );

   
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
  if (MyRank == RootRank){
  write_settings(n_nodes, n_levels, n_mgcycles);
  }
  //! Create the fine mesh and store the x and y coordinates
  
  generate_fine_mesh(x[FINE_MESH], y[FINE_MESH], n_nodes, visualize);
  
  //! Create the coarse multigrid levels in a recursive fashion
  
  coarsen_mesh(x, y, n_nodes, n_levels, FINE_MESH, visualize);
  
  //! Initialize the approximate and exact solutions on all mesh levels
  
  intitialize_solution(phi[FINE_MESH], phi_exact, f[FINE_MESH],
                       x[FINE_MESH], y[FINE_MESH], n_nodes);
  
  //! Compute the initial value of the residual before any smoothing
  
  residual_0 = compute_residual(phi[FINE_MESH], f[FINE_MESH], aux[FINE_MESH],
                                n_nodes);
  
  //! Write the initial residual and solution files if requested
  if (MyRank == RootRank){
  write_output(phi[FINE_MESH], phi_exact, x[FINE_MESH],
               y[FINE_MESH], n_nodes, i_mgcycles, residual_0, visualize);
  }

 
 
 // printf("hihi wating %d\n",MyRank);
  MPI_Barrier(MPI_COMM_WORLD);
  //printf("hihi barrie %d\n",MyRank);

  //MPI_Bcast( &residual_0, 1, MPI_DOUBLE, RootRank, MPI_COMM_WORLD );
  
  
    
    //! Main solver loop over the prescribed number of multigrid cycles
  
  int count=0;
  for (i_mgcycles = 1; i_mgcycles <= n_mgcycles; i_mgcycles++) {
     count++;
    //! Call the recursive multigrid cycle method
  //  printf("start the mg %d",MyRank);
    multigrid_cycle(phi, f, aux, n_nodes, n_sweeps, n_levels, FINE_MESH);
   // printf("done the mg %d",MyRank);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (MyRank == RootRank){   
    residual = compute_residual(phi[FINE_MESH], f[FINE_MESH], aux[FINE_MESH],
                                n_nodes);
    }                                
    //if (residual < pow(10,pow_tol)) stop_calc = true;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast( &residual, 1, MPI_DOUBLE, RootRank, MPI_COMM_WORLD );
    
    //if (abs(residual-residual_old)<pow(10,-3)) stop_calc = true;
    //residual_old=residual;
    if (count==3) stop_calc=true;
    if (MyRank == RootRank){
    if ((i_mgcycles%freq == 0) || stop_calc)
      write_output(phi[FINE_MESH], phi_exact, x[FINE_MESH], y[FINE_MESH],
                   n_nodes, i_mgcycles, residual, visualize);
    }

    if (stop_calc) break;
    
  }
  
  //! Free all memory used by the solver
  if (stop_calc && MyRank == RootRank) {
    printf("\nConverged %3.1f orders of magnitude...\n", tolerance);
    printf("#============================================#\n\n");
  } else if (MyRank==RootRank) printf("\n#============================================#\n\n");
  

  
  MPI_Finalize();
  
  //deallocate_arrays(phi, phi_exact, f, x, y, aux, n_nodes, n_levels);
  
  //! Print final message to console
  
  
  
  return EXIT_SUCCESS;
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
    if ((nodes%2 == 0) && (nodes>= 8)) {
      nodes = (nodes)/2 ;
      n_levels++;
      printf("%d / %d",nodes,n_levels);
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
    
    nodes = (nodes)/2 ;
    
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

void coarsen_mesh(double ***x, double ***y, int n_nodes,
                  int n_levels, int level, bool visualize) {
  
  //! Create coarse MG levels by keeping every other node in
  //! the fine mesh. Note that this is a recursive function.
  
  if (level == n_levels-1) {
    
    //! Do nothing. We have reached the coarsest level.
    
    if (n_levels == 1) printf("No multigrid...\n");
    
    return;
    
  } else {
    
    //! Coarsen the current level
    
    for (int i = 0; i < n_nodes; i++) {
      for (int j = 0; j < n_nodes; j++) {
        
        //! Store the node coordinates at every other fine mesh point
        //! in the coarser mesh using the halved coarse mesh indices.
        
        if ((i%2 == 0) && (j%2 == 0)) {
          x[level+1][i/2][j/2] = x[level][i][j];
          y[level+1][i/2][j/2] = y[level][i][j];
        }
        
      }
    }
    
    //! Set up the new index values and increment the mesh level
    
    int n_coarse      = (n_nodes)/(2) ;
    int levels_coarse = level+1;
    
    printf("Created coarse grid (%d x %d) for level %d...\n", n_coarse,
           n_coarse, levels_coarse);
    
    //! Plot the mesh if requested (can help debug the multigrid)
    
    if (visualize) write_mesh(x[level+1], y[level+1], n_coarse, levels_coarse);
    
    //! Call the function again to coarsen the next level recursively.
    
    coarsen_mesh(x, y, n_coarse, n_levels, levels_coarse, visualize);
    
  }
  
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

void multigrid_cycle(double ***phi, double ***f, double ***aux, int n_nodes,
                     int n_sweeps, int n_levels, int level) {
  
  //! Pre-smooth the solution on this level with a number of sweeps
  
  switch (SMOOTHER) {
    case JACOBI:
      smooth_jacobi(phi[level], f[level], aux[level], n_nodes, n_sweeps);
      break;
    case GAUSS_SEIDEL:
      smooth_gauss_seidel(phi[level], f[level], aux[level], n_nodes, n_sweeps);
      break;
    case SOR:
      //printf( "SOR World on rank %d/%d\n", MyRank, NRank );
      smooth_sor(phi[level], f[level], aux[level], n_nodes, n_sweeps);
      //printf( "Done SOR World on rank %d/%d\n", MyRank, NRank );
      break;
    default:
      printf( "\n   !!! Error !!!\n" );
      printf( " Unrecognized smoother. \n\n");
      exit(1);
      break;
  }

  
    if (level < n_levels-1) {
    
    //! Restrict the fine solution down onto the coarser grid by
    //! computing the forcing term, i.e., f_coarse = restrict(residual_fine)
    
    restrict_weighted(phi, f, aux, n_nodes, level);
    
    //! Compute some information about the coarse level
    
    int n_coarse     = (n_nodes-1)/(2) + 1;
    int level_coarse = level + 1;
    
    //! Call the recursive multigrid cycle method on the next coarse level
    
    multigrid_cycle(phi, f, aux, n_coarse, n_sweeps, n_levels, level_coarse);
    
    //! Prolongate the solution for moving up to finer mesh levels,
    //! i.e., phi_fine = phi_fine + prolong(phi_coarse)
    
    prolongate_weighted(phi, aux, n_nodes, level);
    
  

  }
  switch (SMOOTHER) {
    case JACOBI:
      smooth_jacobi(phi[level], f[level], aux[level], n_nodes, n_sweeps);
      break;
    case GAUSS_SEIDEL:
      smooth_gauss_seidel(phi[level], f[level], aux[level], n_nodes, n_sweeps);
      break;
    case SOR:
      smooth_sor(phi[level], f[level], aux[level], n_nodes, n_sweeps);
      break;
    default:
      printf( "\n   !!! Error !!!\n" );
      printf( " Unrecognized smoother. \n\n");
      exit(1);
      break;
  }

  
  
}


void smooth_sor(double **phi, double **f, double **aux, int n_nodes,
                int n_sweeps) {
//  printf( "smooth World on rank %d/%d\n", MyRank, NRank );
  //! Smooth the system using a prescribed number of SOR sweeps. If the
  //! relaxation factor is set to 1.0, Gauss_Seidel it recovered. For
  //! a relax parameter < 1.0 it is under-relaxation, while if a relax
  //! parameter > 1.0 is chosen, it is over-relaxation.
  //! Set the relaxation parameter and compute the mesh spacing.
  
  double relax = RELAX;
  double h2 = pow(1.0/((double)n_nodes-1.0),2.0);
  
   // prepare the send buffer
  int SendCount = n_nodes*n_nodes/2;
  int RecvCount = SendCount;
  int bcCount=n_nodes;
  int tag = n_nodes;
  double RecvBuf[n_nodes/2][n_nodes];
  double r1_bc[n_nodes];
  double bc_rev[n_nodes];   
  int Barrier=MyRank, Btag=172365,brecv;
  int TargetRank = (MyRank+1)%2;
   
  MPI_Barrier(MPI_COMM_WORLD);

//   printf("sizeof r1bc : %ld \n", sizeof(r1_bc)/sizeof(r1_bc[0]));
   //printf("sizeof recv : %ld \n", sizeof(RecvBuf));


//   printf( "before for rank %d/%d\n", MyRank, NRank );
  MPI_Scatter( &**phi, SendCount, MPI_DOUBLE,&RecvBuf, RecvCount, MPI_DOUBLE,RootRank, MPI_COMM_WORLD );
   
   
  for (int i=1;i<n_nodes/2-1;i++){
  for (int j=1;j<(n_nodes-1);j++){

        RecvBuf[i][j] = (1.0 - relax)*RecvBuf[i][j] + relax*(RecvBuf[i][j-1] + RecvBuf[i-1][j] +\
                         RecvBuf[i+1][j] + RecvBuf[i][j+1] + h2*f[i][j])/4.0;
        //printf( "  %f", RecvBuf[i][j] );
    }//j node
        //printf( "\n" );   
  }// i node
  
 

//  printf( "before 687 rank %d/%d\n", MyRank, NRank );
  if (MyRank == RootRank){
     // printf("error0?");
       MPI_Recv(&bc_rev,n_nodes,MPI_DOUBLE,TargetRank,tag,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       int t=n_nodes/2-1;
       for (int j=1;j<(n_nodes-1);j++){
         
         RecvBuf[t][j] = (1.0 - relax)*RecvBuf[t][j] + relax*(RecvBuf[t][j-1] + RecvBuf[t-1][j] +\
                          bc_rev[j] + RecvBuf[t][j+1] + h2*f[t][j])/4.0;
       }
       for (int i=0;i<n_nodes;i++){ r1_bc[i]=RecvBuf[t][i]; }
       MPI_Ssend(&r1_bc,n_nodes,MPI_DOUBLE,TargetRank,tag,MPI_COMM_WORLD);
  
   }

   else {
      for (int i=0;i<n_nodes;i++){r1_bc[i]=RecvBuf[0][i];}
       
       MPI_Send(&r1_bc,n_nodes,MPI_DOUBLE,TargetRank,tag,MPI_COMM_WORLD);
       MPI_Recv(&bc_rev,n_nodes,MPI_DOUBLE,TargetRank,tag,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       for (int j=1;j<(n_nodes-1);j++){
        RecvBuf[0][j] = (1.0 - relax)*RecvBuf[0][j] + relax*(RecvBuf[0][j-1] +bc_rev[j] +\
                          RecvBuf[0+1][j] + RecvBuf[0][j+1] +h2*f[0][j])/4.0;

       }
      // printf("error1?");
    //MPI_Recv( &brecv, 1, MPI_INT, TargetRank, Btag, MPI_COMM_WORLD, MPI_STATUS_IGNORE );   
  }  // second rank else 

int MPI_Barrier(MPI_Comm comm);
MPI_Gather( &RecvBuf, SendCount, MPI_DOUBLE, &**phi, RecvCount,MPI_DOUBLE,RootRank,MPI_COMM_WORLD );

  //for (int iter = 0; iter < n_sweeps; iter++) {

if (MyRank!=RootRank){
  printf("nodes=%d recv rank = %d\n",n_nodes,MyRank); 
   for (int i=0; i<n_nodes/2; i++) {
            for (int j=0; j<n_nodes; j++) {
               printf( "  %f", RecvBuf[i][j] );
            }
            printf( "\n" );
   }
}  
else {
  printf("nodes=%d recv rank = %d\n",n_nodes,MyRank); 
   for (int i=0; i<n_nodes/2; i++) {
            for (int j=0; j<n_nodes; j++) {
               printf( "  %f", RecvBuf[i][j] );
            }
            printf( "\n" );
   }

  printf("nodes=%d gather\n",n_nodes); 
   for (int i=0; i<n_nodes; i++) {
            for (int j=0; j<n_nodes; j++) {
               printf( "  %f", phi[i][j] );
            }
            printf( "\n" );
   }

 }
//printf( "before barrier rank %d/%d\n", MyRank, NRank );

//}//iter loop
}//function

void restrict_weighted(double ***phi, double ***f, double ***aux, int n_nodes,
                       int level) {
  
  //! Restrict the solution, i.e., transfer it from a fine to coarse level.
  //! First, we need to compute the residual on the fine level, as this is
  //! what we are restricting onto the coarse mesh.
  
  compute_residual(phi[level], f[level], aux[level], n_nodes);
  
  //! Compute some information about the coarse level
  
  int n_coarse     = (n_nodes)/(2) ;
  int level_coarse = level+1;
  int i_fine, j_fine;
  
  //! Initialize the solution guess and forcing term for the coarse level
  
  for (int i = 0; i < n_coarse; i++) {
    for (int j = 0; j < n_coarse; j++) {
      phi[level_coarse][i][j] = 0.0;
      f[level_coarse][i][j]   = 0.0;
    }
  }
  
  //! Transfer the forcing term to the coarse mesh. We are
  //! restricting by weighting the values on the fine mesh that
  //! surround the given coarse node. The residual on the boundary
  //! nodes should be zero, so we avoid those points in our loop.
  
  for (int i = 1; i < n_coarse-1; i++) {
    for (int j = 1; j < n_coarse-1; j++) {
      
      //! Calculate the indices on the fine mesh for clarity
      
      i_fine = (i*2); j_fine = (j*2);
      
      //! Perform the restriction operation for this node by injection
      
      f[level_coarse][i][j] = (  aux[level][i_fine-1][j_fine+1]*(1.0/16.0)
                               + aux[level][ i_fine ][j_fine+1]*(1.0/8.0)
                               + aux[level][i_fine+1][j_fine+1]*(1.0/16.0)
                               + aux[level][i_fine-1][ j_fine ]*(1.0/8.0)
                               + aux[level][ i_fine ][ j_fine ]*(1.0/4.0)
                               + aux[level][i_fine+1][ j_fine ]*(1.0/8.0)
                               + aux[level][i_fine-1][j_fine-1]*(1.0/16.0)
                               + aux[level][ i_fine ][j_fine-1]*(1.0/8.0)
                               + aux[level][i_fine+1][j_fine-1]*(1.0/16.0));
      
    }
  }
  
}

void prolongate_weighted(double ***phi, double ***aux, int n_nodes,
                         int level) {
  
  //! Prolongate the solution, i.e., transfer it from a coarse to fine level.
  //! We compute a correction to the fine grid solution and add. To create
  //! the correction from the coarse solution, nodes that lie on top of each
  //! other will be transfered directly from coarse->fine, while neighbors
  //! are interpolated using a weighting of neighbors.
  
  //! Compute some information about the coarse level
  
  int n_coarse     = (n_nodes)/(2) ;
  int level_coarse = level+1;
  int i_fine, j_fine;
  
  //! Initialize correction to zero, just in case
  
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      aux[level][i][j] = 0.0;
    }
  }
  
  //! Loop over all coarse points and set the solution on the
  //! coincident set of points on the fine grid. At the same time,
  //! contribute weighted values at coarse grid nodes to the set of
  //! neighboring fine grid nodes that are not coincident.
  
  for (int i = 1; i < n_coarse-1; i++) {
    for (int j = 1; j < n_coarse-1; j++) {
      
      //! Calculate the indices on the fine mesh for clarity
      
      i_fine = i*2; j_fine = j*2;
      
      //! Perform the prolongation operation by copying the value for
      //! a coincident node on the fine mesh and also incrementing the
      //! values for the neighbors.
      
      aux[level][i_fine-1][j_fine+1] += phi[level_coarse][i][j]*(1.0/4.0);
      aux[level][ i_fine ][j_fine+1] += phi[level_coarse][i][j]*(1.0/2.0);
      aux[level][i_fine+1][j_fine+1] += phi[level_coarse][i][j]*(1.0/4.0);
      aux[level][i_fine-1][ j_fine ] += phi[level_coarse][i][j]*(1.0/2.0);
      aux[level][ i_fine ][ j_fine ]  = phi[level_coarse][i][j];
      aux[level][i_fine+1][ j_fine ] += phi[level_coarse][i][j]*(1.0/2.0);
      aux[level][i_fine-1][j_fine-1] += phi[level_coarse][i][j]*(1.0/4.0);
      aux[level][ i_fine ][j_fine-1] += phi[level_coarse][i][j]*(1.0/2.0);
      aux[level][i_fine+1][j_fine-1] += phi[level_coarse][i][j]*(1.0/4.0);
      
    }
  }
  
  //! Finally, add the coarse grid correction to the fine grid,
  //! i.e., phi_fine = phi_fine + prolong(phi_coarse)
  
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      phi[level][i][j] += aux[level][i][j];
    }
  }
  
}

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
  printf("start j dimension");
  int nodes = n_nodes;
  for (int i_level = 0; i_level < n_levels; i_level++) {
    
    //! Delete j dimension
    
    for (int i = 0; i < nodes; i++) {
      delete [] x[i_level][i];
      delete [] y[i_level][i];
      //delete [] phi[i_level][i];
      delete [] aux[i_level][i];
      delete [] f[i_level][i];
      if (i_level == 0) delete [] phi_exact[i];
    }
    
    //! Delete i dimension
    
    delete [] x[i_level];
    delete [] y[i_level];
    //delete [] phi[i_level];
    delete [] aux[i_level];
    delete [] f[i_level];
    if (i_level == 0) delete [] phi_exact;
    
    //! Compute number of nodes on the next coarse level
    
    nodes = (nodes)/2 ;
    
  }
  
  //! Delete levels dimension
  
  delete [] x;
  delete [] y;
  //delete [] phi;
  delete [] aux;
  delete [] f;
  
}
