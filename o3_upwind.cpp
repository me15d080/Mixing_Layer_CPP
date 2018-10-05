/*--------------------------------------------------------------------------
DNS OF TEMPORAL 2D MIXING LAYER 
V. John / Journal of Computational and Applied Mathematics 173 (2005) 57–80
+ convective term: 3rd order upwinded
+ spatial terms: 6th order central
+ time stepping: 3rd order Adams Bashforth
---------------------------------------------------------------------------*/

# include <iostream>
# include <cmath>
# include <fstream>
# include <iomanip>
# include <cstdlib>
# include <cstdio>
# include <armadillo>  

using namespace std;
using namespace arma;

# define pi  3.14159265358979

//-> LIST OF SUB ROUTINES

double** make_zeros  (int, int);//R_01
double** make_x      (double**, int, int, double, double);//R_02
double** make_y      (double**, int, int, double, double);//R_03
double** find_ddx    (double**, double**, int, int, double, double, double, double);//R_04
double** find_ddy    (double**, double**, int, int, double, double, double, double);//R_05
double** find_gdfdx  (double**, double**, double**, int, int, double, double, double);//R_06
double** find_gdfdy  (double**, double**, double**, int, int, double, double, double);//R_07
double** find_d2dx2  (double**, double**, int, int, double, double, double, double);//R_08
double** find_d2dy2  (double**, double**, int, int, double, double, double, double);//R_09
double** Poisson_fft (double**, double**, double, double, double, double, int, int, double, double, double);//R_10
void     clean_zeros (double**, int);//R_11
void     write_data  (int, double**, double**, double**, int, int, char const*, char const*, char const*);//R_12
void     display     (double**, int, int );//R_13
double   find_sigT   (double**, int, int);//R_14
void     timestamp   ( );//R_15


/*------------------------- MAIN STARTS HERE---------------------------------------*/

int main()
{
// put a time stamp at the start of the execution
cout<<"PROGRAM STARTS @ "<<endl;
timestamp ( );
// char const* for naming the flow field files
char const* ch1="u"; 
char const* ch2="v"; 
char const* ch3="omg";
// no. of grid points in x and y direction (should be in powers of 2 for fastest FFT implementation)
int nx = 512;//next, go to line 163 
int ny = 512;
// actual computational domain
double xmin =-1.0;
double xmax =+1.0;
double ymin =-1.0;
double ymax =+1.0;
// kinematic viscosity
double nu= 0.000071429;
// grid spacings (uniform)
double dx=(xmax-xmin)/(nx-1);
double dy=(ymax-ymin)/(ny-1);
// compact scheme coefficients
double alp1, a1, b1;
double alp2, a2, b2;
double a3, b3;
// vorticity thickness
double omg_T;

// refer 
a1= 1.555555556;   b1= 0.111111111; alp1= 0.333333333 ;
a2= 1.090909091;   b2= 0.272727273; alp2= 0.181818182 ;
a3= 0.083333333;   b3= 0.25;


//actual grid

  double** x = make_zeros (nx, ny) ;
  double** y = make_zeros (nx, ny) ;
  x = make_x (x, nx, ny, xmin , dx);                 
  y = make_y (y, nx, ny, ymin , dy);  


//flow initialization

double cnoise   = 0.001;
double sig0     = 0.071428571; // initial vortex sheet thickness
double sig0_inv = 14.0;
double Winf     = 1.0; 

double** phi    = make_zeros (nx, ny);
double** unoise = make_zeros (nx, ny);
double** vnoise = make_zeros (nx, ny);
double** u      = make_zeros (nx, ny);
double** v      = make_zeros (nx, ny);
  
for(int i = 0; i < nx; i++)
   {
    for(int j = 0; j < ny; j++)
       {
        phi   [i][j] = exp(-4*sig0_inv*sig0_inv*y[i][j]*y[i][j]);    
        unoise[i][j] = cnoise*Winf*(-8*sig0_inv*sig0_inv*y[i][j])*phi[i][j]*(cos(8*pi*x[i][j]) + cos(20*pi*x[i][j]));
        vnoise[i][j] = cnoise*Winf*pi*phi[i][j]*(8*sin(8*pi*x[i][j]) + 20*sin(20*pi*x[i][j]));
        u[i][j]      = Winf*tanh(2*sig0_inv*y[i][j]) + unoise[i][j];
        v[i][j]      = vnoise[i][j];
       }    
   }

//computational grid    

int nxc       = nx-1;
int nyc       = (2*ny-1)-1;
double** uc   = make_zeros (nxc, nyc);
double** vc   = make_zeros (nxc, nyc);
double** omgc = make_zeros (nxc, nyc);


for(int i = 0; i < nxc; i++)
   {
    for(int j = 0; j < ny; j++)
       {
        uc[i][j] = u[i][j];
        vc[i][j] = v[i][j];

       }    
   }
/* pad with symmetric and anti symmetric extenstions and  
   clip the last row and column due to periodicity*/

for(int i = 0; i < nxc; i++)
   {
    for(int j = ny; j < nyc; j++)
       {
        uc[i][j] =  u[i][2*ny-j-2];//symmetric extension 
        vc[i][j] = -v[i][2*ny-j-2];//anti symmetric extension

       }    
   }

   
double Lx=xmax-xmin;
double Ly=ymax-ymin;
double Lxc=Lx;
double Lyc=2*Ly;
// wavenumber grid spacing
double Qx=2*pi/Lxc;
double Qy=2*pi/Lyc;
double t=0  ;
double t0=sig0/Winf;

/*--------------ATTENTION---------------------------------
dt should be < 0.1*dx & in terms of t0. 
----------------------------------------------------------
 N        dt_max(= 0.1dx)            dt_taken
064        3.1746e-3           7.1428e-4 (=0.01*t0)
128        1.5748e-3           7.1428e-4 (=0.01*t0)
256        7.8431e-4           7.1428e-4 (=0.01*t0)
512        3.9113e-4           3.5714e-4 (=0.5*0.01*t0)
--------------------------------------------------------*/

double dt  = 0.5*0.01*t0; /*0.5*0.01*t0 (for 512)*/ 
double tmax= 50*t0;   
/*Total 10,000 files for 512. For lower resolutions, 5000 files. 
For contours,{k,2k,3k,4k,5k} for N<=256 or {2k, 4k, 6k, 8k, 10k} for N=512*/ 

int    iter=0;// 100 iter = 1 t0 

double** ddx_vc     = make_zeros (nxc, nyc);         
double** ddy_uc     = make_zeros (nxc, nyc);        

double** ucddx_uc   = make_zeros (nxc, nyc);         
double** ucddx_vc   = make_zeros (nxc, nyc);        
double** vcddy_uc   = make_zeros (nxc, nyc);        
double** vcddy_vc   = make_zeros (nxc, nyc); 

double** d2dx2_uc   = make_zeros (nxc, nyc);
double** d2dy2_uc   = make_zeros (nxc, nyc);
double** d2dx2_vc   = make_zeros (nxc, nyc);
double** d2dy2_vc   = make_zeros (nxc, nyc);

double** div_vel    = make_zeros (nxc, nyc);

double** p          = make_zeros (nxc, nyc);
double** dpdx       = make_zeros (nxc, nyc);
double** dpdy       = make_zeros (nxc, nyc);


double** Ax         = make_zeros (nxc, nyc);
double** Ay         = make_zeros (nxc, nyc);
double** Dx         = make_zeros (nxc, nyc);
double** Dy         = make_zeros (nxc, nyc);

double** uc_ss      = make_zeros (nxc, nyc);
double** vc_ss      = make_zeros (nxc, nyc);
double** ddx_uc_ss  = make_zeros (nxc, nyc);
double** ddy_vc_ss  = make_zeros (nxc, nyc);

double** Fx0        = make_zeros (nxc, nyc);
double** Fy0        = make_zeros (nxc, nyc);
double** Fx         = make_zeros (nxc, nyc);
double** Fy         = make_zeros (nxc, nyc);

std::ofstream write_output("sigT.dat");

// start the time loop

while(t<tmax)

{

// -- advective flux calculation -- //

ddx_vc       = find_ddx(ddx_vc, vc, nxc, nyc, alp1, a1, b1, dx );
ddy_uc       = find_ddy(ddy_uc, uc, nxc, nyc, alp1, a1, b1, dy );

ucddx_uc     = find_gdfdx(ucddx_uc, uc, uc, nxc, nyc, a3, b3, dx );         
ucddx_vc     = find_gdfdx(ucddx_vc, uc, vc, nxc, nyc, a3, b3, dx );        
vcddy_uc     = find_gdfdy(vcddy_uc, vc, uc, nxc, nyc, a3, b3, dy );        
vcddy_vc     = find_gdfdy(vcddy_vc, vc, vc, nxc, nyc, a3, b3, dy );        

// calculate vorticity

for (int i=0; i<nxc; i++)
    {
     for (int j=0; j<nyc; j++)
          {
            omgc[i][j]= ddx_vc[i][j]-ddy_uc[i][j];             
          }

    }
/*
write_data(iter, uc, vc, omgc, nxc, nyc, ch1, ch2, ch3);
*/

omg_T = find_sigT(omgc, nxc, nyc);cout<<omg_T<<endl;

write_output << omg_T << "\n";

for (int i=0; i<nxc; i++)
    {
     for (int j=0; j<nyc; j++)
          {
            Ax[i][j]=-(ucddx_uc[i][j] + vcddy_uc[i][j]);               
            Ay[i][j]=-(ucddx_vc[i][j] + vcddy_vc[i][j]);
          }

    }


// -- diffusive flux calculation -- //

d2dx2_uc = find_d2dx2(d2dx2_uc, uc , nxc, nyc, alp2, a2, b2, dx );
d2dy2_uc = find_d2dy2(d2dy2_uc, uc , nxc, nyc, alp2, a2, b2, dy );
d2dx2_vc = find_d2dx2(d2dx2_vc, vc , nxc, nyc, alp2, a2, b2, dx );
d2dy2_vc = find_d2dy2(d2dy2_vc, vc , nxc, nyc, alp2, a2, b2, dy );

for (int i=0; i<nxc; i++)
    {
     for (int j=0; j<nyc; j++)
          {
            Dx[i][j]= nu*(d2dx2_uc[i][j] + d2dy2_uc[i][j]);               
            Dy[i][j]= nu*(d2dx2_vc[i][j] + d2dy2_vc[i][j]);
          }

    }
// Net flux
for (int i=0; i<nxc; i++)
    {
     for (int j=0; j<nyc; j++)
          {
            Fx[i][j]= Ax[i][j] + Dx[i][j];               
            Fy[i][j]= Ay[i][j] + Dy[i][j];
          }

    }

// Euler for initial time step (Net flux calculated = Initial flux)
if(t==0)
{
for (int i=0; i<nxc; i++)
    {
     for (int j=0; j<nyc; j++)
          {
            Fx0[i][j]= Fx[i][j];               
            Fy0[i][j]= Fy[i][j];
          }

    }

}

// projection step

for (int i=0; i<nxc; i++)
    {
     for (int j=0; j<nyc; j++)
          {
            uc_ss[i][j]= uc[i][j] + dt*(1.5*Fx[i][j]-0.5*Fx0[i][j]);               
            vc_ss[i][j]= vc[i][j] + dt*(1.5*Fy[i][j]-0.5*Fy0[i][j]);
          }

    }


ddx_uc_ss = find_ddx(ddx_uc_ss, uc_ss, nxc, nyc, alp1, a1, b1, dx);        
ddy_vc_ss = find_ddy(ddy_vc_ss, vc_ss, nxc, nyc, alp1, a1, b1, dy);    

// find divergence of velocity field 

for (int i=0; i<nxc; i++)
    {
     for (int j=0; j<nyc; j++)
          {
            div_vel[i][j]= ddx_uc_ss[i][j]+ddy_vc_ss[i][j];  
            div_vel[i][j]= div_vel[i][j] / dt; // RHS for Poisson equation                
          }

    }

//-- Pressure  step

p = Poisson_fft(p, div_vel, dx, dy, Qx, Qy, nxc, nyc, alp1, a1, b1);


//-- velocity correction step 

dpdx = find_ddx(dpdx, p , nxc, nyc, alp1, a1, b1, dx );        
dpdy = find_ddy(dpdy, p , nxc, nyc, alp1, a1, b1, dy );    

for (int i=0; i<nxc; i++)
    {
     for (int j=0; j<nyc; j++)
          {
            uc[i][j]= uc_ss[i][j]-dt*dpdx[i][j];  
            vc[i][j]= vc_ss[i][j]-dt*dpdy[i][j];                
          }

    }


// update old flux value

for (int i=0; i<nxc; i++)
    {
     for (int j=0; j<nyc; j++)
          {
            Fx0[i][j]= Fx[i][j];               
            Fy0[i][j]= Fy[i][j];
          }

    }


iter = iter+1;
t    = t + dt;

}
write_output.close();

//clean up arrays
clean_zeros (x, nx);
clean_zeros (y, nx);
clean_zeros (phi,nx); 
clean_zeros (unoise,nx); 
clean_zeros (vnoise,nx); 
clean_zeros (u,nx);
clean_zeros (v,nx);
clean_zeros (uc,nxc);
clean_zeros (vc,nxc);
clean_zeros (omgc,nxc);
clean_zeros (ddx_vc,nxc);
clean_zeros (ddy_uc,nxc); 
clean_zeros (d2dx2_uc,nxc);     
clean_zeros (d2dy2_uc,nxc);     
clean_zeros (d2dx2_vc,nxc);     
clean_zeros (d2dy2_vc,nxc);     
clean_zeros (div_vel,nxc);      
clean_zeros (p,nxc);            
clean_zeros (dpdx,nxc);         
clean_zeros (dpdy,nxc);   
clean_zeros (Ax,nxc);
clean_zeros (Ay,nxc);
clean_zeros (Dx,nxc);
clean_zeros (Dy,nxc);
clean_zeros (uc_ss,nxc) ;
clean_zeros (vc_ss,nxc);
clean_zeros (Fx0,nxc);
clean_zeros (Fy0,nxc);        
clean_zeros (Fx ,nxc);         
clean_zeros (Fy ,nxc);    
clean_zeros (ddx_uc_ss,nxc);
clean_zeros (ddy_vc_ss,nxc);

cout<<"PROGRAM ENDS @"<<endl;
timestamp ( );

return 0;
}


//------------------ MAIN ENDS HERE------------------------------//


//-->> DESCRIPTION OF SUB ROUTINES
//-------------------------------------R_01----------------------------------------------//

/* It is equivalent to MATLAB's zeros. Dynamic memory allocation for our matrices.
   M=zeros(nx,ny)
*/
double** make_zeros(int row, int col)
{
 double** M = 0;
 M = new double*[row];

 for (int i = 0; i < row; i++)
    {
      M[i] = new double[col];

      for (int j = 0; j < col; j++)
      {                  
        M[i][j] =0;
      }
    }

  return M;
 }


//-------------------------------------R_02----------------------------------------------//
/* X of: [X,Y]=ndgrid(x,y)
*/
double** make_x(double**M,int row, int col, double x_min, double dx)
{

  for(int i = 0; i < row; i++)
     {
      for(int j = 0; j < col; j++)
         {
          M[i][j] = x_min + i*dx;        
         }    
     }
     return M;     
}

//-------------------------------------R_03----------------------------------------------//
/* Y of: [X,Y]=ndgrid(x,y)
*/

double** make_y(double**M,int row, int col, double y_min, double dy)
{
  for(int i = 0; i < row; i++)
     {
      for(int j = 0; j < col; j++)
         {
          M[i][j] = y_min + j*dy;        
         }    
     }
     return M;   
}


//-------------------------------------R_04----------------------------------------------//

/* finds the 6th order accurate COMPACT first order spatial derivatives subjected to periodic BCs. In order to invert the matrix,
we used a freely available linear algebra library called 'ARMADILLO'.
*/

double** find_ddx(double**dMdx, double**M , int nx, int ny, double alp, double a, double b, double dx )
{
// coefficient matrix based on compact FD scheme
 mat A = zeros<mat>(nx,nx);           //Try: SpMat<double> A(nx,ny) or sp_mat A = zeros<sp_mat>(5,5);
 A(0,0)=1;A(0,1)=alp;A(0,nx-1)=alp;
 for (int i=1; i<nx-1;i++)
 {
  A(i,i-1)=alp;
  A(i,i)  =1;
  A(i,i+1)=alp;
 }
 A(nx-1,0)=alp;A(nx-1,nx-2)=alp;A(nx-1,nx-1)=1;

// RHS matrix
 mat RHS = zeros<mat>(nx,ny);
// boundary points: periodic BCs
for(int j=0; j<ny;j++)
   {
     RHS(0,j)   =(0.5*a*(M[1][j]   -M[nx-1][j])+0.25*b*(M[2][j]-M[nx-2][j]))/dx;//check
     RHS(1,j)   =(0.5*a*(M[2][j]   -M[0][j]   )+0.25*b*(M[3][j]-M[nx-1][j]))/dx;
     RHS(nx-2,j)=(0.5*a*(M[nx-1][j]-M[nx-3][j])+0.25*b*(M[0][j]-M[nx-4][j]))/dx;
     RHS(nx-1,j)=(0.5*a*(M[0][j]   -M[nx-2][j])+0.25*b*(M[1][j]-M[nx-3][j]))/dx;
   }

// interior points
for(int i=2; i < nx-2;i++)
  {
    for(int j = 0; j< ny;j++)
       {
        RHS(i,j)=(0.5*a*(M[i+1][j]-M[i-1][j])+0.25*b*(M[i+2][j]-M[i-2][j]))/dx;
       }
  }

// find derivative matrix
 mat LHS = zeros<mat>(nx,ny);
 LHS     = solve(A,RHS);//inv(A)*RHS;

// return the output 
for (int i=0;i<nx;i++)
  {
    for(int j=0;j<ny;j++)
       {
        dMdx[i][j]=LHS(i,j);
       }
  }
 
return dMdx;
}
//-------------------------------------R_05----------------------------------------------//
/* finds the 6th order accurate COMPACT first order spatial derivatives subjected to periodic BCs.In order to invert the matrix,
we used a freely available linear algebra library called 'ARMADILLO'.
*/

double** find_ddy(double**dMdy, double**M , int nx, int ny, double alp, double a, double b, double dy )
{
// coefficient matrix based on compact FD scheme
 mat A = zeros<mat>(ny,ny);// Try: SpMat<double> A(nx,ny) or sp_mat A = zeros<sp_mat>(5,5);
 A(0,0)=1;A(0,1)=alp;A(0,ny-1)=alp;
 for (int i=1; i<ny-1;i++)
 {
  A(i,i-1)=alp;
  A(i,i)  =1;
  A(i,i+1)=alp;
 }
 A(ny-1,0)=alp;A(ny-1,ny-2)=alp;A(ny-1,ny-1)=1;
// RHS matrix
 mat RHS = zeros<mat>(ny,nx);
// boundary points: periodic BCs
for(int j=0; j<nx;j++)
   {
     RHS(0,j)   =(0.5*a*(M[j][1]   -M[j][ny-1])+0.25*b*(M[j][2]-M[j][ny-2]))/dy;
     RHS(1,j)   =(0.5*a*(M[j][2]   -M[j][0]   )+0.25*b*(M[j][3]-M[j][ny-1]))/dy;
     RHS(ny-2,j)=(0.5*a*(M[j][ny-1]-M[j][ny-3])+0.25*b*(M[j][0]-M[j][ny-4]))/dy;
     RHS(ny-1,j)=(0.5*a*(M[j][0]   -M[j][ny-2])+0.25*b*(M[j][1]-M[j][ny-3]))/dy;
   }

// interior points
for(int i=2; i < ny-2;i++)
  {
    for(int j = 0; j< nx;j++)
       {
        RHS(i,j)=(0.5*a*(M[j][i+1]-M[j][i-1])+0.25*b*(M[j][i+2]-M[j][i-2]))/dy;
       }
  }

// find derivative matrix
 mat LHS = zeros<mat>(ny,nx);
 LHS     = solve(A,RHS);//inv(A)*RHS;
// return the output 
for (int i=0;i<nx;i++)
  {
    for(int j=0;j<ny;j++)
       {
        dMdy[i][j]=LHS(j,i);
       }
  }
 
return dMdy;
}
//-------------------------------------R_06----------------------------------------------//
/* finds the 3rd order accurate COMPACT convective terms subjected to periodic BCs. Refer to the following:
*/ 

double** find_gdfdx(double** gdfdx, double**g, double**f, int nxc, int nyc, double a, double b, double dx )
{
// boundary points: periodic
for(int j = 0; j< nyc;j++)
   {
    gdfdx[0][j]     = a*g[0][j]*((-f[2][j]+8*(f[1][j]-f[nxc-1][j])+f[nxc-2][j])/dx) + 
                      b*fabs(g[0][j])*((f[2][j]-4*f[1][j]+6*f[0][j]-4*f[nxc-1][j]+f[nxc-2][j])/dx);

    gdfdx[1][j]     = a*g[1][j]*((-f[3][j]+8*(f[2][j]-f[0][j])+f[nxc-1][j])/dx) + 
                      b*fabs(g[1][j])*((f[3][j]-4*f[2][j]+6*f[1][j]-4*f[0][j]+f[nxc-1][j])/dx);

    gdfdx[nxc-2][j] = a*g[nxc-2][j]*((-f[0][j]+8*(f[nxc-1][j]-f[nxc-3][j])+f[nxc-4][j])/dx) + 
                      b*fabs(g[nxc-2][j])*((f[0][j]-4*f[nxc-1][j]+6*f[nxc-2][j]-4*f[nxc-3][j]+f[nxc-4][j])/dx);

    gdfdx[nxc-1][j] = a*g[nxc-1][j]*((-f[1][j]+8*(f[0][j]-f[nxc-2][j])+f[nxc-3][j])/dx) + 
                      b*fabs(g[nxc-1][j])*((f[1][j]-4*f[0][j]+6*f[nxc-1][j]-4*f[nxc-2][j]+f[nxc-3][j])/dx);
   }

// interior points
for(int i=2; i < nxc-2;i++)
  {
    for(int j = 0; j< nyc;j++)
       {
        gdfdx[i][j] = a*g[i][j]*((-f[i+2][j]+8*(f[i+1][j]-f[i-1][j])+f[i-2][j])/dx) + 
                      b*fabs(g[i][j])*((f[i+2][j]-4*f[i+1][j]+6*f[i][j]-4*f[i-1][j]+f[i-2][j])/dx);
       }
  }

return gdfdx;
}
//-------------------------------------R_07----------------------------------------------//
/* finds the 3rd order accurate COMPACT convective terms subjected to periodic BCs. Refer to the following:
*/
double** find_gdfdy(double** gdfdy, double**g, double**f, int nxc, int nyc, double a, double b, double dy )
{
// boundary points: periodic
for(int i = 0; i< nxc;i++)
   {
     gdfdy[i][0]     = a*g[i][0]*((-f[i][2]+8*(f[i][1]-f[i][nyc-1])+f[i][nyc-2])/dy) + 
                       b*fabs(g[i][0])*((f[i][2]-4*f[i][1]+6*f[i][0]-4*f[i][nyc-1]+f[i][nyc-2])/dy);
                     
     gdfdy[i][1]     = a*g[i][1]*((-f[i][3]+8*(f[i][2]-f[i][0])+f[i][nyc-1])/dy) + 
                       b*fabs(g[i][1])*((f[i][3]-4*f[i][2]+6*f[i][1]-4*f[i][0]+f[i][nyc-1])/dy);

     gdfdy[i][nyc-2] = a*g[i][nyc-2]*((-f[i][0]+8*(f[i][nyc-1]-f[i][nyc-3])+f[i][nyc-4])/dy) + 
                       b*fabs(g[i][nyc-2])*((f[i][0]-4*f[i][nyc-1]+6*f[i][nyc-2]-4*f[i][nyc-3]+f[i][nyc-4])/dy);

                       
     gdfdy[i][nyc-1] = a*g[i][nyc-1]*((-f[i][1]+8*(f[i][0]-f[i][nyc-2])+f[i][nyc-3])/dy) + 
                       b*fabs(g[i][nyc-1])*((f[i][1]-4*f[i][0]+6*f[i][nyc-1]-4*f[i][nyc-2]+f[i][nyc-3])/dy);
                      
   }

// interior points
for(int i=0;i<nxc;i++)
  {
   for(int j=2;j<nyc-2;j++)
       {
        gdfdy[i][j] = a*g[i][j]*((-f[i][j+2]+8*(f[i][j+1]-f[i][j-1])+f[i][j-2])/dy) + 
                      b*fabs(g[i][j])*((f[i][j+2]-4*f[i][j+1]+6*f[i][j]-4*f[i][j-1]+f[i][j-2])/dy);
       }
  }


return gdfdy;
}

//-------------------------------------R_08----------------------------------------------//
/* finds the 6th order accurate COMPACT second order spatial derivatives subjected to periodic BCs.In order to invert the matrix,
we used a freely available linear algebra library called 'ARMADILLO'.
*/

double** find_d2dx2(double**d2Mdx2, double**M , int nx, int ny, double alp, double a, double b, double dx )
{
// coefficient matrix based on compact FD scheme
 mat A = zeros<mat>(nx,nx);//SpMat<double> A(nx,ny) or sp_mat A = zeros<sp_mat>(5,5);
 A(0,0)=1;A(0,1)=alp;A(0,nx-1)=alp;
 for (int i=1; i<nx-1;i++)
 {
  A(i,i-1)=alp;
  A(i,i)  =1;
  A(i,i+1)=alp;
 }
 A(nx-1,0)=alp;A(nx-1,nx-2)=alp;A(nx-1,nx-1)=1;
//RHS matrix
 mat RHS = zeros<mat>(nx,ny);
// boundary points: periodic
for(int j=0; j<ny;j++)
   {
     RHS(0,j)   = (a*(M[1][j]    - 2*M[0][j]    + M[nx-1][j]) + 0.25*b*(M[2][j]- 2*M[0][j]   + M[nx-2][j]))/(dx*dx);
     RHS(1,j)   = (a*(M[2][j]    - 2*M[1][j]    + M[0][j]   ) + 0.25*b*(M[3][j]- 2*M[1][j]   + M[nx-1][j]))/(dx*dx);
     RHS(nx-2,j)= (a*(M[nx-1][j] - 2*M[nx-2][j] + M[nx-3][j]) + 0.25*b*(M[0][j]- 2*M[nx-2][j]+ M[nx-4][j]))/(dx*dx);
     RHS(nx-1,j)= (a*(M[0][j]    - 2*M[nx-1][j] + M[nx-2][j]) + 0.25*b*(M[1][j]- 2*M[nx-1][j]+ M[nx-3][j]))/(dx*dx);
   }

// interior points
for(int i=2; i < nx-2;i++)
  {
    for(int j = 0; j< ny;j++)
       {
        RHS(i,j)= (a*(M[i+1][j]  - 2*M[i][j]   +  M[i-1][j]) +  0.25*b*(M[i+2][j]-2*M[i][j] + M[i-2][j]))/(dx*dx);
       }
  }

// find derivative matrix
 mat LHS = zeros<mat>(nx,ny);
 LHS     = solve(A,RHS);//inv(A)*RHS;
// return the output 
for (int i=0;i<nx;i++)
  {
    for(int j=0;j<ny;j++)
       {
        d2Mdx2[i][j]=LHS(i,j);
       }
  }
 
return d2Mdx2;
}
//-------------------------------------R_09----------------------------------------------//
/* finds the 6th order accurate COMPACT second order spatial derivatives subjected to periodic BCs.In order to invert the matrix,
we used a freely available linear algebra library called 'ARMADILLO'.
*/

double** find_d2dy2(double**d2Mdy2, double**M , int nx, int ny, double alp, double a, double b, double dy )
{
// coefficient matrix based on compact FD scheme
 mat A = zeros<mat>(ny,ny);//SpMat<double> A(nx,ny) or sp_mat A = zeros<sp_mat>(5,5);
 A(0,0)=1; A(0,1)=alp; A(0,ny-1)=alp;
 for (int i=1; i<ny-1;i++)
 {
  A(i,i-1)=alp;
  A(i,i)  =1;
  A(i,i+1)=alp;
 }
 A(ny-1,0)=alp; A(ny-1,ny-2)=alp; A(ny-1,ny-1)=1;

// RHS matrix
 mat RHS = zeros<mat>(ny,nx);
// boundary points: periodic
for(int j=0; j<nx;j++)
   {
     RHS(0,j)   =(a*(M[j][1]   -2*M[j][0]    +M[j][ny-1]) + 0.25*b*(M[j][2]-2*M[j][0]    +M[j][ny-2]))/(dy*dy);
     RHS(1,j)   =(a*(M[j][2]   -2*M[j][1]    +M[j][0]   ) + 0.25*b*(M[j][3]-2*M[j][1]    +M[j][ny-1]))/(dy*dy);
     RHS(ny-2,j)=(a*(M[j][ny-1]-2*M[j][ny-2] +M[j][ny-3]) + 0.25*b*(M[j][0]-2*M[j][ny-2] +M[j][ny-4]))/(dy*dy);
     RHS(ny-1,j)=(a*(M[j][0]   -2*M[j][ny-1] +M[j][ny-2]) + 0.25*b*(M[j][1]-2*M[j][ny-1] +M[j][ny-3]))/(dy*dy);
   }

// interior points
for(int i=2; i < ny-2;i++)
  {
    for(int j = 0; j< nx;j++)
       {
        RHS(i,j)=(a*(M[j][i+1]-2*M[j][i]+M[j][i-1])+0.25*b*(M[j][i+2]-2*M[j][i]+M[j][i-2]))/(dy*dy);
       }
  }

// find derivative matrix
 mat LHS = zeros<mat>(ny,nx);
 LHS     = solve(A,RHS);//inv(A)*RHS;
// return the output 
for (int i=0;i<nx;i++)
  {
    for(int j=0;j<ny;j++)
       {
        d2Mdy2[i][j]=LHS(j,i);
       }
  }
 
return d2Mdy2;

}
//-------------------------------------R_10----------------------------------------------//
/* Poisson solver using "ARMADILLO" library. We used its MATLAB like inverse, fft and ifft functions in particular.
*/

double** Poisson_fft(double**p, double**RHS , double dx, double dy, double dkx,
                     double dky, int nx, int ny, double alp, double a, double b  )
{

double dxs = dkx*dx;double dys = dky*dy;

mat rhs = zeros<mat>(nx,ny);
for (int i=0;i<nx;i++)
  {
    for(int j=0;j<ny;j++)
       {
        rhs(i,j)=RHS[i][j];
       }
  }
// Here we take fft
cx_mat RHSh = fft2(rhs); // cx_mat stands for complex matrix

vec kx = zeros<vec>(nx);
vec ky = zeros<vec>(ny);

//--------------------- ATTENTION PLEASE---------------------------//
// nx is odd due to clipping
for (int i=0           ; i < 0.5*(nx+1)  ; i++ ){ kx(i)   =  i; }
for (int i=0.5*(nx+1)  ; i < nx          ; i++ ){ kx(i)   = -(nx-i); }

// ny is still even due to mirroring and clipping.
for (int j=0      ; j < 0.5*ny  ; j++ ){ ky(j)   =  j; }
for (int j=0.5*ny ; j < ny      ; j++ ){ ky(j)   = -(ny-j); }

//-----------------------------------------------------------------//

mat KX = zeros<mat>(nx,ny);
mat KY = zeros<mat>(nx,ny);

// wavenumber matrix 

for(int i=0; i<nx;i++)
{
 for(int j=0; j<ny; j++)
    {
      KX(i,j)=kx(i);
      KY(i,j)=ky(j);
    }
}

mat theta_x=zeros<mat>(nx,ny);
mat theta_y=zeros<mat>(nx,ny);
mat KmX=zeros<mat>(nx,ny);
mat KmY=zeros<mat>(nx,ny);
mat I=ones<mat>(nx,ny);
mat Den=zeros<mat>(nx,ny);


theta_x=KX*dxs;
theta_y=KY*dys;

// Modified wavenumber for COMPACT scheme
KmX=(a*sin(theta_x)+0.5*b*sin(2*theta_x))/((I+2*alp*cos(theta_x))*dxs);
KmY=(a*sin(theta_y)+0.5*b*sin(2*theta_y))/((I+2*alp*cos(theta_y))*dys);


Den=-(dkx*dkx*(KmX%KmX) + dky*dky*(KmY%KmY));
cx_mat F=zeros<cx_mat>(nx,ny);
F=RHSh/Den;
// Remove unphysical entries
for(int i=0; i<nx; i++)
{
 for(int j=0; j<ny; j++)
    {
     if(fabs(Den(i,j))<1e-13)
       {
        real(F(i,j))=0;
        imag(F(i,j))=0;
       }
    }
}

// Here we take inverse FFT
cx_mat LHS = ifft2(F);

// return the output 
for (int i=0;i<nx;i++)
  {
    for(int j=0;j<ny;j++)
       {
        p[i][j]=real(LHS(i,j));
       }
  }

return p;

}

//-------------------------------------R_11----------------------------------------------//
/*
clean the allocated memory space.  
*/
void clean_zeros(double**M, int row)
{
for (int  i = 0; i < row; i++)
    {
     delete [] M[i];
    }
 delete [] M;
}
//-------------------------------------R_12----------------------------------------------//
void     write_data (int iter, double** M1, double** M2, double** M3, int nxc, int nyc, 
                     char const* ch1, char const* ch2, char const* ch3)
{

int ny= 0.5*(nyc+2);// neglect the extended domain

// Create a folder

char command[50];
sprintf(command,"mkdir /home/vikas/Desktop/mixinglayer_cpp/out_o3upwind/%d",iter);
system (command);

// Create files

char fname1[50];
sprintf(fname1, "/home/vikas/Desktop/mixinglayer_cpp/out_o3upwind/%d/%s",iter,ch1);
ofstream fout1;
fout1.open(fname1);

char fname2[50];
sprintf(fname2, "/home/vikas/Desktop/mixinglayer_cpp/out_o3upwind/%d/%s",iter,ch2);
ofstream fout2;
fout2.open(fname2);

char fname3[50];
sprintf(fname3, "/home/vikas/Desktop/mixinglayer_cpp/out_o3upwind/%d/%s",iter,ch3);
ofstream fout3;
fout3.open(fname3);


 // openFoam style 
 for(int j = 0; j < ny; j++)
    {
     for(int i = 0; i < nxc; i++)
         {

          fout1.precision(18); 
          fout1 << M1[i][j]; 
          fout1 << "\n";  

          fout2.precision(18); 
          fout2 << M2[i][j]; 
          fout2 << "\n";      

          fout3.precision(18); 
          fout3 << M3[i][j]; 
          fout3 << "\n";      

         }
       
    }

fout1.close();
fout2.close();
fout3.close();

}
//-------------------------------------R_13----------------------------------------------//
void display(double** M, int nx, int ny)
/*when called, displays the matrix contents*/
{
  for(int i = 0; i < nx; i++)//row wise
    {
     for(int j = 0; j < ny; j++)
         {
           cout<< M[i][j]<<"   ";
         }
   
     cout<<endl;
       
    }

}
//-------------------------------------R_14----------------------------------------------//
/*
when called, finds vorticity thickness. For the formula, refer to V.John paper. 
*/
double find_sigT(double** omg, int nxc, int nyc)
{
int NN = 0.5*(nyc+2);

double s;
double omg_t;
double Imax;

double* x;
double* y;
double* I;
x = new double [nxc];// x(1:end-1)
y = new double [nxc];// omg(x(1:end-1)) @ given y(j)
I = new double [NN]; // integral at each y locations

//---- define x ----//

double dx = 2.0/nxc;


for (int i=0; i<nxc; i++)
   {
     x[i] = -1 + i*dx;
   }


//---- define y ----//
for (int j=0; j < NN; j++)
   {
     for (int i=0; i < nxc; i++)
         {
           y[i] = omg[i][j];
         }

     //----find sigT @ y[j] ----//
     s = 0;
         for (int k=0; k< nxc-1 ; k++)
             {
              s = s +0.5*(x[k+1]-x[k])*(y[k]+y[k+1]);//-->Integration using trapezoidal rule
             }

             I[j]=s;
   }

// find sigMAX
Imax=fabs(I[0]);
for (int j=1; j < NN; j++)
{
  if(fabs(I[j]) > Imax)
 {
  Imax  = fabs(I[j]);
 }

}
         
omg_t = 56/Imax;// refer the definition
delete[] x;
delete[] y;
delete[] I;

return omg_t;

}

//-------------------------------------R_15----------------------------------------------//

void timestamp ( )
/* when called, displays the date and time on the terminal. Used for estimating the total exection time.  
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
