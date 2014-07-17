#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "spme_kernel.h"


inline void interpolate_kernel (int porder3, double *P, int *ind,
                                double *grid, int ld3, double *vels,
                                double alpha, double beta)
{
    int j;
    __declspec(align(64)) double tmp[8];    
    int idx;
    double pvalue;
    
    tmp[0] = 0.0;
    tmp[1] = 0.0;
    tmp[2] = 0.0;
    for (j = 0; j < porder3; j++)
    {
        idx = ind[j];
        pvalue = P[j];
        tmp[0] += grid[0 * ld3 + idx] * pvalue;
        tmp[1] += grid[1 * ld3 + idx] * pvalue;
        tmp[2] += grid[2 * ld3 + idx] * pvalue;           
    }
    vels[0] =  beta * vels[0] + alpha * tmp[0];
    vels[1] =  beta * vels[1] + alpha * tmp[1];
    vels[2] =  beta * vels[2] + alpha * tmp[2];
}


void interpolate (int np, int porder3,
                  double *P, int ldP, int *ind, int ldind,
                  double *grid, int ld3, double *vels,
                  double alpha, double beta)
{
    struct timeval tv1;
    struct timeval tv2;
    printf ( "    interpolate ...\n");
    gettimeofday (&tv1, NULL);

    int i;
    int k;
    int endi;
    double flops;
    double mops;
    
    flops = (double)2.0 * np * porder3 * 3;
    mops = (double)np * porder3 * (4 + 8 * 4);

    #pragma omp parallel for schedule(dynamic) private(i, k, endi)
    for (i = 0; i < np; i+=8)
    {
        endi = i + 8 > np ? np : i + 8;
        for (k = i; k < endi; k++)
        {
            interpolate_kernel (porder3, &(P[k*ldP]), &(ind[k*ldind]),
                                grid, ld3, &(vels[3*k]), alpha, beta);
        }
    }

    gettimeofday (&tv2, NULL);
    timersub (&tv2, &tv1, &tv1);
    printf ( "      takes %.3le secs %.3lf GFlops %.3lf GB/s\n",
        tv1.tv_sec + tv1.tv_usec/1e6,
        flops/(tv1.tv_sec + tv1.tv_usec/1e6)/1e9,
        mops/(tv1.tv_sec + tv1.tv_usec/1e6)/1e9);
}


inline void influence_kernel (double *grid, int ld3,
                             double *B0, double *B1, double *B2,
                             double *B3, double *B4, double *B5)
{  
    // scalar kernel
    double real[3];
    double imag[3];    

    real[0] = grid[0 * ld3 + 0];
    imag[0] = grid[0 * ld3 + 1];
    real[1] = grid[1 * ld3 + 0];
    imag[1] = grid[1 * ld3 + 1];
    real[2] = grid[2 * ld3 + 0];
    imag[2] = grid[2 * ld3 + 1];
    grid[0 * ld3 + 0] =
            B0[0] * real[0] +
            B1[0] * real[1] +
            B2[0] * real[2];
    grid[0 * ld3 + 1] = 
            B0[0] * imag[0] +
            B1[0] * imag[1] +
            B2[0] * imag[2];
    grid[1 * ld3 + 0] = 
            B1[0] * real[0] +
            B3[0] * real[1] +
            B4[0] * real[2];
    grid[1 * ld3 + 1] = 
            B1[0] * imag[0] +
            B3[0] * imag[1] +
            B4[0] * imag[2];
    grid[2 * ld3 + 0] = 
            B2[0] * real[0] +
            B4[0] * real[1] +
            B5[0] * real[2];
    grid[2 * ld3 + 1] = 
            B2[0] * imag[0] +
            B4[0] * imag[1] +
            B5[0] * imag[2];
}


void apply_influence (int dim, double *grid,
                      int ld1, int ld2, int ld3,
                      double *map, double *lm2,
                      double **lB0, double **lB1, double **lB2,
                      double **lB3, double **lB4, double **lB5)
{
    struct timeval tv1;
    struct timeval tv2;
    printf ( "    applying influence ...\n");
    gettimeofday (&tv1, NULL);
    
    int x;
    int y;
    int z;
    int j;
    int i;
    int ld1c;
    int ld2c;
    int tid;
    double kx;
    double ky;
    double kz;
    double kk;
    double m2;
    double exx;
    double eyy;
    double ezz;
    double exy;
    double exz;
    double eyz;
    double *B0;
    double *B1;
    double *B2;
    double *B3;
    double *B4;
    double *B5;
    double *mm;
    double kyz;
    double flops;
    double mops;
    
    ld1c = ld1/2;
    ld2c = ld2/2;
    flops = 1.0 * ld3/2 * 36;
    mops = (double)ld3/2 * (8 + 12 * 8);
    
    // grid(x,y,z,:) = B(:,:,x,y,z)*squeeze(grid(x,y,z,:));    
    #pragma omp parallel default(none)\
                         shared (ld1c, ld2c, ld3, dim, grid,\
                                 map, lm2, lB0, lB1, lB2, lB3, lB4, lB5)\
                         private(tid, j, i, x, y, z, kx, ky, kz, kk,\
                                 m2, exx, eyy, ezz, exy, exz, eyz,\
                                 B0, B1, B2, B3, B4, B5, mm, kyz)
    {      
        tid = omp_get_thread_num();
        B0 = lB0[tid];
        B1 = lB1[tid];
        B2 = lB2[tid];
        B3 = lB3[tid];
        B4 = lB4[tid];
        B5 = lB5[tid];
        #pragma omp for schedule(static)
        for (j = 0; j < dim * dim; j++)
        {
            z = j/dim;
            y = j%dim;
            kz = map[z];
            ky = map[y];
            mm = &(lm2[j * ld1c]);
            kyz = ky*ky + kz*kz;
            #pragma simd
            for (x = 0; x <= dim/2; x++)
            {
                kx = map[x];
                m2 = mm[x];
                kk = 1.0/(kx*kx + kyz);
                exx = kx * kx * kk;
                eyy = ky * ky * kk;
                ezz = kz * kz * kk;
                exy = kx * ky * kk;
                eyz = ky * kz * kk;
                exz = kx * kz * kk;              
                B0[x] = m2 - m2 * exx; // 0
                B1[x] = -m2 * exy;     // 1 and 3
                B2[x] = -m2 * exz;     // 2 and 6
                B3[x] = m2 - m2 * eyy; // 4
                B4[x] = -m2 * eyz;     // 5 and 7
                B5[x] = m2 - m2 * ezz; // 8
            }
            for (x = 0; x <= dim/2; x+=1)
            {
                i = z * ld2c + y * ld1c + x;
                influence_kernel (&(grid[2*i]), ld3,
                                  &(B0[x]), &(B1[x]), &(B2[x]),
                                  &(B3[x]), &(B4[x]), &(B5[x]));
            }
        }
        if (tid == 0)
        {
            grid[0 * ld3 + 0] = 0.0;
            grid[0 * ld3 + 1] = 0.0;
            grid[1 * ld3 + 0] = 0.0;
            grid[1 * ld3 + 1] = 0.0;
            grid[2 * ld3 + 0] = 0.0;
            grid[2 * ld3 + 1] = 0.0;
        }
    }

    gettimeofday (&tv2, NULL);
    timersub (&tv2, &tv1, &tv1);
    printf ( "      takes %.3le secs %.3lf GFlops %.3lf GB/s\n",
        tv1.tv_sec + tv1.tv_usec/1e6,
        flops/(tv1.tv_sec + tv1.tv_usec/1e6)/1e9,
        mops/(tv1.tv_sec + tv1.tv_usec/1e6)/1e9);
}


inline void spread_kernel (int porder3, double *P, int *ind,
                           double *grid, int ld3, double *forces)
{
    int j;
    int idx;
    double pvalue;

    for (j = 0; j < porder3; j++)
    {
        idx = ind[j];
        pvalue = P[j];
        grid[0 * ld3 + idx] += pvalue * forces[0];
        grid[1 * ld3 + idx] += pvalue * forces[1];
        grid[2 * ld3 + idx] += pvalue * forces[2];
    }

}


void spread (int np, int nb, int *head, int *next,
             int porder3, double *P, int ldP, int *ind, int ldind,
             double *forces, double *grid, int ld3)
{
    struct timeval tv1;
    struct timeval tv2;
    double timepass;
    printf ( "    srpead ...\n");
    gettimeofday (&tv1, NULL);

    int i;
    int k;
    char set;
    int nb3;
    int n8;
    double mops;

    nb3 = nb * nb * nb;
    n8 = nb3 / 8;
    mops = 8.0 * ld3 * 6 + 36.0 * np * porder3;
    
    #pragma omp parallel for
    for (i = 0; i < ld3 * 3; i++)
    {
        grid[i] = 0.0;
    }

    for (set = 0; set < 8; set++)
    {
        #pragma omp parallel for private (k, i)
        for (k = n8 * set; k < (set + 1) * n8; k++)
        {
            i = head[k];
            while (i != -1)
            {
                spread_kernel (porder3, &(P[i * ldP]), &(ind[i * ldind]),
                               grid, ld3, &(forces[i * 3]));
                i = next[i];
            }
        }
    }

/*
	for (i = 0; i < np; i++)
	{
    	spread_kernel (porder3, &(P[i * ldP]), &(ind[i * ldind]),
                       grid, ld3, &(forces[i * 3]));  
    }
*/

/*
	FILE *fp = fopen("./verify/c-2-spread", "w");
	for (i = 0; i < ld3*3; ++i)
		fprintf(fp, "%.17lg\n", grid[i]);
	fclose(fp);
*/

    gettimeofday (&tv2, NULL);
    timersub (&tv2, &tv1, &tv1);
    timepass = tv1.tv_sec + tv1.tv_usec/1e6;
    printf ( "      takes %.3le secs, %.3lf GB/s\n",
       timepass, mops/timepass/1e9);
}
