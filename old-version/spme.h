#ifndef __SPME_H__
#define __SPME_H__

#include <mkl.h>
#include <cufft.h>

typedef struct _gpu_spme_t
{
	double 	xi;
	double	box_size;
	int		np;

	double*	P;		
	int		ldP;
	int*	ind;
	int		ldind;
	int		porder;

	double* grid;
	int 	dim;
	int 	ld1;
	int		ld2;
	int		ld3;

	double*	map;
	double* lm2;

    // for spread    
    int nb;
    int sizeb;
    int *head;    /* nb * nb *nb */
    int *bidx;    /* nb * nb *nb */
    int *pidx;    /* np */
    int *next;    /* np */

	cufftHandle		cufwplan[3];
	cufftHandle 	cubwplan[3];
	cudaStream_t	custream[3];

} gpu_spme_t;

typedef struct _spme_t
{
    int nthreads;
    double xi; 
    double *pos;
    int np;
    double box_size;

    double *map;
    double *lm2;
    double **lB0;
    double **lB1;
    double **lB2;
    double **lB3;
    double **lB4;
    double **lB5;
    
    double *P;   /* porder3*np */
    int ldP;     /* = porder3  */
    int *ind;
    int ldind;
    int porder;
    double **qbuf;
    
    double *grid;  
    int dim;
    int ld1;
    int ld2;
    int ld3;

    // for spread    
    int nb;
    int sizeb;
    int *head;    /* nb * nb *nb */
    int *bidx;    /* nb * nb *nb */
    int *pidx;    /* np */
    int *next;    /* np */
     
    DFTI_DESCRIPTOR_HANDLE bwhandle;
    DFTI_DESCRIPTOR_HANDLE fwhandle;
} spme_t;


void cutoff2xi (double rmax, double ewald_eps,
                double box_size, double *_xi);


void create_spme_engine (double xi, int dim, int porder,
                         int np, double box_size,
                         int nthreads, spme_t **_spme);

void destroy_spme_engine (spme_t *spme);

void compute_spme (spme_t *spme, double *pos, int nrhs,
                   double alpha, double *vec_in, int ldin,
                   double beta, double *vec_out, int ldout);


#define PAD_LEN(N,size)     ( ((N+(64/size)-1)/(64/size))     * (64/size) )
#define PAD_FFT_LEN(N,size) ( (((N+(64/size)-1)/(64/size))|1) * (64/size) )


#endif /* __SPME_H__ */
