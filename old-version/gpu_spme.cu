#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>

#include <sys/time.h>
#include <mkl.h>
#include <cufft.h>

#include "gpu_spme.cuh"

#define GPU_SYNC

#define GPU_THREADS 		128
#define HARD_CODEDE_LDP		8

__constant__ double d_splineval[5];
__constant__ double d_W[6*6];

static double gpu_tab_splines[6][5] = 
{
	{0.0},
	{0.0},
	{0.0},
	{1.0/6.0, 4.0/6.0, 1.0/6.0},
	{0.0},
	{1.0/120.0, 26.0/120.0, 66.0/120.0, 26.0/120.0, 1.0/120.0}
};

static double gpu_tab_W[6][6*6]= {
  {0.0},
  {0.0},
  {0.0},
  { 1.0/48.0,  -6.0/48.0,  12.0/48.0,  -8.0/48.0,
   23.0/48.0, -30.0/48.0, -12.0/48.0,  24.0/48.0,
   23.0/48.0,  30.0/48.0, -12.0/48.0, -24.0/48.0,
    1.0/48.0,   6.0/48.0,  12.0/48.0,   8.0/48.0},
  {0.0},
  {   1.0/3840.0,   -10.0/3840.0,   40.0/3840.0,
	-80.0/3840.0,    80.0/3840.0,  -32.0/3840.0,
    237.0/3840.0,  -750.0/3840.0,  840.0/3840.0,
   -240.0/3840.0,  -240.0/3840.0,  160.0/3840.0,
   1682.0/3840.0, -1540.0/3840.0, -880.0/3840.0,
   1120.0/3840.0,   160.0/3840.0, -320.0/3840.0,
   1682.0/3840.0,  1540.0/3840.0, -880.0/3840.0,
  -1120.0/3840.0,   160.0/3840.0,  320.0/3840.0,
    237.0/3840.0,   750.0/3840.0,  840.0/3840.0,
	240.0/3840.0,  -240.0/3840.0, -160.0/3840.0,
      1.0/3840.0,    10.0/3840.0,   40.0/3840.0,
	 80.0/3840.0,    80.0/3840.0,   32.0/3840.0}
};

double* gpu_f;		// global input
double* gpu_v;		// global output
double* gpu_pos;

extern "C" void
gpu_create_spme_engine (double xi, int dim, int porder, int np,
						double box_size, gpu_spme_t **_spme, const spme_t* s) 
{
	gpu_spme_t *spme;
	spme			= (gpu_spme_t *)malloc(sizeof(gpu_spme_t));
	assert(spme);

	int pad_dim1;
	int pad_dim2;
	pad_dim1		= PAD_FFT_LEN(dim, sizeof(double));
	pad_dim2		= (dim / 2 + 1) * 2;
	pad_dim2		= PAD_FFT_LEN(pad_dim2, sizeof(double));
	spme->dim		= dim;
	spme->ld1		= pad_dim2;
	spme->ld2		= pad_dim1 * pad_dim2;
	spme->ld3		= dim * pad_dim1 * pad_dim2;
	spme->porder	= porder;
	spme->box_size	= box_size;
	spme->np		= np;

	int size[]		= {dim, dim, dim};
	int inembed[]	= {dim, pad_dim1, pad_dim2};
	int idist		= dim * pad_dim1 * pad_dim2;
	int onembed[]	= {dim, pad_dim1, pad_dim2/2};
	int odist		= dim * pad_dim1 * pad_dim2 / 2;

	typedef cufftDoubleReal DReal;
	int BYTES		= sizeof(DReal) * (dim*pad_dim1*pad_dim2) * 3;
	CUDA_CHECK(cudaMalloc((void**)&spme->grid, BYTES));

	for (int i = 0; i < 3; ++i) {
		CUFFT_CHECK	(cufftPlanMany(&spme->cufwplan[i], 3, size, inembed, 1, 
									idist, onembed, 1, odist, CUFFT_D2Z, 1));
		CUFFT_CHECK (cufftPlanMany(&spme->cubwplan[i], 3, size, onembed, 1,
									odist, inembed, 1, idist, CUFFT_Z2D, 1));
		CUDA_CHECK	(cudaStreamCreate(&spme->custream[i]));
		CUFFT_CHECK	(cufftSetStream(spme->cufwplan[i], spme->custream[i]));
		CUFFT_CHECK (cufftSetStream(spme->cubwplan[i], spme->custream[i]));
	}

	spme->xi = xi;
	/* compute spline */
	gpu_compute_spline(xi, dim, porder, box_size, spme->ld1, spme->ld2,
					   &(spme->map), &(spme->lm2));

	/* sparse P */
	int porder2		= porder * porder;
	int porder3		= porder * porder2;	
	spme->ldP		= PAD_LEN(porder3, sizeof(double));
	spme->ldind		= PAD_LEN(porder3, sizeof(int));
	size_t size_P	= sizeof(double) * spme->ldP * np;
	size_t size_ind	= sizeof(int) * spme->ldind * np;
	CUDA_CHECK(cudaMalloc((void**)&(spme->P), size_P));
	CUDA_CHECK(cudaMalloc((void**)&(spme->ind), size_ind));

	/* init independent set */
	spme->sizeb		= spme->porder;
	int nb			= (spme->dim + spme->sizeb - 1) / spme->sizeb;
	spme->nb		= nb;
	assert(nb >= 4);
	size_t size_np	= sizeof(int) * np;
	size_t size_nb3	= sizeof(int) * nb * nb * nb;
	CUDA_CHECK(cudaMalloc((void**)&spme->next, size_np));
	CUDA_CHECK(cudaMalloc((void**)&spme->pidx, size_np));
	CUDA_CHECK(cudaMalloc((void**)&spme->bidx, size_nb3));
	CUDA_CHECK(cudaMalloc((void**)&spme->head, size_nb3));

	// dependency
	CUDA_CHECK(cudaMemcpy(spme->bidx, s->bidx, 
						  size_nb3, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(spme->grid, 0, BYTES));

	*_spme = spme;
}

extern "C" void
gpu_destroy_spme_engine (gpu_spme_t *spme)
{
	CUDA_CHECK(cudaFree(spme->grid));
	CUDA_CHECK(cudaFree(spme->map));
	CUDA_CHECK(cudaFree(spme->lm2));
	for (int i = 0; i < 3; ++i) {
		CUDA_CHECK(cudaStreamDestroy(spme->custream[i]));	
		CUFFT_CHECK(cufftDestroy(spme->cufwplan[i]));
	}
	free(spme);
}

extern "C" void
gpu_allocate(int ldm, int c, int np, double *f, double *v, double *pos)
{
	size_t fv_size = sizeof(double)*ldm*c;
	size_t pos_size = sizeof(double)*np*3 + 8;
	CUDA_CHECK(cudaHostRegister(f, fv_size, cudaHostRegisterPortable));
	CUDA_CHECK(cudaHostRegister(v, fv_size, cudaHostRegisterPortable));
	CUDA_CHECK(cudaHostRegister(pos, pos_size, cudaHostRegisterPortable));
	CUDA_CHECK(cudaMalloc((void**)&gpu_f, fv_size));
	CUDA_CHECK(cudaMalloc((void**)&gpu_v, fv_size));
	CUDA_CHECK(cudaMalloc((void**)&gpu_pos, pos_size));
	CUDA_CHECK(cudaMemcpy(gpu_f, f, fv_size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(gpu_v, v, fv_size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(gpu_pos, pos, pos_size, cudaMemcpyHostToDevice));
}

extern "C" void
gpu_deallocate(double *f, double *v, double *pos)
{
	CUDA_CHECK(cudaHostUnregister(f));
	CUDA_CHECK(cudaHostUnregister(v));
	CUDA_CHECK(cudaHostUnregister(pos));
	CUDA_CHECK(cudaFree(gpu_f));
	CUDA_CHECK(cudaFree(gpu_v));
	CUDA_CHECK(cudaFree(gpu_pos));
}

extern "C" void
gpu_compute_spline (double xi, int dim, int porder, double box_size,
					int ld1, int ld2, double **map_, double **lm2_) 
{
	double 	*map;
	double 	*lm2;
	Complex *b;

	assert(porder == 4 || porder == 6);
	CUDA_CHECK(cudaMemcpyToSymbol(d_splineval, gpu_tab_splines[porder-1],
			   sizeof(double)*5, 0, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc((void**)&map, sizeof(double)*dim));
	CUDA_CHECK(cudaMalloc((void**)&lm2, sizeof(double)*dim*ld2/2));
	CUDA_CHECK(cudaMalloc((void**)&b,	sizeof(Complex)*dim));
	
	CUDA_CHECK(cudaMemset(lm2, 0, sizeof(double)*dim*ld2/2));
	
	kel_cp_spline_1<<<2, 64>>>(map, dim/2, box_size);
	kel_cp_spline_2<<<2, 64>>>(map, dim/2, box_size);
	kel_cp_spline_3<<<2, 64>>>(map, dim, b, porder, box_size);
	kel_cp_spline_4<<<256, 128>>>(map, b, lm2, xi, ld1, dim);
	// kel_cp_spline_5<<<512, 64>>>(map, b, lm2, xi, ld1, dim);

	*map_ 	= map;
	*lm2_	= lm2;

	CUDA_CHECK(cudaFree(b));
}

extern "C" void
gpu_compute_spme (gpu_spme_t *spme, double *h_pos, int nrhs,
				  double alpha, int in_idx, int ldin,
				  double beta, int out_idx, int ldout)
{
	timeval tv1, 	tv2,	tv3,	tv4;
	double	timepass;
	printf("    gpu Computing spme ...\n");
	gettimeofday(&tv1, NULL);

	int 	ld1, 	ld2, 	ld3;
	int 	dim, 	dim2, 	dim3;
	int 	porder, porder3;
	int 	np,		ldind,	ldP;
	int*	ind;
	int		nn;
	double 	box_size;
	double*	grid;
	double* P;
	double* _vec_in;
	double* _vec_out;
	double	_alpha;
	double	flops;

	np			= spme->np;
	box_size	= spme->box_size;
	ld1			= spme->ld1;
	ld2			= spme->ld2;
	ld3			= spme->ld3;
	dim			= spme->dim;
	dim2		= dim * dim;
	dim3		= dim * dim2;
	porder		= spme->porder;
	porder3		= porder * porder * porder;
	P			= spme->P;
	ind			= spme->ind;
	ldP			= spme->ldP;
	ldind		= spme->ldind;
	grid		= spme->grid;

	CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
	
	printf("  GPU Computing spme ...\n");
	if (h_pos != NULL) {
		gpu_compute_P(dim, porder, gpu_pos, np, box_size, ld1, ld2,
					  P, ldP, ind, ldind, spme->head, spme->next, spme->bidx,
					  spme->pidx, spme->nb, spme->sizeb);
	}

	for (nn = 0; nn < nrhs; ++nn) {
		_vec_in 	= gpu_f + (in_idx+nn) * ldin;
		_vec_out	= gpu_v + (out_idx+nn) * ldout;
		flops = 1 * 3.0 * 2.5 * dim3 * log2(1.0 * dim3);

		// interpolate_spread
		gpu_spread(np, spme->nb, porder3, P, ldP, 
				   ind, ldind, _vec_in, grid, ld3);

		// forward fft
		printf("    gpu forward fft ...\n");
		gettimeofday(&tv3, NULL);

		const int size = spme->ld3;
		for (int i = 0; i < 3; ++i) {
			CUFFT_CHECK(cufftExecD2Z(spme->cufwplan[i],
						(cufftDoubleReal*)(spme->grid+i*size), 
						(cufftDoubleComplex*)(spme->grid+i*size)));
		}

#ifdef GPU_SYNC
		CUDA_CHECK(cudaDeviceSynchronize());
#endif
		gettimeofday(&tv4, NULL);
		timersub (&tv4, &tv3, &tv3);
        timepass = tv3.tv_sec + tv3.tv_usec/1e6;
        printf ( "      takes %.3le secs %.3lf GFlops\n",
            timepass, flops/timepass/1e9);

		// apply influence
		gpu_apply_influence(dim, spme->grid, ld1, ld2, ld3, 
							spme->map, spme->lm2);

		// backward fft
		printf("    gpu backward fft ...\n");
		gettimeofday(&tv3, NULL);

		for (int i = 0; i < 3; ++i) {
			CUFFT_CHECK(cufftExecZ2D(spme->cubwplan[i],
						(cufftDoubleComplex*)(spme->grid+i*size), 
						(cufftDoubleReal*)(spme->grid+i*size)));
		}

#ifdef GPU_SYNC
		CUDA_CHECK(cudaDeviceSynchronize());
#endif
		gettimeofday(&tv4, NULL);
        timersub (&tv4, &tv3, &tv3);
        timepass = tv3.tv_sec + tv3.tv_usec/1e6;
        printf ( "      takes %.3le secs %.3lf GFlops\n",
            timepass, flops/timepass/1e9);

		for (int i = 0; i < 3; ++i) {
			kel_fft_scalling<<<120, 128>>>(spme->grid+i*size,
										  size,
										  1.0/(dim * dim * dim));
		}
#ifdef GPU_SYNC
		CUDA_CHECK(cudaDeviceSynchronize());
#endif

		// interpolate
		_alpha = alpha * ((double)dim*dim*dim) / (box_size*box_size*box_size);
		gpu_interpolate(np, porder3, P, ldP, ind, ldind, grid, 
						ld3, _vec_out, _alpha, beta);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	gettimeofday(&tv2, NULL);
	timersub(&tv2, &tv1, &tv1);
    timepass = tv1.tv_sec + tv1.tv_usec/1e6;
	printf("    totally takes %.3le secs\n", timepass);

/*
	double *h_vec_out;
	int BYTES = sizeof(double) * nrhs * ldout;
	CUDA_CHECK(cudaHostAlloc((void**)&h_vec_out, BYTES, cudaHostAllocDefault));
	CUDA_CHECK(cudaMemcpy(h_vec_out, _vec_out, BYTES, cudaMemcpyDeviceToHost));
	FILE *fp = fopen("./verify/g-vec-out.bin", "wb");
	//
	for (int i = 0; i < nrhs * ldout; ++i)
		fprintf(fp, "%lg\n", h_vec_out[i]);
	//
	fwrite(h_vec_out, sizeof(double), nrhs*ldout, fp);
	fclose(fp);
	CUDA_CHECK(cudaFreeHost(h_vec_out));
*/
}

extern "C" void
gpu_compute_P (int dim, int porder, double *pos, int np, double box_size,
			   int ld1, int ld2, double *P, int ldP, int *ind, int ldind,
			   int *head, int *next, int *bidx, int *pidx, int nb, int sizeb)
{
	timeval tv1;
	timeval tv2;
	printf("    gpu compute P ...\n");
	gettimeofday(&tv1, NULL);

	int BYTES = sizeof(double) * (6*6); 
	CUDA_CHECK(cudaMemcpyToSymbol(d_W, gpu_tab_W[porder-1], BYTES,
								  0, cudaMemcpyHostToDevice));

	int BLOCKS = 32;
	kel_cp_P_1<<<BLOCKS, GPU_THREADS>>>(np, dim, box_size, porder, ldP,
		pos, sizeb, pidx, bidx, nb, P, ind, ldind, ld1, ld2);

#ifdef GPU_SYNC
	CUDA_CHECK(cudaDeviceSynchronize());
#endif
	gettimeofday(&tv2, NULL);
	timersub(&tv2, &tv1, &tv1);
	printf("      takes %.3le secs\n", tv1.tv_sec + tv1.tv_usec/1e6);

/*
	FILE *fp1 = fopen("./verify/g-ind", "w");
	int *h_ind = (int*)malloc(sizeof(int)*np*ldind);
	CUDA_CHECK(cudaMemcpy(h_ind, ind, sizeof(int)*np*ldind, cudaMemcpyDeviceToHost));
	for (int i = 0; i < np * ldind; ++i)
		fprintf(fp1, "%d\n", h_ind[i]);
	fclose(fp1);
	free(h_ind);
*/
}

extern "C" void
gpu_spread (int np, int nb, int porder3, double *P, int ldP, int *ind,
			int ldind, double *forces, double *grid, int ld3) {
	timeval tv1;
	timeval tv2;
	double timepass;
	printf("    spread ...\n");

	double mops = 8.0 * ld3 * 6 + 36.0 * np * porder3;

	gettimeofday(&tv1, NULL);
	CUDA_CHECK(cudaMemset(grid, 0.0, sizeof(double)*ld3*3));

	kel_spread_1<<<90, 64>>>(porder3, P, ldP, ind, ldind, 
							  np, grid, ld3, forces);
#ifdef GPU_SYNC
	CUDA_CHECK(cudaDeviceSynchronize());
#endif
	gettimeofday(&tv2, NULL);
	timersub(&tv2, &tv1, &tv1);
	timepass = tv1.tv_sec + tv1.tv_usec/1e6;
    printf ( "      takes %.3le secs, %.3lf GB/s\n",
       timepass, mops/timepass/1e9);
}

extern "C" void
gpu_apply_influence (int dim, double *grid, int ld1, int ld2, 
					 int ld3, double *map, double *lm2)
{
	timeval tv1, tv2;
	printf("    gpu applying influence ...\n");
	gettimeofday(&tv1, NULL);

	int ld1c = ld1 / 2;
	int ld2c = ld2 / 2;
    double flops = 1.0 * ld3/2 * 36;
    double mops = (double)ld3/2 * (8 + 12 * 8);
	// kel_ap_influence_1<<<210, 64>>>(map, lm2, dim, ld1c, ld2c, grid, ld3);
	// kel_ap_influence_2<<<135, 128>>>(map, lm2, dim, ld1c, ld2c, grid, ld3);
	// kel_ap_influence_3<<<240, 64>>>(map, lm2, dim, ld1c, ld2c, grid, ld3);
	kel_ap_influence_4<<<240, 64>>>(map, lm2, dim, ld1c, ld2c, grid, ld3);
#ifdef GPU_SYNC
	CUDA_CHECK(cudaDeviceSynchronize());
#endif

	gettimeofday(&tv2, NULL);
	timersub(&tv2, &tv1, &tv1);
    printf( "      takes %.3le secs %.3lf GFlops %.3lf GB/s\n",
        tv1.tv_sec + tv1.tv_usec/1e6,
        flops/(tv1.tv_sec + tv1.tv_usec/1e6)/1e9,
        mops/(tv1.tv_sec + tv1.tv_usec/1e6)/1e9);

/*
	FILE *fp = fopen("./verify/grid-5", "w");
	double *h_grid = (double*)malloc(sizeof(double)*ld3*3);
	CUDA_CHECK(cudaMemcpy(h_grid, grid, sizeof(double)*ld3*3, cudaMemcpyDeviceToHost));
	for (int i = 0; i < ld3*3; ++i)
		fprintf(fp, "%lg\n", h_grid[i]);
	free(h_grid);
	fclose(fp);
*/

}

extern "C" void
gpu_interpolate (int np, int porder3, double *P, int ldP, int *ind, int ldind,
				 double *grid, int ld3, double *vels, double alpha, double beta)
{
	timeval tv1, tv2;
	printf("    interpolate ...\n");
	gettimeofday(&tv1, NULL);

	double flops	= (double)2.0 * np * porder3 * 3;
	double mops 	= (double)np * porder3 * (4 + 8 * 4);
/*
	kel_interpolate<<<64, 128>>>(np, porder3, P, ldP, ind, ldind, grid,
								 ld3, vels, alpha, beta);
*/
	kel_interpolate_2<<<210, 64, sizeof(double)*64*3>>>
		(np, porder3, P, ldP, ind, ldind, grid, ld3, vels, alpha, beta);

#ifdef GPU_SYNC
	CUDA_CHECK(cudaDeviceSynchronize());
#endif
	gettimeofday(&tv2, NULL);
	timersub(&tv2, &tv1, &tv1);
    printf( "      takes %.3le secs %.3lf GFlops %.3lf GB/s\n",
        tv1.tv_sec + tv1.tv_usec/1e6,
        flops/(tv1.tv_sec + tv1.tv_usec/1e6)/1e9,
        mops/(tv1.tv_sec + tv1.tv_usec/1e6)/1e9);

/*
	FILE *fp1 = fopen("./verify/ind", "w");
	int *h_ind = (int*)malloc(sizeof(int)*np*ldind);
	CUDA_CHECK(cudaMemcpy(h_ind, ind, sizeof(int)*np*ldind, cudaMemcpyDeviceToHost));
	for (int i = 0; i < np * ldind; ++i)
		fprintf(fp1, "%d\n", h_ind[i]);
	fclose(fp1);
	free(h_ind);
*/

/*
	FILE *fp = fopen("./verify/vels-2", "w");
	double *h_vels = (double *)malloc(sizeof(double)*np*3);
	CUDA_CHECK(cudaMemcpy(h_vels, vels, sizeof(double)*np*3, cudaMemcpyDeviceToHost));
	for (int i = 0; i < np * 3; ++i)
		fprintf(fp, "%lg\n", h_vels[i]);
	fclose(fp);
	free(h_vels);
*/
}

typedef unsigned long long int	ULLI;

__device__ double atomicAdd(double* address, double val)
{
	ULLI* address_as_ull = (ULLI*)address;
	ULLI old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
						__double_as_longlong(val +
						__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void kel_cp_spline_1(double *map, int half_dim, double box_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tn	= blockDim.x * gridDim.x;
	for (int i = tid; i < half_dim; i += tn)
		map[i] = i * 2.0 * M_PI / box_size;
}

__global__ void kel_cp_spline_2(double *map, int half_dim, double box_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tn	= blockDim.x * gridDim.x;
	for (int i = tid; i < half_dim; i += tn)
		map[i+half_dim] = (-half_dim + i) * 2.0 * M_PI / box_size;
	if (tid == 0)
		map[half_dim] = -map[half_dim];
}

__global__ void kel_cp_spline_3(double *map, int dim, Complex *b,
								int porder, double box_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tn	= blockDim.x * gridDim.x;
	double kx, phi, m2;
	Complex den, btmp;

	for (int x = tid; x < dim; x += tn) {
		kx = map[x];
		den.x = 0.0;
		den.y = 0.0;
		for (int i = 0; i < porder-1; ++i) {
			phi = i*kx*box_size/dim;
			den.x += d_splineval[i]*cos(phi);
			den.y += d_splineval[i]*sin(phi);
		}
		phi = box_size*kx*(porder-1)/dim;
		btmp.x = cos(phi);
		btmp.y = sin(phi);
		m2 = den.x*den.x + den.y*den.y;
		b[x].x = (btmp.x*den.x + btmp.y*den.y)/m2;
		b[x].y = (btmp.y*den.x - btmp.x*den.y)/m2;
	}
}

__device__ double dev_scalar_recip(double k, double xi, double aa, double ab)
{
	double kx2, k2, v, aa2, ab2;

	kx2 = (k/xi) * (k/xi);
	k2 	= k * k;
	aa2	= aa * aa;
	ab2 = ab * ab;
	v	= 6.0 * M_PI * exp(-kx2/4.0) * (1.0 - k2*(aa2+ab2)/6.0) /
		  k2 * (1.0 + kx2*(1.0/4.0 + kx2/8.0));
	return v;
}

__global__ void kel_cp_spline_4(double *map, Complex *b, double *lm2,
								double xi, int ld1, int dim)
{
	int		z, y;
	double	kz, ky, kx, kk;
	double	m2;
	int		tid = threadIdx.x + blockIdx.x * blockDim.x;
	int		tn	= blockDim.x * gridDim.x;
	Complex	btmp, bbb;
	for (int j = tid; j < dim * dim; j += tn) {
		z 	= j/dim;
		y 	= j%dim;
		kz 	= map[z];
		ky	= map[y];
		for (int x = 0; x <= dim/2; ++x) {
			for (int i = 0; i < 1; ++i) {
				kx 		= map[x+i];
				kk 		= sqrt(kx*kx + ky*ky + kz*kz);
				m2		= dev_scalar_recip(kk, xi, 1.0, 1.0);
				btmp.x	= b[x+i].x*b[y].x - b[x+i].y*b[y].y;
				btmp.y	= b[x+i].x*b[y].y + b[x+i].y*b[y].x;
				bbb.x	= btmp.x*b[z].x - btmp.y*b[z].y;
				bbb.y	= btmp.x*b[z].y + btmp.y*b[z].x;
				m2		= m2 * (bbb.x*bbb.x + bbb.y*bbb.y);
				lm2[j*ld1/2+x+i] = m2;
			}
		}
	}
}

__global__ void kel_cp_spline_5(double *map, Complex *b, double *lm2,
								double xi, int ld1, int dim)
{
	int		z, y;
	double	kz, ky, kx, kk;
	double	m2;
	int 	bid = blockIdx.x;
	int		bn = gridDim.x;
	int		tid = threadIdx.x;
	int		tn = blockDim.x;
	Complex	btmp, bbb;
	for (int j = bid; j < dim * dim; j += bn) {
		z = j / dim;
		y = j % dim;
		kz = map[z];
		ky = map[y];
		for (int x = tid; x <= dim/2; x += tn) {
			kx = map[x];
			kk = sqrt(kx*kx + ky*ky + kz*kz);
			m2 = dev_scalar_recip(kk, xi, 1.0, 1.0);
			btmp.x = b[x].x * b[y].x - b[x].y * b[y].y;
			btmp.y = b[x].x * b[y].y - b[x].y * b[y].x;
			bbb.x = btmp.x * b[z].x - btmp.y * b[z].y;
			bbb.y = btmp.x * b[z].y + btmp.y * b[z].x;
			m2 = m2 * (bbb.x*bbb.x + bbb.y*bbb.y);
			lm2[j*ld1/2+x] = m2;
		}
	}
}

__global__ void kel_cp_P_1(int np, int dim, double box_size, 
						   int porder, int ldP, double *pos, int sizeb,
						   int *pidx, int *bidx, int nb, double *P,
						   int *ind, int ldind, int ld1, int ld2)
{
	__shared__ double buffer[GPU_THREADS][6][HARD_CODEDE_LDP];
	int tid 	= threadIdx.x + blockIdx.x * blockDim.x;
	int tn		= blockDim.x * gridDim.x;
	double *p1 	= buffer[threadIdx.x][0]; 
	double *p2	= buffer[threadIdx.x][1];
	double *p3	= buffer[threadIdx.x][2];
	double *q1	= buffer[threadIdx.x][3];
	double *q2	= buffer[threadIdx.x][4];
	double *q3	= buffer[threadIdx.x][5];
	double cx, cy, cz;
	int bx, by, bz;
	int indx, indy, indz;
	int nb2 = nb * nb;
	int porder2 = porder * porder;
	int zz, yy, xx;
	double g[3];
	
	for (int i = tid; i < np; i += tn) {
		cx = pos[i * 3 + 0]/box_size * dim;
		cy = pos[i * 3 + 1]/box_size * dim;
		cz = pos[i * 3 + 2]/box_size * dim;
		bx = (int)(cx/sizeb);
		by = (int)(cy/sizeb);
		bz = (int)(cz/sizeb);
		pidx[i] = bidx[bz * nb2 + by * nb + bx];
		

		g[0] = floor(cx);
		g[1] = floor(cy);
		g[2] = floor(cz);

		indx = (int)(g[0] - porder/2 + 1);
		indx = (indx + dim) % dim;
		indy = (int)(g[1] - porder/2 + 1);
		indy = (indy + dim) % dim;
		indz = (int)(g[2] - porder/2 + 1);
		indz = (indz + dim) % dim;

		g[0] = cx - g[0] - 0.5;
		g[1] = cy - g[1] - 0.5;
		g[2] = cz - g[2] - 0.5;

		p1[0] = 1.0;
		p2[0] = 1.0;
		p3[0] = 1.0;

		for (int x = 1; x < porder; ++x) {
			p1[x] = g[0] * p1[x - 1];
			p2[x] = g[1] * p2[x - 1];
			p3[x] = g[2] * p3[x - 1];
		}

		for (int x = 0; x < porder; ++x) {
			q1[x] = 0.0;
			q2[x] = 0.0;
			q3[x] = 0.0;
			for (int y = 0; y < porder; ++y) {
				q1[x] += d_W[x * porder + y] * p1[y];
				q2[x] += d_W[x * porder + y] * p2[y];
				q3[x] += d_W[x * porder + y] * p3[y];
			}
		}

		for (int z = 0; z < porder; ++z) {
			zz = (z + indz) % dim;
			for (int y = 0; y < porder; ++y) {
				yy = (y + indy) % dim;
				for (int x = 0; x < porder; ++x) {
					xx = (x + indx) % dim;
					P[i * ldP + z * porder2 + y * porder + x]
						= q1[x] * q2[y] * q3[z];
					ind[i * ldind + z * porder2 + y * porder + x]
						= zz * ld2 + yy * ld1 + xx;
				}
			}
		}
	}
}

__global__ void kel_spread_1(int porder3, double *P, int ldP, 
							 int *ind, int ldind, int np, double *grid, 
							 int ld3, double *forces)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tn	= blockDim.x * gridDim.x;

	for (int i = tid; i < np; i += tn) {
		dev_spread_kernel(porder3, P+i*ldP, ind+i*ldind, grid, ld3, forces+i*3);
	}
}

__device__ void dev_spread_kernel(int porder3, double *P, int *ind, 
								  double *grid, int ld3, double *forces)
{
	int idx;
	double pvalue;
	
	for (int j = 0; j < porder3; ++j) {
		idx = ind[j];
		pvalue = P[j];
		if (idx >= ld3)
			printf("> ld3\n");
		/*
		grid[0 * ld3 + idx] += pvalue * forces[0];
		grid[1 * ld3 + idx] += pvalue * forces[1];
		grid[2 * ld3 + idx] += pvalue * forces[2];
		*/
		atomicAdd(&grid[0 * ld3 + idx], pvalue * forces[0]);
		atomicAdd(&grid[1 * ld3 + idx], pvalue * forces[1]);
		atomicAdd(&grid[2 * ld3 + idx], pvalue * forces[2]);
	}
}

__device__ void dev_influence_kernel
	(double *grid, int ld3, double B0, double B1, double B2, double B3, 
	 double B4, double B5) 
{
/*
	double real[3];
	double imag[3];

	real[0] = grid[0 * ld3 + 0];
	imag[0] = grid[0 * ld3 + 1];
	real[1] = grid[1 * ld3 + 0];
	imag[1] = grid[1 * ld3 + 1];
	real[2] = grid[2 * ld3 + 0];
	imag[2] = grid[2 * ld3 + 1];

	grid[0 * ld3 + 0] = B0*real[0]+B1*real[1]+B2*real[2];
	grid[0 * ld3 + 1] = B0*imag[0]+B1*imag[1]+B2*imag[2];
	grid[1 * ld3 + 0] = B1*real[0]+B3*real[1]+B4*real[2];
	grid[1 * ld3 + 1] = B1*imag[0]+B3*imag[1]+B4*imag[2];
	grid[2 * ld3 + 0] = B2*real[0]+B4*real[1]+B5*real[2];
	grid[2 * ld3 + 1] = B2*imag[0]+B4*imag[1]+B5*imag[2];
*/

/*
	double real0, real1, real2;
	double imag0, imag1, imag2;

	real0 = grid[0];
	imag0 = grid[1];
	real1 = grid[ld3];
	imag1 = grid[ld3+1];
	real2 = grid[ld3+ld3];
	imag2 = grid[ld3+ld3+1];

	grid[0] = B0*real0+B1*real1+B2*real2;
	grid[1] = B0*imag0+B1*imag1+B2*imag2;
	grid[ld3] = B1*real0+B3*real1+B4*real2;
	grid[ld3+1] = B1*imag0+B3*imag1+B4*imag2;
	grid[ld3+ld3] = B2*real0+B4*real1+B5*real2;
	grid[ld3+ld3+1] = B2*imag0+B4*imag1+B5*imag2;
*/

	double cplx0, cplx1, cplx2;

	cplx0 = grid[0];
	cplx1 = grid[ld3];
	cplx2 = grid[ld3+ld3];
	grid[0] = B0*cplx0+B1*cplx1+B2*cplx2;
	grid[ld3] = B1*cplx0+B3*cplx1+B4*cplx2;
	grid[ld3+ld3] = B2*cplx0+B4*cplx1+B5*cplx2;

	cplx0 = grid[1];
	cplx1 = grid[ld3+1];
	cplx2 = grid[ld3+ld3+1];
	grid[1] = B0*cplx0+B1*cplx1+B2*cplx2;
	grid[ld3+1] = B1*cplx0+B3*cplx1+B4*cplx2;
	grid[ld3+ld3+1] = B2*cplx0+B4*cplx1+B5*cplx2;
}

__device__ void dev_influence_kernel_2
	(double *grid, int ld3, double B0, double B1, double B2, double B3, 
	 double B4, double B5) 
{
/*
	double real[3];
	double imag[3];

	real[0] = grid[0 * ld3 + 0];
	imag[0] = grid[0 * ld3 + 1];
	real[1] = grid[1 * ld3 + 0];
	imag[1] = grid[1 * ld3 + 1];
	real[2] = grid[2 * ld3 + 0];
	imag[2] = grid[2 * ld3 + 1];

	grid[0 * ld3 + 0] = B0*real[0]+B1*real[1]+B2*real[2];
	grid[0 * ld3 + 1] = B0*imag[0]+B1*imag[1]+B2*imag[2];
	grid[1 * ld3 + 0] = B1*real[0]+B3*real[1]+B4*real[2];
	grid[1 * ld3 + 1] = B1*imag[0]+B3*imag[1]+B4*imag[2];
	grid[2 * ld3 + 0] = B2*real[0]+B4*real[1]+B5*real[2];
	grid[2 * ld3 + 1] = B2*imag[0]+B4*imag[1]+B5*imag[2];
*/
	double reim0, reim1, reim2;

	reim0 = grid[0];
	reim1 = grid[ld3];
	reim2 = grid[ld3+ld3];

	grid[0] = B0*reim0+B1*reim1+B2*reim2;
	grid[ld3] = B1*reim0+B3*reim1+B4*reim2;
	grid[ld3+ld3] = B2*reim0+B4*reim1+B5*reim2;
}

__global__ void kel_ap_influence_1(double *map, double *lm2, int dim, int ld1c,
				   				   int ld2c, double *grid, int ld3)
{
	int z, y;
	double kz, ky, kx, kyz, kk;
	double m2, exx, eyy, ezz, exy, eyz, exz;
	double B0, B1, B2, B3, B4, B5;
	double *mm;
	int i;
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tn	= blockDim.x * gridDim.x;

	for (int j = tid; j < dim * dim; j += tn) {
		z = j / dim;
		y = j % dim;
		kz = map[z];
		ky = map[y];
		mm = &(lm2[j * ld1c]);
		kyz = ky*ky + kz*kz;

		for (int x = 0; x <= dim/2; ++x) {
			kx = map[x];
			m2 = mm[x];
			kk = 1.0/(kx*kx + kyz);
			exx = kx*kx*kk;
			eyy = ky*ky*kk;
			ezz = kz*kz*kk;
			exy = kx*ky*kk;
			eyz = ky*kz*kk;
			exz = kx*kz*kk;
			B0 = m2 - m2 * exx;
			B1 = -m2 * exy;
			B2 = -m2 * exz;
			B3 = m2 - m2 * eyy;
			B4 = -m2 * eyz;
			B5 = m2 - m2 * ezz;

			i = z * ld2c + y * ld1c + x;
			dev_influence_kernel(&(grid[2*i]), ld3, B0, B1, B2, B3, B4, B5);
		}
	}

	if (tid == 0) {
        grid[0 * ld3 + 0] = 0.0;
        grid[0 * ld3 + 1] = 0.0;
        grid[1 * ld3 + 0] = 0.0;
        grid[1 * ld3 + 1] = 0.0;
        grid[2 * ld3 + 0] = 0.0;
        grid[2 * ld3 + 1] = 0.0;
	}
}

__global__ void kel_ap_influence_2(double *map, 
								   double *lm2, 
								   int dim, int ld1c,
				   				   int ld2c, double *grid, int ld3)
{
	int z, y;
	double kz, ky, kx, kyz, kk;
	double m2, exx, eyy, ezz, exy, eyz, exz;
	double B0, B1, B2, B3, B4, B5;
	const double *mm;
	int i;

	int bid = blockIdx.x;
	int bn  = gridDim.x;
	int tid = threadIdx.x;
	int tn	= blockDim.x;
	for (int j = bid; j < dim * dim; j += bn) {
		z = j / dim;
		y = j % dim;
		kz = map[z];
		ky = map[y];
		mm = &(lm2[j * ld1c]);
		kyz = ky*ky + kz*kz;

		for (int x = tid; x <= dim+1; x += tn) {
			kx = map[x/2];
			m2 = mm[x/2];
			kk = 1.0/(kx*kx + kyz);
			exx = kx*kx*kk;
			eyy = ky*ky*kk;
			ezz = kz*kz*kk;
			exy = kx*ky*kk;
			eyz = ky*kz*kk;
			exz = kx*kz*kk;
			B0 = m2 - m2*exx;
			B1 = -m2 * exy;
			B2 = -m2 * exz;
			B3 = m2 - m2 * eyy;
			B4 = -m2 * eyz;
			B5 = m2 - m2 * ezz;
	
			i = z*ld2c*2 + y*ld1c*2 + x;
			dev_influence_kernel_2(&(grid[i]), ld3, B0, B1, B2, B3, B4, B5);
		}
	}

	// if (bid == 0 && tid == 0) {
    grid[0 * ld3 + 0] = 0.0;
    grid[0 * ld3 + 1] = 0.0;
    grid[1 * ld3 + 0] = 0.0;
    grid[1 * ld3 + 1] = 0.0;
    grid[2 * ld3 + 0] = 0.0;
    grid[2 * ld3 + 1] = 0.0;
	// }
}

__global__ void kel_ap_influence_3(double *map, double *lm2, int dim, int ld1c,
								   int ld2c, double *grid, int ld3)
{
	int z, y;
	double kz, ky, kx, kyz, kk;
	double m2, exx, eyy, ezz, exy, eyz, exz;
	double B0, B1, B2, B3, B4, B5;
	const double *mm;
	int i;

	int bid = blockIdx.x;
	int bn  = gridDim.x;
	int tid = threadIdx.x;
	int tn	= blockDim.x;
	for (int j = bid; j < dim * dim; j += bn) {
		z = j / dim;
		y = j % dim;
		kz = map[z];
		ky = map[y];
		mm = &(lm2[j * ld1c]);
		kyz = ky*ky + kz*kz;

		for (int x = tid; x <= dim/2; x += tn) {
			kx = map[x];
			m2 = mm[x];
			kk = 1.0/(kx*kx + kyz);
			exx = kx*kx*kk;
			eyy = ky*ky*kk;
			ezz = kz*kz*kk;
			exy = kx*ky*kk;
			eyz = ky*kz*kk;
			exz = kx*kz*kk;
			B0 = m2 - m2*exx;
			B1 = -m2 * exy;
			B2 = -m2 * exz;
			B3 = m2 - m2 * eyy;
			B4 = -m2 * eyz;
			B5 = m2 - m2 * ezz;
	
			i = z*ld2c + y*ld1c + x;
			dev_influence_kernel(&(grid[2*i]), ld3, B0, B1, B2, B3, B4, B5);
		}
	}

	// if (bid == 0 && tid == 0) {
    grid[0 * ld3 + 0] = 0.0;
    grid[0 * ld3 + 1] = 0.0;
    grid[1 * ld3 + 0] = 0.0;
    grid[1 * ld3 + 1] = 0.0;
    grid[2 * ld3 + 0] = 0.0;
    grid[2 * ld3 + 1] = 0.0;
	// }
}

__global__ void kel_ap_influence_dp_1(double *map, double *lm2, int dim, int ld1c,
									  int ld2c, double *grid, int ld3)
{
	int x, z, y;
	double kz, ky, kx, /*kyz,*/ kk;
	double m2, exx, eyy, ezz, exy, eyz, exz;
	double B0, B1, B2, B3, B4, B5;
	// double *mm;
	int i;
	
	extern __shared__ double s_map[];
	for (i = threadIdx.x; i < dim; i += blockDim.x)
		s_map[i] = map[i];
	__syncthreads();

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tn	= blockDim.x * gridDim.x;
	for (int j = tid; j < dim * dim; j += tn) {
		z = j / dim;
		y = j % dim;
		// kz = map[z];
		// ky = map[y];
		ky = s_map[y];
		// mm = &(lm2[j * ld1c]);
		// kyz = ky*ky + kz*kz;

		x = dim/2;
		// kx = map[x];
		kx = s_map[x];
		kz = s_map[z];
		// m2 = mm[x];
		m2 = lm2[j*ld1c+x];
		// kk = 1.0/(kx*kx + kyz);
		kk = 1.0/(kx*kx + ky*ky + kz*kz);
		exx = kx*kx*kk;
		eyy = ky*ky*kk;
		ezz = kz*kz*kk;
		exy = kx*ky*kk;
		eyz = ky*kz*kk;
		exz = kx*kz*kk;
		B0 = m2 - m2*exx;
		B1 = -m2 * exy;
		B2 = -m2 * exz;
		B3 = m2 - m2 * eyy;
		B4 = -m2 * eyz;
		B5 = m2 - m2 * ezz;

		i = z*ld2c + y*ld1c + x;
		dev_influence_kernel(&(grid[2*i]), ld3, B0, B1, B2, B3, B4, B5);
	}
}

__global__ void kel_ap_influence_dp_2(double *map, double *lm2, int dim, int ld1c,
									  int ld2c, double *grid, int ld3)
{
	int x, z, y;
	double kz, ky, kx, kk;
	double m2, exx, eyy, ezz, exy, eyz, exz;
	double B0, B1, B2, B3, B4, B5;
	int i;
	
	extern __shared__ double s_map[];
	for (i = threadIdx.x; i < dim; i += blockDim.x)
		s_map[i] = map[i];
	__syncthreads();

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tn	= blockDim.x * gridDim.x;
	for (int j = tid; j < dim * dim * 2; j += tn) {
		z = (j / 2) / dim;
		y = (j / 2) % dim;
		ky = s_map[y];

		x = dim/2;
		kx = s_map[x];
		kz = s_map[z];
		m2 = lm2[j/2*ld1c+x];
		kk = 1.0/(kx*kx + ky*ky + kz*kz);
		exx = kx*kx*kk;
		eyy = ky*ky*kk;
		ezz = kz*kz*kk;
		exy = kx*ky*kk;
		eyz = ky*kz*kk;
		exz = kx*kz*kk;
		B0 = m2 - m2*exx;
		B1 = -m2 * exy;
		B2 = -m2 * exz;
		B3 = m2 - m2 * eyy;
		B4 = -m2 * eyz;
		B5 = m2 - m2 * ezz;

		i = z*ld2c + y*ld1c + x;
		dev_influence_kernel_2(&(grid[2*i+(threadIdx.x%2)]), ld3, B0, B1, B2, B3, B4, B5);
	}
}

__global__ void kel_ap_influence_4(double *map, double *lm2, int dim, int ld1c,
								   int ld2c, double *grid, int ld3)
{
	int z, y;
	double kz, ky, kx, kyz, kk;
	double m2, exx, eyy, ezz, exy, eyz, exz;
	double B0, B1, B2, B3, B4, B5;
	const double *mm;
	int i;

	int bid = blockIdx.x;
	int bn  = gridDim.x;
	int tid = threadIdx.x;
	int tn	= blockDim.x;
	if (bid == 0 && tid == 0)
		kel_ap_influence_dp_2<<<480, 32, sizeof(double)*dim>>>(map, lm2, dim, ld1c, ld2c, grid, ld3);

	for (int j = bid; j < dim * dim; j += bn) {
		z = j / dim;
		y = j % dim;
		kz = map[z];
		ky = map[y];
		mm = &(lm2[j * ld1c]);
		kyz = ky*ky + kz*kz;

		for (int x = tid; x < dim/2; x += tn) {
			kx = map[x];
			m2 = mm[x];
			kk = 1.0/(kx*kx + kyz);
			exx = kx*kx*kk;
			eyy = ky*ky*kk;
			ezz = kz*kz*kk;
			exy = kx*ky*kk;
			eyz = ky*kz*kk;
			exz = kx*kz*kk;
			B0 = m2 - m2*exx;
			B1 = -m2 * exy;
			B2 = -m2 * exz;
			B3 = m2 - m2 * eyy;
			B4 = -m2 * eyz;
			B5 = m2 - m2 * ezz;
	
			i = z*ld2c + y*ld1c + x;
			dev_influence_kernel(&(grid[2*i]), ld3, B0, B1, B2, B3, B4, B5);
		}
	}

	if (bid == 0 && tid == 0) {
		// kel_ap_influence_dp<<<210, 128>>>(map, lm2, dim, ld1c, ld2c, grid, ld3);
    	grid[0] = 0.0;
    	grid[1] = 0.0;
    	grid[ld3] = 0.0;
    	grid[ld3+1] = 0.0;
    	grid[ld3+ld3] = 0.0;
    	grid[ld3+ld3+1] = 0.0;
	}
}

__global__ void kel_fft_scalling(double *grid, int length, double factor)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tn	= blockDim.x * gridDim.x;
	for (int i = tid; i < length; i += tn)
		grid[i] *= factor;
}

__device__ void dev_interpolate_kernel(int porder3, double *P, int *ind,
									   double *grid, int ld3, double *vels,
									   double alpha, double beta)
{
	int idx;
	double pvalue;
	double tmp0, tmp1, tmp2;
	tmp0 = 0.0;
	tmp1 = 0.0;
	tmp2 = 0.0;
	for (int j = 0; j < porder3; ++j) {
		idx = ind[j];
		pvalue = P[j];
		tmp0 += grid[0 * ld3 + idx] * pvalue;
		tmp1 += grid[1 * ld3 + idx] * pvalue;
		tmp2 += grid[2 * ld3 + idx] * pvalue;
	}
	vels[0] = beta * vels[0] + alpha * tmp0;
	vels[1] = beta * vels[1] + alpha * tmp1;
	vels[2] = beta * vels[2] + alpha * tmp2;
}

__global__ void kel_interpolate(int np, int porder3, double *P, int ldP, 
								int *ind, int ldind, double *grid, int ld3,
								double *vels, double alpha, double beta)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tn	= blockDim.x * gridDim.x;
	
	for (int i = tid; i < np; i += tn) {
		dev_interpolate_kernel(porder3, P+i*ldP, ind+i*ldind, grid, ld3,
							   vels + 3*i, alpha, beta);
	}
}

__device__ void dev_interpolate_kernel_2(int porder3, double *P, int *ind,
									   	 double *grid, int ld3, double *vels,
									   	 double alpha, double beta)
{
	int idx;
	double pvalue;

	extern __shared__ double tmp[];
	int tid = threadIdx.x;
	int tn	= blockDim.x;
	tmp[tid + 0 * tn] = 0.0;
	tmp[tid + 1 * tn] = 0.0;
	tmp[tid + 2 * tn] = 0.0;
	__syncthreads();

/*
	double tmp0, tmp1, tmp2;
	tmp0 = 0.0;
	tmp1 = 0.0;
	tmp2 = 0.0;
	for (int j = 0; j < porder3; ++j) {
		idx = ind[j];
		pvalue = P[j];
		tmp0 += grid[0 * ld3 + idx] * pvalue;
		tmp1 += grid[1 * ld3 + idx] * pvalue;
		tmp2 += grid[2 * ld3 + idx] * pvalue;
	}
*/

	for (int j = tid; j < porder3; j += tn) {
		idx = ind[j];
		pvalue = P[j];
		tmp[tid + 0 * tn] += grid[0 * ld3 + idx] * pvalue;
		tmp[tid + 1 * tn] += grid[1 * ld3 + idx] * pvalue;
		tmp[tid + 2 * tn] += grid[2 * ld3 + idx] * pvalue;
	}
	__syncthreads();

	for (int j = 1; j < tn; j *= 2) {
		if (tid % (j * 2) == 0) {
			tmp[tid + 0 * tn] += tmp[tid + 0 * tn + j];
			tmp[tid + 1 * tn] += tmp[tid + 1 * tn + j];
			tmp[tid + 2 * tn] += tmp[tid + 2 * tn + j];
		}
		__syncthreads();
	}

/*
	vels[0] = beta * vels[0] + alpha * tmp0;
	vels[1] = beta * vels[1] + alpha * tmp1;
	vels[2] = beta * vels[2] + alpha * tmp2;
*/
	if (tid == 0) {
		vels[0] = beta * vels[0] + alpha * tmp[tid + 0 * tn];
		vels[1] = beta * vels[1] + alpha * tmp[tid + 1 * tn];
		vels[2] = beta * vels[2] + alpha * tmp[tid + 2 * tn];
	}
}
__global__ void kel_interpolate_2(int np, int porder3, double *P, int ldP,
								  int *ind, int ldind, double *grid, int ld3,
								  double *vels, double alpha, double beta)
{
	int bid = blockIdx.x;
	int bn	= gridDim.x;

	for (int i = bid; i < np; i += bn) {
		dev_interpolate_kernel_2(porder3, P+i*ldP, ind+i*ldind, grid, ld3,
							   	 vels + 3*i, alpha, beta);
	}
}
