#ifndef __SPME_KERNEL_H__
#define __SPME_KERNEL_H__


void apply_influence (int dim, double *grid,
                      int ld1, int ld2, int ld3,
                      double *map, double *lm2,
                      double **lB0, double **lB1, double **lB2,
                      double **lB3, double **lB4, double **lB5);

void interpolate (int np, int porder3,
                  double *P, int ldP, int *ind, int ldind,
                  double *grid, int ld3, double *vels,
                  double alpha, double beta);

void spread (int np, int nb, int *head, int *next,
             int porder3, double *P, int ldP, int *ind, int ldind,
             double *forces, double *grid, int ld3);


#endif /* __SPME_KERNEL_H__ */
