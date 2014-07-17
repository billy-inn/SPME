#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>
#include <getopt.h>
#include <omp.h>
#include <assert.h>

#include "spme.h"

extern void vs_spme(spme_t*, gpu_spme_t*);

const struct option long_options[] = {
    {"help",     0, NULL, 'h'},
    {"version",  0, NULL, 'v'},
    {"xyz",      1, NULL, 12},
    {"dim",      1, NULL, 21},
    {"porder",   1, NULL, 22},
};

const char *const short_options = ":hv";
const char *version_info = "0.1.0";


static void usage (char *call)
{
    fprintf (stderr, "Usage: %s [OPTIONS]\n", call);
    fprintf (stderr, "Options:\n");
    fprintf (stderr, "\t-h or --help         Display this information\n");
    fprintf (stderr, "\t-v or --version      Display version information\n");
    fprintf (stderr, "\t--dim                nfft grids\n");
    fprintf (stderr, "\t--porder             interpolation order\n");
    fprintf (stderr, "\t--xyz                xyz file\n");
}


static void print_version (char *call)
{
    fprintf (stdout, "%s version %s\n", call, version_info);
}


static int read_xyz (char *xyz_file, double **pos, int *np, double *L)
{
    char line[1024];
    char strbuf[128];
    FILE *fp;
    double *_pos;
    double _L;
    int _np;
    double x;
    double y;
    double z;
    double a;
    int np_count;
    int nsc;
    int i;
        
    if (NULL == (fp = fopen (xyz_file, "r")))
    {
        fprintf (stderr, "Can not open file %s\n", xyz_file);
        return -1;
    }
    np_count = 0;
    _np = 0;
    while (fgets (line, 1024, fp) != NULL)
    {
        nsc = sscanf (line, "%s\n", strbuf);
        if (strbuf[0] != '#' && nsc != EOF)
        {
            sscanf (line, "%d %lf\n", &_np, &_L);
            break;
        }
    }
    if (_np <= 0)
    {
        fprintf (stderr, "Invalid number of particles\n");
        return -1;
    }   
    if (_L <= 0.0)
    {
        fprintf (stderr, "Invalid box size\n");
        return -1;   
    }
    _pos = (double *)_mm_malloc (_np * 3 * sizeof(double) + 8, 64);
    assert (_pos != NULL);
    while (fgets (line, 1024, fp) != NULL)
    {
        nsc = sscanf (line, "%s %lf %lf %lf %lf\n", strbuf, &x, &y, &z, &a);
        if (strbuf[0] == '#' || nsc == EOF)
            continue;
        _pos[3 * np_count] = x;
        _pos[3 * np_count + 1] = y;
        _pos[3 * np_count + 2] = z;
        np_count++;
    }
    if (np_count != _np)
    {
        fprintf (stderr, "%d particles are found in %s, but num-particles = %d\n",
                    np_count, xyz_file, _np);
    }    
    fclose (fp);

    #pragma omp parallel for default(none)\
                             private(i)\
                             shared (_pos, _L, _np)
    for (i = 0; i < _np; i++)
    {
        _pos[3 * i + 0] = fmod(_pos[3 * i + 0], _L);
        _pos[3 * i + 1] = fmod(_pos[3 * i + 1], _L);
        _pos[3 * i + 2] = fmod(_pos[3 * i + 2], _L);
        _pos[3 * i + 0] = (_pos[3 * i + 0] >= 0.0 ? _pos[3 * i + 0] : 
            _L + _pos[3 * i + 0]);
        _pos[3 * i + 1] = (_pos[3 * i + 1] >= 0.0 ? _pos[3 * i + 1] : 
            _L + _pos[3 * i + 1]);
        _pos[3 * i + 2] = (_pos[3 * i + 2] >= 0.0 ? _pos[3 * i + 2] : 
            _L + _pos[3 * i + 2]);
    }
    *pos = _pos;
    *np = _np;
    *L = _L;

    return 0;
}


int main (int argc, char **argv)
{
    int c = 0;
    char xyz_file[1024];
    int dim;
    int porder;
    spme_t *spme;
    double xi;
    double L;  /* simulation box size */
    int np;       /* number of particles */
    double *pos;  /* particle postitions */
    int nthreads;
    double alpha;
    double beta;
    double *f;
    double *v;
    int ldm;
    int i; 

	// gpu related var
	gpu_spme_t *gpu_spme; 	
	
    // default settings
    strcpy (xyz_file, "input.xyz");

    dim = 64;
    porder = 4;
    nthreads = omp_get_max_threads ();
    omp_set_num_threads (nthreads);
    
    /* parse arguments */
    while ((c = getopt_long (argc, argv, short_options,
                             long_options, NULL)) != -1)
    {
        switch (c)
        {
            case 'h':
                usage (argv[0]);
                return 0;
            case 'v':
                print_version (argv[0]);
                return 0;
            case 12:
                strcpy (xyz_file, optarg);
                break;
            case 21:
                dim = atoi (optarg);
                assert (dim > 0);
                break;
            case 22:
                porder = atoi (optarg);
                assert (porder > 0);
                break;
            case ':':
                fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                return -1;
            case '?':
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                return -1;
            default:
                usage (argv[0]);
                return -1;
        }
    }

    printf ("nthreads = %d\n", nthreads);
    
    // import particle positions
    if (read_xyz (xyz_file, &pos, &np, &L) != 0)
        return -1;
    printf ("imported %d particles, L = %lf\n", np, L);
    
    // memmory is aligned for better performance
    ldm = PAD_LEN(np * 3, sizeof(double));
    
    if (4 * porder >= dim)
    {
        fprintf (stderr, "dim must be larger than 4 * porder\n");
        return -1;
    }
    xi = 0.5;
    printf ("SPME: xi = %lf, dim = %d, porder = %d\n",
                 xi, dim, porder);

    // init spme engine
    create_spme_engine (xi, dim, porder,
                        np, L, nthreads, &spme);
	
	// gpu init spme engine
	gpu_create_spme_engine(xi, dim, porder, np, L, &gpu_spme, spme);

	// vs
	// vs_spme(spme, gpu_spme);

    alpha = 2.1;
    beta = 2.2;
    // input forces
    f = (double *)_mm_malloc (sizeof(double) * ldm * 5, 64);
    // output velocities 
    v = (double *)_mm_malloc (sizeof(double) * ldm * 5, 64);

    srand (1234);
    for (i = 0; i < ldm * 5; i++)
    {
        f[i] = rand () * 3.0;
        v[i] = rand () * 4.0;
    }

	gpu_allocate(ldm, 5, np, f, v, pos);

    for (i = 0; i < 1/*5*/; i++)
    {
        // main computation, need to be optimized on GPUs
        // v = alpha * M * f + beta * v
        compute_spme (spme, pos, 1, alpha,
                      &(f[i * ldm]), ldm, beta, &(v[i * ldm]), ldm);

	gpu_compute_spme (gpu_spme, pos, 1, alpha,
			  i, ldm, beta, i, ldm);
    }
    destroy_spme_engine (spme);

	gpu_destroy_spme_engine(gpu_spme);
	gpu_deallocate(f, v, pos);

    _mm_free (pos);
    _mm_free (f);
    _mm_free (v);
    
    return 0;
}
