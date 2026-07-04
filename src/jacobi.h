#ifndef JACOBI_H
#define JACOBI_H

#include <stdbool.h>
#include "matrix.h"
#include "results.h"

extern double precision;

/* Se il tuo progetto ha gia' questo struct definito altrove (es. in
 * results.h nel tuo repo reale), RIMUOVI questa definizione da qui per
 * evitare un errore di ridefinizione, e assicurati solo che l'header
 * che lo definisce sia incluso sopra. I campi (x, e, k) sono dedotti
 * dall'uso in jacobi_mpi.c / jacobi_parallel_omp.c / jacobi_serial.c. */
typedef struct
{
	double *x; // soluzione
	double e;  // errore finale (norma)
	int k;	   // numero di iterazioni
} jacobi_result;

jacobi_result *jacobi_serial(matrix *m, bool verbose);

jacobi_result *jacobi_parallel_omp(matrix *m, int thread_count, bool verbose);

jacobi_result *jacobi_mpi(matrix *m, bool verbose, int process_num, int process_count);

/* Versione ibrida: partiziona le righe tra i processi MPI (come jacobi_mpi)
 * e, all'interno di ciascun processo, partiziona ulteriormente le proprie
 * righe tra i thread OpenMP in base al numero di non-zeri (come jacobi_parallel_omp). */
jacobi_result *jacobi_hybrid(matrix *m, int thread_count, bool verbose,
							 int process_num, int process_count);

#endif