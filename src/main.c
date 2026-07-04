#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "matrix.h"
#include "stdlib.h"
#include "timer.h"
#include "results.h"
#include "jacobi.h"

/*
 * Programma unico, compilato con mpicc, che puo' eseguire una delle 4
 * varianti dell'algoritmo di Jacobi scelta da linea di comando:
 *
 *   S = Serial            (usa solo il rank 0, lanciare con -np 1)
 *   O = OpenMP puro       (usa solo il rank 0, lanciare con -np 1)
 *   M = MPI puro          (usa tutti i processi, 1 thread ciascuno)
 *   H = Ibrido MPI+OpenMP (usa tutti i processi, ognuno con thread_count thread)
 *
 * Uso:
 *   mpirun -np <num_processi> ./jacobi <input_file> <algoritmo:S|O|M|H> \
 *          <thread_count> <verbose:0|1>
 *
 * Esempi:
 *   mpirun -np 1 ./jacobi matrice.txt S 1 0
 *   mpirun -np 1 ./jacobi matrice.txt O 8 0
 *   mpirun -np 4 ./jacobi matrice.txt M 1 0
 *   mpirun -np 4 ./jacobi matrice.txt H 4 0   (4 processi x 4 thread = 16 worker)
 */

/* Soglia di convergenza usata da tutte le varianti di Jacobi (jacobi.h la
 * dichiara come 'extern double precision;'). Se nel tuo progetto originale
 * questo valore era diverso, cambialo qui. */
double precision = 1e-5;

int main(int argc, char *argv[])
{
	int my_rank, num_procs;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	if (argc != 5)
	{
		if (my_rank == 0)
		{
			puts("Argomenti non corretti!");
			puts("Uso: mpirun -np <num_processi> ./jacobi <input_file> <S|O|M|H> "
				 "<thread_count> <verbose:0|1>");
		}
		MPI_Finalize();
		exit(1);
	}

	char *input_file = argv[1];
	char algorithm = argv[2][0];
	int thread_count = strtol(argv[3], NULL, 10);
	int verbose = strtol(argv[4], NULL, 10);

	bool uses_mpi = (algorithm == 'M' || algorithm == 'H');

	if (!uses_mpi && num_procs > 1 && my_rank == 0)
	{
		printf("Attenzione: l'algoritmo '%c' non usa MPI, lancialo con -np 1. "
			   "Solo il rank 0 eseguira' il calcolo, gli altri %i processi "
			   "verranno terminati subito.\n",
			   algorithm, num_procs - 1);
	}

	// i processi diversi da 0 non servono per S e O: escono subito
	if (!uses_mpi && my_rank != 0)
	{
		MPI_Finalize();
		return 0;
	}

	if (verbose && my_rank == 0)
		puts("---BEGIN---");

	if (verbose && my_rank == 0)
		printf("Input file: '%s', algoritmo: %c, thread: %i, processi MPI: %i\n\n",
			   input_file, algorithm, thread_count, uses_mpi ? num_procs : 1);

	// carica la matrice
	// per M/H tutti i processi hanno bisogno della matrice completa
	// (come nella versione jacobi_mpi originale)
	matrix *m = matrix_load(input_file);

	if (verbose && my_rank == 0)
		matrix_print(m);

	timer *t = NULL;
	if (my_rank == 0)
		t = start_timer();

	jacobi_result *result = NULL;

	switch (algorithm)
	{
	case 'S': // serial
		if (my_rank == 0)
			result = jacobi_serial(m, verbose);
		break;

	case 'O': // OpenMP puro
		if (my_rank == 0)
			result = jacobi_parallel_omp(m, thread_count, verbose);
		break;

	case 'M': // MPI puro
		result = jacobi_mpi(m, verbose, my_rank, num_procs);
		break;

	case 'H': // ibrido MPI + OpenMP
		result = jacobi_hybrid(m, thread_count, verbose, my_rank, num_procs);
		break;

	default:
		if (my_rank == 0)
			printf("Algoritmo sconosciuto: '%c' (usa S, O, M o H)\n", algorithm);
	}

	if (my_rank == 0)
		stop_timer(t, verbose);

	if (verbose && my_rank == 0 && result != NULL)
	{
		int i;
		printf("\nResults: ");
		for (i = 0; i < m->size && i < 200; i++)
			printf("%f, ", result->x[i]);
		printf("\nIterations: %i ", result->k);
	}

	if (my_rank == 0 && result != NULL)
	{
		// "unita' di calcolo" salvata nel file dei risultati:
		// - S: 1 (nessun parallelismo)
		// - O: thread_count
		// - M: num_procs
		// - H: num_procs * thread_count
		int work_units;
		switch (algorithm)
		{
		case 'O':
			work_units = thread_count;
			break;
		case 'M':
			work_units = num_procs;
			break;
		case 'H':
			work_units = num_procs * thread_count;
			break;
		case 'S':
		default:
			work_units = 1;
		}
		write_results(t, input_file, work_units, algorithm, m->size);
	}

	matrix_destroy(m);

	if (my_rank == 0)
	{
		if (t != NULL)
			free(t);
		if (result != NULL)
			free(result);
	}

	if (verbose && my_rank == 0)
		puts("\n\n---END---");

	MPI_Finalize();
	return 0;
}