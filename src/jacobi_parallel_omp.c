#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "matrix.h"
#include "jacobi.h"

jacobi_result *jacobi_parallel_omp(matrix *m, int thread_count, bool verbose)
{

	int k = 0, t, termina = 0;
	double norma = 0, norma_ant = 0, n1 = 0, n2 = 0;

	int size = m->size;
	int *row_ptr = m->row_ptr;
	int *col_idx = m->col_idx;
	double *values = m->values;
	double *inv_diag = m->inv_diag;
	double *b = m->b;

	// initialize temp arrays
	int line_size = size * sizeof(double);
	double *x = malloc(line_size);
	double *x0 = malloc(line_size);

	// initial position
	for (t = 0; t < size; t++)
	{
		x0[t] = 0;
	}

	//--------------------------------------------------------------
	// PARTIZIONAMENTO A BLOCCHI BILANCIATI PER NUMERO DI NON-ZERI
	//--------------------------------------------------------------
	// Invece di dividere le righe in blocchi di uguale dimensione
	// (che puo' sbilanciare il carico se le righe hanno un numero
	// molto diverso di elementi non nulli), calcoliamo una volta
	// sola dei confini [thread_start[t], thread_start[t+1]) tali
	// che ogni thread lavori su un numero di nnz simile.
	// Questo si fa UNA SOLA VOLTA, fuori dal loop principale, dato
	// che la struttura sparsa non cambia durante l'iterazione.
	int *thread_start = malloc((thread_count + 1) * sizeof(int));
	int nnz_tot = row_ptr[size];

	thread_start[0] = 0;
	for (t = 1; t < thread_count; t++)
	{
		int target = (int)(((long long)nnz_tot * t) / thread_count);
		// ricerca binaria della prima riga i tale che row_ptr[i] >= target
		int lo = thread_start[t - 1], hi = size;
		while (lo < hi)
		{
			int mid = (lo + hi) / 2;
			if (row_ptr[mid] < target)
				lo = mid + 1;
			else
				hi = mid;
		}
		thread_start[t] = lo;
	}
	thread_start[thread_count] = size;

	// accumulatori privati per thread, per evitare una reduction
	// OpenMP automatica (che qui non useremmo comunque, dato che il
	// ciclo non e' piu' un #pragma omp for ma un range esplicito per
	// thread). Ogni thread scrive in una propria cella: per evitare
	// false sharing, il tipo occupa piu' spazio di un semplice double
	// o viene allineato ad una cache line separata.
	typedef struct
	{
		double n1;
		double n2;
		char pad[64 - 2 * sizeof(double)]; // padding per evitare false sharing
	} partial_t;
	partial_t *partial = calloc(thread_count, sizeof(partial_t));

	if (verbose)
	{
		printf("Partizionamento per nnz (thread_count = %d):\n", thread_count);
		for (t = 0; t < thread_count; t++)
		{
			printf("  thread %d -> righe [%d, %d)  nnz = %d\n",
				   t, thread_start[t], thread_start[t + 1],
				   row_ptr[thread_start[t + 1]] - row_ptr[thread_start[t]]);
		}
	}

// main loop
#pragma omp parallel num_threads(thread_count) \
	shared(norma, norma_ant, k, termina, x, x0, thread_start, partial)
	{
		double soma, diff;
		int i, idx;

		int tid = omp_get_thread_num();
		int row_start = thread_start[tid];
		int row_end = thread_start[tid + 1];

		if (verbose)
		{
#pragma omp critical
			printf("THREAD COUNT = %i, EXPECTED = %i, CURRENT=%i, ROWS=[%d,%d)\n",
				   omp_get_num_threads(), thread_count, tid, row_start, row_end);
		}

		while (termina == 0 && k < 100)
		{

			double local_n1 = 0.0, local_n2 = 0.0;

			// sum up items for each row (SOLO fuori diagonale, niente branch j!=i)
			// range esplicito per thread, bilanciato per nnz: nessun
			// bisogno di #pragma omp for / schedule, ogni thread ha gia'
			// il proprio blocco contiguo di righe -> nessun false sharing
			// su x[] tra thread diversi (ogni thread scrive un blocco
			// contiguo e disgiunto di x[]).
			for (i = row_start; i < row_end; i++)
			{
				soma = 0.0;
				int start = row_ptr[i];
				int end = row_ptr[i + 1];
				for (idx = start; idx < end; idx++)
				{
					soma += values[idx] * x0[col_idx[idx]];
				}

				// moltiplicazione per l'inverso precalcolato, niente divisione qui
				x[i] = (b[i] - soma) * inv_diag[i];

				diff = x[i] - x0[i];
				local_n1 += diff * diff;
				local_n2 += x[i] * x[i];
			}

			partial[tid].n1 = local_n1;
			partial[tid].n2 = local_n2;

// barriera esplicita: aspetta che tutti i thread abbiano
// finito di scrivere il proprio blocco di x[] e i propri
// accumulatori parziali, prima che il thread master li legga
#pragma omp barrier

// solo un thread esegue la parte seriale: somma dei parziali,
// calcolo norma, swap puntatori. "master" invece di "single":
// nessun controllo interno su "chi arriva prima", e' sempre
// il thread 0 (leggermente piu' leggero di "single").
#pragma omp master
			{
				n1 = 0.0;
				n2 = 0.0;
				for (t = 0; t < thread_count; t++)
				{
					n1 += partial[t].n1;
					n2 += partial[t].n2;
				}

				// calculate current error as "norma"
				norma = sqrt(n1 / n2);

				if (verbose)
					printf("\nk = %i, norma = %.20f, norma_ant = %.6f, n1 = %.6f, n2 = %.6f \n", k, norma, norma_ant, n1, n2);

				if ((k > 1 && (norma <= precision)) || isnan(norma))
				{
					termina = 1;
				}
				else
				{
					norma_ant = norma;
					// pointer swap invece di memcpy: O(1) invece di O(n)
					double *tmp = x0;
					x0 = x;
					x = tmp;
					k++;
				}
			} // fine regione master

// barriera esplicita: nessun thread puo' iniziare la prossima
// iterazione (che legge x0 aggiornato e "termina") finche' il
// master non ha finito di scrivere quei valori condivisi
#pragma omp barrier

		} // end of while, the main loop from jacobi
	} // end of parallel block

	// prepare results
	jacobi_result *res = malloc(sizeof(jacobi_result));
	res->x = malloc(line_size);
	for (t = 0; t < size; t++)
	{
		res->x[t] = x[t];
	}
	res->e = norma;
	res->k = k;

	// free memory
	free(x);
	free(x0);
	free(thread_start);
	free(partial);

	return res;
}