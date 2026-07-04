#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix.h"
#include "jacobi.h"

jacobi_result *jacobi_serial(matrix *m, bool verbose)
{
	int i, idx, k = 0;
	double norma, norma_ant = 0, soma, n1, n2, diff;

	int size = m->size;
	int *row_ptr = m->row_ptr;
	int *col_idx = m->col_idx;
	double *values = m->values;
	double *inv_diag = m->inv_diag;
	double *b = m->b;

	// initialize temp arrays
	double *x = malloc(size * sizeof(double));
	double *x0 = malloc(size * sizeof(double));

	// initial position
	for (i = 0; i < size; i++)
	{
		x0[i] = 0;
	}

	// main loop
	while (k < 100)
	{

		n1 = 0;
		n2 = 0;
		for (i = 0; i < size; i++)
		{
			// sum up line items (SOLO fuori diagonale, niente branch j!=i)
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
			n1 += diff * diff;
			n2 += x[i] * x[i];
		}

		// calculate current error as "norma"
		norma = sqrt(n1 / n2);
		if (verbose)
			printf("\nk = %i, norma = %.20f, norma_ant = %.6f, n1 = %.6f, n2 = %.6f \n", k, norma, norma_ant, n1, n2);

		if ((k > 1 && (norma <= precision)) || isnan(norma))
		{
			break;
		}
		else
		{
			norma_ant = norma;
			double *tmp = x0;
			x0 = x;
			x = tmp;
			k++;
		}
	}

	// prepare results
	jacobi_result *res = malloc(sizeof(jacobi_result));
	res->x = malloc(size * sizeof(double));
	for (i = 0; i < size; i++)
	{
		res->x[i] = x[i];
	}
	res->e = norma;
	res->k = k;

	// free memory
	free(x);
	free(x0);

	return res;
}