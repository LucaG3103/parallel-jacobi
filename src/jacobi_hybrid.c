#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include "matrix.h"
#include "stdlib.h"
#include "timer.h"
#include "results.h"
#include "jacobi.h"

jacobi_result *jacobi_hybrid(matrix *m, int thread_count, bool verbose,
                             int process_num, int process_count)
{
    // --- stessa logica di partizionamento righe -> processi di jacobi_mpi ---
    int *counts = malloc(process_count * sizeof(int));
    int *disps = malloc(process_count * sizeof(int));
    int resto = m->size % process_count;
    int qtd = ceil((double)m->size / process_count);
    int i;
    for (i = 0; i < process_count; i++)
    {
        counts[i] = qtd + 2;
        if (i == 0)
            disps[i] = 0;
        else
            disps[i] = disps[i - 1] + counts[i - 1];
        if (i == resto - 1)
            qtd--;
    }

    qtd = ceil((double)m->size / process_count);
    int initial_line;
    if (resto > 0 && process_num > resto - 1)
    {
        initial_line = qtd * resto + (qtd - 1) * (process_num - resto);
        qtd--;
    }
    else
    {
        initial_line = qtd * process_num;
    }
    int end_line = initial_line + qtd - 1;
    int local_rows = qtd; // righe assegnate a QUESTO processo MPI

    if (verbose)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        printf("A Process %i: %i -> %i (%i) con %i thread OMP\n",
               process_num, initial_line, end_line, counts[process_num] - 2, thread_count);
    }

    int j, z, k = 0, terminar = 0;
    int n1 = disps[process_num] + qtd;
    int n2 = disps[process_num] + qtd + 1;
    double norma = 0, norma_ant = 0, n1_sum, n2_sum;

    double *shared = malloc((m->size + 2 * process_count) * sizeof(double));
    double *x0 = malloc(m->size * sizeof(double));

    for (i = 0; i < m->size; i++)
        x0[i] = 1.0;

    int *row_ptr = m->row_ptr;
    int *col_idx = m->col_idx;
    double *values = m->values;
    double *inv_diag = m->inv_diag;
    double *b = m->b;

    // -----------------------------------------------------------------
    // Partizionamento per nnz del blocco di righe LOCALE tra i thread OMP
    // (stessa idea usata in jacobi_parallel_omp, ma ristretta a
    // [initial_line, end_line]). Calcolato una sola volta, fuori dal
    // loop, dato che la struttura sparsa non cambia durante le iterazioni.
    // -----------------------------------------------------------------
    int *thread_start = malloc((thread_count + 1) * sizeof(int));
    int nnz_local_base = row_ptr[initial_line];
    int nnz_local_tot = (local_rows > 0) ? (row_ptr[end_line + 1] - nnz_local_base) : 0;

    thread_start[0] = initial_line;
    for (int t = 1; t < thread_count; t++)
    {
        int target = nnz_local_base + (int)(((long long)nnz_local_tot * t) / thread_count);
        int lo = thread_start[t - 1], hi = end_line + 1;
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
    thread_start[thread_count] = end_line + 1;

    // accumulatori privati per thread, con padding anti false-sharing
    typedef struct
    {
        double n1;
        double n2;
        char pad[64 - 2 * sizeof(double)];
    } partial_t;
    partial_t *partial = calloc(thread_count, sizeof(partial_t));

    if (verbose && local_rows > 0)
    {
        printf("Process %i, partizionamento nnz tra %i thread:\n", process_num, thread_count);
        for (int t = 0; t < thread_count; t++)
        {
            printf("  proc %i thread %d -> righe [%d, %d)  nnz = %d\n",
                   process_num, t, thread_start[t], thread_start[t + 1],
                   row_ptr[thread_start[t + 1]] - row_ptr[thread_start[t]]);
        }
    }

    // main loop
    while (k < 100 && !terminar)
    {
        shared[n1] = 0;
        shared[n2] = 0;

#pragma omp parallel num_threads(thread_count) \
    shared(shared, x0, thread_start, partial, row_ptr, col_idx, values, inv_diag, b, disps, initial_line, process_num, verbose)
        {
            int tid = omp_get_thread_num();
            int row_start = thread_start[tid];
            int row_end = thread_start[tid + 1];
            double local_n1 = 0.0, local_n2 = 0.0;
            double soma, x2;

            for (int ii = row_start; ii < row_end; ii++)
            {
                soma = 0.0;
                int start = row_ptr[ii];
                int end = row_ptr[ii + 1];
                for (int idx = start; idx < end; idx++)
                    soma += values[idx] * x0[col_idx[idx]];

                if (verbose)
                {
#pragma omp critical
                    printf("Process %i (thread %i): linha = %i, soma = %f\n",
                           process_num, tid, ii, soma);
                }

                int zz = disps[process_num] + (ii - initial_line);
                shared[zz] = (b[ii] - soma) * inv_diag[ii];
                x2 = shared[zz] - x0[ii];

                local_n1 += x2 * x2;
                local_n2 += shared[zz] * shared[zz];
            }

            partial[tid].n1 = local_n1;
            partial[tid].n2 = local_n2;
        } // fine regione parallela OMP (barriera implicita in uscita)

        for (int t = 0; t < thread_count; t++)
        {
            shared[n1] += partial[t].n1;
            shared[n2] += partial[t].n2;
        }

        // riduzione/scambio tra i processi MPI (identico a jacobi_mpi)
        MPI_Allgatherv(
            MPI_IN_PLACE,
            0,
            MPI_DATATYPE_NULL,
            &(shared[0]),
            counts,
            disps,
            MPI_DOUBLE,
            MPI_COMM_WORLD);

        if (process_num == 0)
        {
            n1_sum = 0;
            n2_sum = 0;
            for (i = 0; i < process_count; i++)
            {
                n1_sum += shared[disps[i] + counts[i] - 2];
                n2_sum += shared[disps[i] + counts[i] - 1];
            }
            norma = sqrt(n1_sum / n2_sum);

            if (verbose)
            {
                printf("\nk = %i, norma = %.20f, norma_ant = %.6f, n1 = %.6f, n2 = %.6f \n",
                       k, norma, norma_ant, n1_sum, n2_sum);
            }

            if ((k > 1 && (norma <= precision)) || isnan(norma))
            {
                terminar = 1;
            }
            else
            {
                norma_ant = norma;
            }
        }

        MPI_Bcast(&terminar, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!terminar)
        {
            k++;
            for (i = 0, z = 0; i < process_count; i++)
            {
                for (j = 0; j < counts[i] - 2; ++j)
                {
                    x0[z++] = shared[disps[i] + j];
                }
            }
        }

        if (verbose && process_num == 0)
        {
            printf("\nx0 = ");
            for (i = 0; i < m->size; i++)
                printf("%f, ", x0[i]);
            printf("\n");
        }
    }

    jacobi_result *res = NULL;
    if (process_num == 0)
    {
        res = malloc(sizeof(jacobi_result));
        res->x = malloc(m->size * sizeof(double));
        for (i = 0, z = 0; i < process_count; i++)
        {
            for (j = 0; j < counts[i] - 2; ++j)
            {
                res->x[z++] = shared[disps[i] + j];
            }
        }
        res->e = norma;
        res->k = k;
    }

    free(counts);
    free(disps);
    free(shared);
    free(x0);
    free(thread_start);
    free(partial);

    return res;
}