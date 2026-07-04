#ifndef MATRIX_H_
#define MATRIX_H_

/*
 * Formato CSR (Compressed Sparse Row) "vero", Structure-of-Arrays.
 * La diagonale NON è inclusa in col_idx/values: viene tenuta a parte,
 * già invertita (inv_diag[i] = 1.0 / A[i][i]), così nel ciclo caldo
 * di Jacobi non serve più alcun branch "j != i" né alcuna divisione.
 */
typedef struct matrix_t
{
	int *row_ptr;	  // size+1 elementi: row_ptr[i]..row_ptr[i+1]-1 sono gli indici in col_idx/values della riga i (SOLO fuori diagonale)
	int *col_idx;	  // nnz_off_diag elementi: indice di colonna di ogni elemento fuori diagonale
	double *values;	  // nnz_off_diag elementi: valore di ogni elemento fuori diagonale
	double *inv_diag; // size elementi: 1.0 / A[i][i], precalcolato una sola volta
	double *b, *x;
	unsigned int size;
} matrix;

void matrix_print(matrix *m);
matrix *matrix_load(char *filename);
void matrix_destroy(matrix *m);

#endif