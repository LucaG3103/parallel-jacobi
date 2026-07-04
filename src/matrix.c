#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include "matrix.h"

/**
 * http://math.nist.gov/MatrixMarket/formats.html
 *
 * It's necessary to remove headers
 *
 * Costruzione CSR a due passate:
 *  1) passata di lettura: si contano gli elementi fuori diagonale per riga
 *     e si memorizza la diagonale a parte (niente più assunzione che le
 *     triplette siano ordinate per riga, come nel loader originale).
 *  2) prefix-sum per costruire row_ptr, poi si riempiono col_idx/values
 *     usando un cursore per riga.
 */
matrix* matrix_load_mm(char* filename) {

	long i, size, size_cols, count, row, col;
	double val;
	bool is_symmetric = false;
	char line[1024];

	//open file
	FILE *file;
	file = fopen(filename, "r");
	if (file == NULL) {
		puts("\nCannot open file");
		puts(filename);
		exit(1);
	}

	//--------------------------------------------------------------
	// Skip dell'header MatrixMarket e delle righe di commento.
	// Un file .mtx scaricato "cosi' com'e'" (es. da SuiteSparse) e'
	// fatto cosi':
	//   %%MatrixMarket matrix coordinate real symmetric
	//   %-------------------------------------------------
	//   % eventuali commenti
	//   %-------------------------------------------------
	//   size size_cols count
	//   riga colonna valore
	//   ...
	// fscanf("%li", ...) non sa saltare righe di testo che iniziano
	// con '%': si blocca sul primo carattere non numerico e fallisce
	// silenziosamente, lasciando size/size_cols/count non inizializzati.
	// Serve quindi leggere riga per riga finche' non se ne trova una
	// che non inizia con '%': quella e' la riga con le dimensioni.
	//--------------------------------------------------------------
	long file_pos;
	do {
		file_pos = ftell(file);
		if (fgets(line, sizeof(line), file) == NULL) {
			puts("\nUnexpected end of file while skipping header");
			exit(1);
		}
		if (line[0] == '%') {
			//il banner (prima riga, "%%MatrixMarket ...") contiene
			//anche l'informazione sulla simmetria della matrice
			if (strncmp(line, "%%MatrixMarket", 14) == 0) {
				if (strstr(line, "symmetric") != NULL ||
				    strstr(line, "hermitian") != NULL) {
					is_symmetric = true;
				}
			}
		}
	} while (line[0] == '%');

	//torna all'inizio della riga con le dimensioni e la parsa
	fseek(file, file_pos, SEEK_SET);
	fscanf(file, "%li", &size);
	fscanf(file, "%li", &size_cols);
	fscanf(file, "%li", &count);

	if (size != size_cols) {
		puts("\nMatrix is not square");
		exit(1);
	}

	//creates matrix
	matrix *m = malloc(sizeof(matrix));
	m->size = size;

	m->b = malloc(size * sizeof(double));
	m->x = malloc(size * sizeof(double));
	m->inv_diag = malloc(size * sizeof(double));

	//--------------------------------------------------------------
	// Se la matrice e' "symmetric" nel banner MatrixMarket, il file
	// contiene UNA SOLA tripletta per ogni coppia (i,j)/(j,i) fuori
	// diagonale (di solito solo il triangolo inferiore, i >= j).
	// Serve quindi specchiare ogni entry fuori diagonale: per ogni
	// (row,col,val) letta con row != col, la matrice ha sia
	// A[row][col] = val sia A[col][row] = val.
	// Alloco quindi il doppio dello spazio per le triplette temporanee
	// (caso peggiore: tutte fuori diagonale) cosi' non serve una terza
	// passata sul file per ricontare.
	//--------------------------------------------------------------
	long tmp_capacity = is_symmetric ? (2 * count) : count;

	//temp storage per le triplette lette dal file (+ eventuali specchiate)
	long *tmp_row = malloc(tmp_capacity * sizeof(long));
	long *tmp_col = malloc(tmp_capacity * sizeof(long));
	double *tmp_val = malloc(tmp_capacity * sizeof(double));

	//0 come sentinella: "diagonale non ancora trovata"
	for (i = 0; i < size; i++) {
		m->inv_diag[i] = 0.0;
	}

	int *row_count = calloc(size, sizeof(int));
	long off_diag_count = 0;
	long tmp_count = 0; //numero effettivo di triplette in tmp_* (>= count se symmetric)

	//1a passata: leggo tutto, separo diagonale da fuori-diagonale, conto per riga
	for (i = 0; i < count; i++) {
		fscanf(file, "%li", &row);
		fscanf(file, "%li", &col);
		if (!fscanf(file, "%lf", &val)) {
			break;
		}
		row--; col--;

		tmp_row[tmp_count] = row;
		tmp_col[tmp_count] = col;
		tmp_val[tmp_count] = val;
		tmp_count++;

		if (row == col) {
			m->inv_diag[row] = val; //per ora valore grezzo, invertito dopo
		} else {
			row_count[row]++;
			off_diag_count++;

			if (is_symmetric) {
				//specchio anche l'entry (col,row)
				tmp_row[tmp_count] = col;
				tmp_col[tmp_count] = row;
				tmp_val[tmp_count] = val;
				tmp_count++;

				row_count[col]++;
				off_diag_count++;
			}
		}
	}

	//prefix-sum -> row_ptr
	m->row_ptr = malloc((size + 1) * sizeof(int));
	m->row_ptr[0] = 0;
	for (i = 0; i < size; i++) {
		m->row_ptr[i + 1] = m->row_ptr[i] + row_count[i];
	}

	m->col_idx = malloc(off_diag_count * sizeof(int));
	m->values = malloc(off_diag_count * sizeof(double));

	//cursore di scrittura per ogni riga, parte da row_ptr[i]
	int *cursor = malloc(size * sizeof(int));
	for (i = 0; i < size; i++) {
		cursor[i] = m->row_ptr[i];
	}

	//2a passata: riempio col_idx/values (solo fuori diagonale)
	//NB: uso tmp_count, non count, perche' se la matrice e' symmetric
	//tmp_count include anche le entry specchiate aggiunte sopra
	for (i = 0; i < tmp_count; i++) {
		if (tmp_row[i] != tmp_col[i]) {
			int pos = cursor[tmp_row[i]]++;
			m->col_idx[pos] = (int) tmp_col[i];
			m->values[pos] = tmp_val[i];
		}
	}

	//inverto la diagonale una sola volta (mai più divisioni nel ciclo caldo)
	for (i = 0; i < size; i++) {
		if (m->inv_diag[i] == 0.0) {
			puts("\nWarning: diagonale mancante o nulla su una riga, la matrice potrebbe essere singolare");
		} else {
			m->inv_diag[i] = 1.0 / m->inv_diag[i];
		}
	}

	free(tmp_row);
	free(tmp_col);
	free(tmp_val);
	free(row_count);
	free(cursor);
	fclose(file);

	//read vector B
	//Nota: molte matrici scaricate da SuiteSparse NON includono un
	//file "_b.mtx" separato (a differenza del formato di test locale
	//usato finora). Se manca, si genera un b sintetico: b = A * x_ones,
	//cioe' il prodotto della matrice per il vettore di soli 1. Questo
	//garantisce che il sistema Ax=b abbia una soluzione esatta nota
	//(x = vettore di 1), utile anche per verificare la correttezza
	//del risultato di Jacobi a fine esecuzione.
	char filename_b[100];
	strncpy((char*)  &filename_b, filename, strlen(filename) - 4);
	filename_b[strlen(filename) - 4] = '\0';
	strcat((char*) &filename_b, "_b.mtx");

	if (access(filename_b, R_OK) != -1) {
		file = fopen(filename_b, "r");

		//ignores
		fscanf(file, "%li", &count);
		fscanf(file, "%li", &size_cols);

		for (i = 0; i < count; i++) {
			if (!fscanf(file, "%lf", &m->b[i])) {
				break;
			}
		}

		fclose(file);
	} else {
		//b = A * 1: per ogni riga, diagonale + somma degli elementi fuori diagonale
		for (i = 0; i < size; i++) {
			double diag_val = (m->inv_diag[i] != 0.0) ? (1.0 / m->inv_diag[i]) : 0.0;
			double sum = diag_val;
			int start = m->row_ptr[i];
			int end = m->row_ptr[i + 1];
			for (int idx = start; idx < end; idx++) {
				sum += m->values[idx];
			}
			m->b[i] = sum;
		}
	}

	//read expected results from vector X
	char filename_x[100];
	strncpy((char*) &filename_x, filename, strlen(filename) - 4);
	filename_x[strlen(filename) - 4] = '\0';
	strcat((char*) &filename_x, "_x.mtx");

	if (access(filename_x, R_OK) != -1) {

		file = fopen(filename_x, "r");
		if (file == NULL) {
			puts("\nCannot open file");
			puts(filename_x);
			exit(1);
		}

		//ignores
		fscanf(file, "%li", &count);
		fscanf(file, "%li", &size_cols);

		for (i = 0; i < count; i++) {
			if (!fscanf(file, "%lf", &m->x[i])) {
				break;
			}
		}

		fclose(file);
	} else {
		for (i = 0; i < size; i++) {
			m->x[i] = -1.0;
		}
	}

	return m;
}

/*
 * Formato "dense-as-sparse": ogni riga ha esattamente size elementi
 * (compresa la diagonale), quindi il layout CSR è banale da calcolare
 * a priori: size-1 elementi fuori diagonale per riga, row_ptr[i] = i*(size-1).
 */
matrix* matrix_load_original(char* filename) {
	int i, j, size;
	matrix *m = malloc(sizeof(matrix));

	//open file
	FILE *file;
	file = fopen(filename, "r");
	if (file == NULL) {
		puts("\nCannot open file");
		puts(filename);
		exit(1);
	}

	//read matrix size (era "%iu", specificatore non valido: corretto in "%d")
	fscanf(file, "%d", &size);

	m->size = size;
	m->b = malloc(size * sizeof(double));
	m->x = malloc(size * sizeof(double));
	m->inv_diag = malloc(size * sizeof(double));

	int off_diag_per_row = size - 1;
	long total_off_diag = (long) size * off_diag_per_row;

	m->row_ptr = malloc((size + 1) * sizeof(int));
	for (i = 0; i <= size; i++) {
		m->row_ptr[i] = i * off_diag_per_row;
	}
	m->col_idx = malloc(total_off_diag * sizeof(int));
	m->values = malloc(total_off_diag * sizeof(double));

	char c[10];
	double val;

	for (i = 0; i < size; i++) {
		int cursor = m->row_ptr[i];
		for (j = 0; j < size; j++) {
			if (!fscanf(file, "%lf", &val)) {
				break;
			}
			if (j == i) {
				m->inv_diag[i] = (val != 0.0) ? (1.0 / val) : 0.0;
			} else {
				m->col_idx[cursor] = j;
				m->values[cursor] = val;
				cursor++;
			}
		}

		fscanf(file, "%s", (char*) &c);

		if (!fscanf(file, "%lf", &m->b[i])) {
			break;
		}
	}

	//read expected results
	for (i = 0; i < size; i++) {
		if (!fscanf(file, "%lf", &m->x[i])) {
			break;
		}
	}
	fclose(file);

	return m;
}


bool is_mm(char* filename) {
	char *dot = strrchr(filename, '.');
	return dot && !strcmp(dot, ".mtx");
}

matrix* matrix_load(char* filename) {
	if (is_mm(filename)) {
		return matrix_load_mm(filename);
	}
	return matrix_load_original(filename);
}

void matrix_destroy(matrix* m) {
	//con array separati non serve più il workaround "solo linux":
	//ogni blocco è stato allocato singolarmente ed è sicuro liberarlo ovunque.
	free(m->row_ptr);
	free(m->col_idx);
	free(m->values);
	free(m->inv_diag);
	free(m->b);
	free(m->x);
	free(m);
}

void matrix_print(matrix *m) {
	int i, idx, s = m->size;
	puts("aqui");
	for (i = 0; i < s && i < 200; i++) {
		double diag = (m->inv_diag[i] != 0.0) ? (1.0 / m->inv_diag[i]) : 0.0;
		printf("(%i,%i)=%f ", i, i, diag);
		int start = m->row_ptr[i];
		int end = m->row_ptr[i + 1];
		for (idx = start; idx < end && idx < start + 200; idx++) {
			printf("(%i,%i)=%f ", i, m->col_idx[idx], m->values[idx]);
		}
		printf("= %f\n", m->b[i]);
	}
	for (i = 0; i < s && i < 200; i++) {
		if (i > 0) printf(", ");
		printf("%f", m->x[i]);
	}
	puts("\n");
}