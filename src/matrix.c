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
 */
matrix* matrix_load_mm(char* filename) {

    long i, size, size_cols, count;
    long row, col;
    double val;

    bool is_symmetric = false;
    char line[1024];
    long file_pos;

    FILE *file = fopen(filename, "r");
    if (!file) {
        puts("Cannot open file");
        exit(1);
    }

    // ----------------------------
    // SKIP HEADER MATRIX MARKET
    // ----------------------------
    do {
        file_pos = ftell(file);

        if (!fgets(line, sizeof(line), file)) {
            puts("Unexpected EOF");
            exit(1);
        }

        if (line[0] == '%') {
            if (strncmp(line, "%%MatrixMarket", 14) == 0) {
                if (strstr(line, "symmetric") ||
                    strstr(line, "hermitian")) {
                    is_symmetric = true;
                }
            }
        }

    } while (line[0] == '%');

    fseek(file, file_pos, SEEK_SET);

    if (fscanf(file, "%li %li %li", &size, &size_cols, &count) != 3) {
        puts("Invalid header");
        exit(1);
    }

    if (size != size_cols) {
        puts("Matrix not square");
        exit(1);
    }

    matrix *m = malloc(sizeof(matrix));
    m->size = size;

    m->b = malloc(size * sizeof(double));
    m->x = malloc(size * sizeof(double));

    m->a = malloc(size * sizeof(item_matrix*));

    // init pointers
    for (i = 0; i < size; i++) {
        m->a[i] = NULL;
    }

    // ----------------------------
    // TEMP STORAGE (triplets)
    // ----------------------------
    long cap = is_symmetric ? 2 * count : count;

    long *Trow = malloc(cap * sizeof(long));
    long *Tcol = malloc(cap * sizeof(long));
    double *Tval = malloc(cap * sizeof(double));

    long nnz = 0;

    // row counts for CSR-style build
    int *row_count = calloc(size, sizeof(int));

    // ----------------------------
    // READ TRIPLETS
    // ----------------------------
    for (i = 0; i < count; i++) {

        if (fscanf(file, "%li %li %lf", &row, &col, &val) != 3)
            break;

        row--; col--;

        Trow[nnz] = row;
        Tcol[nnz] = col;
        Tval[nnz] = val;
        nnz++;

        row_count[row]++;

        if (row == col) continue;

        if (is_symmetric) {
            Trow[nnz] = col;
            Tcol[nnz] = row;
            Tval[nnz] = val;
            nnz++;

            row_count[col]++;
        }
    }
	    // ----------------------------
    // ALLOCAZIONE STRUTTURA RIGHE
    // ----------------------------

    // massimo possibile (over-allocation sicura)
    long *row_size = calloc(size, sizeof(long));

    for (i = 0; i < nnz; i++) {
        row_size[Trow[i]]++;
    }

    m->a[0] = malloc((nnz + size) * sizeof(item_matrix));

    item_matrix *ptr = m->a[0];

    for (i = 1; i < size; i++) {
        m->a[i] = NULL;
    }

    // cursori per riga
    item_matrix **cursor = malloc(size * sizeof(item_matrix*));

    long offset = 0;
    for (i = 0; i < size; i++) {
        m->a[i] = ptr + offset;
        cursor[i] = m->a[i];
        offset += (row_size[i] + 1); // +1 sentinella
    }

    // reset row_size per uso come contatore
    for (i = 0; i < size; i++) {
        row_size[i] = 0;
    }

    // ----------------------------
    // FILL MATRICE
    // ----------------------------

    for (i = 0; i < nnz; i++) {

        long r = Trow[i];
        long c = Tcol[i];

        item_matrix *p = cursor[r];

        p->column = (int)c;
        p->value = Tval[i];

        cursor[r]++;
    }

    // sentinelle
    for (i = 0; i < size; i++) {
        cursor[i]->column = -1;
    }

    free(cursor);
    free(Trow);
    free(Tcol);
    free(Tval);
    free(row_count);
    free(row_size);

    // ----------------------------
    // READ / GENERATE B
    // ----------------------------

    char filename_b[256];
    strncpy(filename_b, filename, strlen(filename) - 4);
    filename_b[strlen(filename) - 4] = '\0';
    strcat(filename_b, "_b.mtx");

    if (access(filename_b, R_OK) != -1) {

        FILE *fb = fopen(filename_b, "r");

        fscanf(fb, "%li %li", &size, &size_cols);

        for (i = 0; i < m->size; i++) {
            fscanf(fb, "%lf", &m->b[i]);
        }

        fclose(fb);

    } else {

        // b = A * 1
        for (i = 0; i < m->size; i++) {

            double sum = 0.0;

            item_matrix *it = m->a[i];

            while (it->column != -1) {
                sum += it->value;
                it++;
            }

            m->b[i] = sum;
        }
    }

    // ----------------------------
    // READ X (optional)
    // ----------------------------

    char filename_x[256];
    strncpy(filename_x, filename, strlen(filename) - 4);
    filename_x[strlen(filename) - 4] = '\0';
    strcat(filename_x, "_x.mtx");

    if (access(filename_x, R_OK) != -1) {

        FILE *fx = fopen(filename_x, "r");

        fscanf(fx, "%li %li", &size, &size_cols);

        for (i = 0; i < m->size; i++) {
            fscanf(fx, "%lf", &m->x[i]);
        }

        fclose(fx);

    } else {
        for (i = 0; i < m->size; i++) {
            m->x[i] = -1.0;
        }
    }

    fclose(file);
    return m;
}

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

	//read matrix size
	fscanf(file, "%iu", &size);

	//allocates vector right "B"
	m->b = malloc(size * sizeof(double));

	//allocates vector with expected results
	m->x = malloc(size * sizeof(double));

	//allocates "rows"
	m->a = malloc(size * sizeof(item_matrix*));

	//allocates matrix contents
	m->a[0] = malloc((size*size + size) * sizeof(item_matrix));

	char c[10];

	//read rows
	item_matrix *position = m->a[0];
	for (i = 0; i < size; i++) {
		m->a[i] = position;
		//read cols
		for (j = 0; j < size; j++) {
			position->column = j;
			if (!fscanf(file, "%lf", &position->value)) {
				break;
			}
			position++;
		}
		position->column = -1;
		position++;

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

	m->size = size;
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


void matrix_destroy(matrix* matrix) {
	#ifndef __linux__
		return;
	#endif
	//int i;
	//for (i = 0; i < matrix->size; i++) {
		//free(matrix->a[i]);
	//}
	//free(matrix->a);
	//free(matrix->b);
	//free(matrix->x);
	free(matrix);
}

void matrix_print(matrix *m) {
	int i, j = 0, s = m->size;
	puts("aqui");
	for (i = 0; i < s && i < 200; i++) {
		item_matrix *item = m->a[i];
		if (item) {
			while (item->column >= 0 && j < 200) {
				j = item->column;
				printf("(%i,%i)=%f ", i, j, item->value);
				item++;
			}
		}
		printf("= %f\n", m->b[i]);
	}
	for (i = 0; i < s && i < 200; i++) {
		if (i > 0) printf(", ");
		printf("%f", m->x[i]);
	}
	puts("\n");
}