#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int rows;        // users
    int cols;        // products
    int nnz;         // # of non-zero vals
    double *values;  // ratings (non-zero)
    int *col_index;  // product index
    int *row_ptr;    // user_ratings pointer
} CSRMatrix;

CSRMatrix* read_csr_matrix(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    CSRMatrix* matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    fscanf(file, "%d %d %d", &matrix->rows, &matrix->cols, &matrix->nnz);

    matrix->values = (double*)malloc(matrix->nnz * sizeof(double));
    matrix->col_index = (int*)malloc(matrix->nnz * sizeof(int));
    matrix->row_ptr = (int*)malloc((matrix->rows + 1) * sizeof(int));

    for (int i = 0; i < matrix->nnz; i++) {
        fscanf(file, "%lf", &matrix->values[i]);
    }
    for (int i = 0; i < matrix->nnz; i++) {
        fscanf(file, "%d", &matrix->col_index[i]);
    }
    for (int i = 0; i <= matrix->rows; i++) {
        fscanf(file, "%d", &matrix->row_ptr[i]);
    }

    fclose(file);
    return matrix;
}

double compute_l2_norm(CSRMatrix* matrix, int row) {
    double norm = 0.0;
    int start = matrix->row_ptr[row];
    int end = matrix->row_ptr[row + 1];
    for (int i = start; i < end; i++) {
        norm += matrix->values[i] * matrix->values[i];
    }
    return sqrt(norm);
}

double dot_product(CSRMatrix* matrix, int row1, int row2) {
    int i = matrix->row_ptr[row1];
    int j = matrix->row_ptr[row2];
    double dot = 0.0;

    while (i < matrix->row_ptr[row1 + 1] && j < matrix->row_ptr[row2 + 1]) {
        if (matrix->col_index[i] == matrix->col_index[j]) {
            dot += matrix->values[i] * matrix->values[j];
            i++;
            j++;
        } else if (matrix->col_index[i] < matrix->col_index[j]) {
            i++;
        } else {
            j++;
        }
    }

    return dot;
}

double** pairwise_cosine_similarity(CSRMatrix* matrix) {
    double** similarity_matrix = (double**)malloc(matrix->rows * sizeof(double*));
    for (int i = 0; i < matrix->rows; i++) {
        similarity_matrix[i] = (double*)malloc(matrix->rows * sizeof(double));
    }

    double* norms = (double*)malloc(matrix->rows * sizeof(double));
    for (int i = 0; i < matrix->rows; i++) {
        norms[i] = compute_l2_norm(matrix, i);
    }

    // cosine similarity
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = i; j < matrix->rows; j++) {
            if (norms[i] == 0 || norms[j] == 0) {
                similarity_matrix[i][j] = 0.0;
                similarity_matrix[j][i] = 0.0;
            } else {
                double dot = dot_product(matrix, i, j);
                similarity_matrix[i][j] = dot / (norms[i] * norms[j]);
                similarity_matrix[j][i] = similarity_matrix[i][j];  // Symmetry
            }
        }
    }

    free(norms);
    return similarity_matrix;
}

void save_similarity_matrix(const char* filename, double** similarity_matrix, int rows) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            fprintf(file, "%.4f ", similarity_matrix[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void free_similarity_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void free_csr_matrix(CSRMatrix* matrix) {
    free(matrix->values);
    free(matrix->col_index);
    free(matrix->row_ptr);
    free(matrix);
}

int main(int argc, char **argv) {
    printf("reading in files\n");
    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    CSRMatrix* matrix = read_csr_matrix(input_filename);
    printf("read in the matrix!!! time to jack in\n");

    double** similarity_matrix = pairwise_cosine_similarity(matrix);

    save_similarity_matrix(output_filename, similarity_matrix, matrix->rows);

    free_similarity_matrix(similarity_matrix, matrix->rows);
    free_csr_matrix(matrix);

    printf("cosine similarity matrix has been saved to '%s'.\n", output_filename);

    return 0;
}
