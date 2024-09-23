#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define NUM_THREADS 35

typedef struct {
    int rows;        // users
    int cols;        // products
    int nnz;         // # of non-zero vals
    double *values;  // ratings (non-zero)
    int *col_index;  // product index
    int *row_ptr;    // user_ratings pointer
} CSRMatrix;

// Thread argument structure
typedef struct {
    CSRMatrix* matrix;
    double** similarity_matrix;
    double* norms;
    int start_product;
    int end_product;
} ThreadData;

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

// l2 norm of a column in the csr matrix (product)
double compute_l2_norm(CSRMatrix* matrix, int col) {
    double norm = 0.0;
    for (int i = 0; i < matrix->rows; i++) {
        int start = matrix->row_ptr[i];
        int end = matrix->row_ptr[i + 1];
        for (int j = start; j < end; j++) {
            if (matrix->col_index[j] == col) {
                norm += matrix->values[j] * matrix->values[j];
            }
        }
    }
    return sqrt(norm);
}

// compute dot product of two columns (products!!!!)
double dot_product(CSRMatrix* matrix, int col1, int col2) {
    double dot = 0.0;
    for (int i = 0; i < matrix->rows; i++) {
        int start = matrix->row_ptr[i];
        int end = matrix->row_ptr[i + 1];
        double val1 = 0.0, val2 = 0.0;
        for (int j = start; j < end; j++) {
            if (matrix->col_index[j] == col1) {
                val1 = matrix->values[j];
            }
            if (matrix->col_index[j] == col2) {
                val2 = matrix->values[j];
            }
        }
        dot += val1 * val2;
    }
    return dot;
}

// threaded compute of pairwise cosine similarity for a subset of products
void* thread_compute_similarity(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    CSRMatrix* matrix = data->matrix;
    double** similarity_matrix = data->similarity_matrix;
    double* norms = data->norms;
    int start = data->start_product;
    int end = data->end_product;

    for (int i = start; i < end; i++) {
        for (int j = i; j < matrix->cols; j++) {
            if (norms[i] == 0 || norms[j] == 0) {
                similarity_matrix[i][j] = 0.0;
                similarity_matrix[j][i] = 0.0;
            } else {
                double dot = dot_product(matrix, i, j);
                similarity_matrix[i][j] = dot / (norms[i] * norms[j]);
                similarity_matrix[j][i] = similarity_matrix[i][j];  // symmetry <|>_<|>
            }
        }
    }

    pthread_exit(NULL);
}

// pairwise cosine sim multi threaded
double** pairwise_cosine_similarity_multithreaded(CSRMatrix* matrix) {
    double** similarity_matrix = (double**)malloc(matrix->cols * sizeof(double*));
    for (int i = 0; i < matrix->cols; i++) {
        similarity_matrix[i] = (double*)malloc(matrix->cols * sizeof(double));
    }

    // l2
    double* norms = (double*)malloc(matrix->cols * sizeof(double));
    for (int i = 0; i < matrix->cols; i++) {
        norms[i] = compute_l2_norm(matrix, i);
    }

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    int products_per_thread = matrix->cols / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        thread_data[t].matrix = matrix;
        thread_data[t].similarity_matrix = similarity_matrix;
        thread_data[t].norms = norms;
        thread_data[t].start_product = t * products_per_thread;
        thread_data[t].end_product = (t == NUM_THREADS - 1) ? matrix->cols : (t + 1) * products_per_thread;
        pthread_create(&threads[t], NULL, thread_compute_similarity, (void*)&thread_data[t]);
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
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

int main() {
    const char* input_filename = "csr_matrix.txt";
    const char* output_filename = "similarity_matrix.txt";

    CSRMatrix* matrix = read_csr_matrix(input_filename);

    printf("jsut read in\n");

    double** similarity_matrix = pairwise_cosine_similarity_multithreaded(matrix);

    save_similarity_matrix(output_filename, similarity_matrix, matrix->cols);

    // free the memory
    free_similarity_matrix(similarity_matrix, matrix->cols);
    free_csr_matrix(matrix);

    printf("Pairwise cosine similarity matrix has been saved to '%s'.\n", output_filename);

    return 0;
}
