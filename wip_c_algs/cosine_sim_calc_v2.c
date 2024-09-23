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

typedef struct {
    CSRMatrix* matrix;
    double* similarity;
    int* target_items;
    int num_target_items;
    int start_item;
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
    double* similarity = data->similarity;
    int* target_items = data->target_items;
    int num_target_items = data->num_target_items;
    int start_item = data->start_item;

    for (int i = start_item; i < num_target_items; i++) {
        int item1 = target_items[i];
        double norm1 = compute_l2_norm(matrix, item1);

        for (int j = 0; j < matrix->cols; j++) {
            double norm2 = compute_l2_norm(matrix, j);
            if (norm1 == 0 || norm2 == 0) {
                similarity[i * matrix->cols + j] = 0.0;
            } else {
                double dot = dot_product(matrix, item1, j);
                similarity[i * matrix->cols + j] = dot / (norm1 * norm2);
            }
        }
    }

    pthread_exit(NULL);
}

// pairwise cosine sim multi threaded
double* pairwise_cosine_similarity_for_items(CSRMatrix* matrix, int* target_items, int num_target_items) {
    double* similarity = (double*)malloc(num_target_items * matrix->cols * sizeof(double));

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    int items_per_thread = num_target_items / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        thread_data[t].matrix = matrix;
        thread_data[t].similarity = similarity;
        thread_data[t].target_items = target_items;
        thread_data[t].num_target_items = num_target_items;
        thread_data[t].start_item = t * items_per_thread;
        pthread_create(&threads[t], NULL, thread_compute_similarity, (void*)&thread_data[t]);
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    return similarity;
}

void save_similarity_array(const char* filename, double* similarity, int num_target_items, int total_items) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file for writing.\n");
        exit(1);
    }

    for (int i = 0; i < num_target_items; i++) {
        for (int j = 0; j < total_items; j++) {
            fprintf(file, "%.4f ", similarity[i * total_items + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void free_similarity_array(double* array) {
    free(array);
}

void free_csr_matrix(CSRMatrix* matrix) {
    free(matrix->values);
    free(matrix->col_index);
    free(matrix->row_ptr);
    free(matrix);
}

int main() {
    const char* input_filename = "csr_matrix.txt";
    const int target_items[] = {146854, 147467,147956,253570, 253758}; // indices
    const int num_target_items = sizeof(target_items) / sizeof(target_items[0]);
    const char* output_filename = "similarity_matrix.txt";

    CSRMatrix* matrix = read_csr_matrix(input_filename);


    printf("read-in csr matrix \n");
    double* similarity_matrix = pairwise_cosine_similarity_for_items(matrix, (int*)target_items, num_target_items);

    printf("calc'd sim matrix \n");
    save_similarity_array(output_filename, similarity_matrix, num_target_items, matrix->cols);

    printf("saved that bad boy\n");
    free_similarity_array(similarity_matrix);
    free_csr_matrix(matrix);

    printf("Pairwise cosine similarity matrix for specified items against all items has been saved to '%s'.\n", output_filename);

    return 0;
}

    
