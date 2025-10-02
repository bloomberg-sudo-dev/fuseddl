/*
A tiny MLP for XOR:
- layers: input(2) -> hidden(4) -> output(1)
- activation: sigmoid (hidden & output)
- training: online backpropagation with momentum (optional)
- data: 4 XOR patterns

Routines:
- forward(x -> yhat) uses params + buffers
- loss(yhat, t) returns scalar
- backward_and_update(x, t) reads buffers, writes param updates
- shuffle(indices) for training order
- init_params(scale) random init of params
- train_loop(epochs, eta, alpha, stop_threshold)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int mat_mul(const int a[3][3], const int b[3][3], int c[3][3]) // A and B are constant input matrices, C is the output matrix
{
    int i, j, k;
    // multiply matrices
    for (i = 0; i < 3; i++) { // iterate over rows of A
        for (j = 0; j < 3; j++) { // iterate over columns of B
            c[i][j] = 0; //initialize the result cell C[i][j]
            for (k = 0; k < 3; k++) { // iterate over columns of A / rows of B
                c[i][j] += a[i][k] * b[k][j]; // accumulate the dot product
            }
        }
    }
    // each element C[i][j] is the dot product of row i of A and column j of B
    return 0; // success
}

int add_mat(const int a[3][3], const int b[3][3], int c[3][3]) // add two 3x3 matrices
{
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i][j] = a[i][j] + b[i][j]; // element-wise addition
        }
    }
    return 0; // success
}

int sub_mat(const int a[3][3], const int b[3][3], int c[3][3]) // subtract two 3x3 matrices
{
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i][j] = a[i][j] - b[i][j]; // element-wise subtraction
        }
    }
    return 0; // success
}

int scalar_mul_mat(int scalar, const int a[3][3], int c[3][3]) // multiply a 3x3 matrix by a scalar
{
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i][j] = scalar * a[i][j]; // element-wise multiplication
        }
    }
    return 0; // success
}

int sigmoid_mat(const int a[3][3], int c[3][3]) // apply sigmoid function element-wise to a 3x3 matrix
{
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i][j] = 1 / (1 + exp(-a[i][j])); // sigmoid function
        }
    }
    return 0; // success
}

int forward_pass(const int x[3][1], // 1 dimensional input vector (3x1 matrix with one column)
                const int w[3][3], 
                const int b[3][1], 
                const int w2[3][3], 
                const int b2[3][1], 
                int h[3][1], 
                int yhat[3][1]) // 2-layer MLP forward pass
{
    // Layer 1: hidden layer
    int z1[3][3]; // pre-activation
    mat_mul(x, w, z1); // z1 = x * w
    add_mat(z1, b, z1); // z1 = z1 + b
    sigmoid_mat(z1, h); // h = sigmoid(z1)
    // Layer 2: output layer
    int z2[3][3]; // pre-activation
    mat_mul(h, w2, z2); // z2 = h * w2
    add_mat(z2, b2, z2); // z2 = z2 + b2    
    sigmoid_mat(z2, yhat); // yhat = sigmoid(z2)
    return 0; // success
}

int loss_mse(const int yhat[3][1], const int y[3][1]) // mean squared error loss (difference between predicted and true values)
{
    int i;
    int loss = 0;
    for (i = 0; i < 3; i++) {
        loss += (yhat[i][0] - y[i][0]) * (yhat[i][0] - y[i][0]); // accumulate squared errors
    }
    return loss / 3; // return mean
}

int print_mat(int c[3][1]) // print a 3x1 matrix
{
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            printf("%d ", c[i][j]); // returns signed integer
        } // end of row
        printf("\n");
    }
    return 0; // success
}

int main(void){
    clock_t start, end; // timer variables
    double cpu_time_used; // time used variable

    start = clock(); // start timer

    int a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int b[3][3] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int c[3][3]; // resulting matrix

    // mat_mul(a, b, c);
    // print_mat(c);

    end = clock(); // end timer
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Exec_time: %f seconds\n", cpu_time_used);

    return 0;
}