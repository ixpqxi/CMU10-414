#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;

void mat_mul(const float *A, float *B, float *R, size_t m, size_t n, size_t l)
{
    for(size_t i=0; i<m; i++)
        for(size_t j=0; j<l; j++){
            R[i*l+j] = 0;
            for(size_t k=0; k<n; k++)
                R[i*l+j] += A[i*n+k] * B[k*l+j];
        }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t iterations = (m + batch - 1) / batch;
    for(size_t iter = 0; iter < iterations; iter++){
        float *h = new float[batch*k];
        const float *X_batch = &X[iter*batch*n];

        //1 h
        mat_mul(X_batch, theta, h, batch, n, k);

        //2 Z = h
        float *exp_sum = new float[batch];
        for(size_t i=0; i<batch; i++){
            float sum = 0;
            for(size_t j=0; j<k; j++)
                sum += exp(h[i*k+j]);
            exp_sum[i] = sum;
        }
        for(size_t i=0; i<batch; i++)
            for(size_t j=0; j<k; j++)
                h[i*k+j] = exp(h[i*k+j]) / exp_sum[i];

        //3 (Z - I)
        for(size_t i=0; i<batch; i++){
            h[i*k+y[iter*batch+i]] -= 1;
        }

        //4 X_batch @ (Z - I)
        float *grad = new float[n*k];

        float *X_batch_T = new float[n*batch];
        for(size_t i=0; i<n; i++)
            for(size_t j=0; j<batch; j++)
                X_batch_T[i*batch+j] = X_batch[j*n+i];
        mat_mul(X_batch_T, h, grad, n, batch, k);

        for(size_t i=0; i<n*k; i++)
            theta[i] -= lr / batch * grad[i];
        
        delete[] h;
        delete[] exp_sum;
        delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
