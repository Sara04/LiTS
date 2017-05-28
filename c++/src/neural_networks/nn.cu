/*
 * nn.cu
 *
 *  Created on: Apr 18, 2017
 *      Author: sara
 */

#include "nn.cuh"
#include "nn_kernels.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>

# define MAX_THREADS 1024

/******************************************************************************
 * data_buffer_split: creating data buffer split, according to the input data
 * sizes and layers' outputs
 *
 * Arguments:
 *      data_S: input data size
 *      layers: vector of nn layers (possible: fcn, cnn, pool)
 *      W_sizes: sizes of fcn or/and cnn kernels' weights
 *      b_sizes: sizes of fcn or/and cnn biases
 *      pool_sizes: sizes of pool strides
 *      data_split: array to store data buffer split
 *****************************************************************************/
void data_buffer_split(unsigned *data_S, std::vector<std::string> layers,
                       unsigned **W_sizes, unsigned **pool_sizes,
                       unsigned *data_split)
{
    unsigned idx = 0;
    unsigned idx_w = 0;
    unsigned idx_p = 0;
    unsigned int w_in = data_S[0];
    unsigned int h_in = data_S[1];
    unsigned int d_in = data_S[2];
    data_split[idx] = w_in * h_in * d_in;
    idx += 1;
    for(unsigned int i = 0; i < layers.size(); i++)
    {
        if(!strcmp(layers.at(i).c_str(), "fcn"))
        {
            data_split[idx] = W_sizes[idx_w][1];
            idx_w += 1;
            idx += 1;
        }
        if(!strcmp(layers.at(i).c_str(), "conv"))
        {
            w_in = (w_in - W_sizes[idx_w][0] + 1);
            h_in = (h_in - W_sizes[idx_w][1] + 1);
            d_in  = W_sizes[idx_w][3];
            data_split[idx] = w_in * h_in * d_in;
            idx_w += 1;
            idx += 1;
        }
        if(!strcmp(layers.at(i).c_str(), "pool"))
        {
            w_in = w_in / pool_sizes[idx_p][0];
            h_in = h_in / pool_sizes[idx_p][1];
            d_in = d_in / pool_sizes[idx_p][2];
            idx_p += 1;
            idx += 1;
        }
    }
}

/******************************************************************************
 * transfer_trainable_parameters_to_gpu: transferring all trainable parameters
 * from the cput to the gpu (kernels' weights and biases)
 *
 * Arguments:
 *      layers: vector of layers that are part of network
 *      weights: pointer to the pointers to the fcn or/and cnn kernels' weights
 *      biases: pointer to the pointers to the fcn or/and cnn biases
 *      W_sizes: pointer to the kernels' sizes
 *      b_sizes: pointer to the biases' sizes
 *      weights_d: pointer to the pointer to the weights at gpu
 *      biases_d: pointer to the pointer to the biases at gpu
 *      delta_weights_d: pointer to the pointer to the weights' updates at gpu
 *      delta_biases_d: pointer to the pointer to the biases' updates at gpu
 *****************************************************************************/
void transfer_trainable_parameters_to_gpu_(std::vector<std::string> layers,
                                           float **weights, unsigned **W_sizes,
                                           float **biases, unsigned **b_sizes,
                                           float **weights_d, float **biases_d,
                                           float **delta_weights_d,
                                           float **delta_biases_d)
{
    /**************************************************************************
     * 1. Count trainable weights and biases
    **************************************************************************/
    long weights_N = 0;
    long biases_N = 0;
    unsigned idx = 0;
    for(unsigned int i = 0; i < layers.size(); i++)
        if(!strcmp(layers.at(i).c_str(), "fcn") or
           !strcmp(layers.at(i).c_str(), "conv"))
        {
            weights_N += (W_sizes[idx][0] * W_sizes[idx][1] *
                          W_sizes[idx][2] * W_sizes[idx][3]);
            biases_N += b_sizes[idx][0];
            idx += 1;
        }
    /**************************************************************************
     * 2. Allocate memory on gpu for the trainable parameters and its updates
    **************************************************************************/
    cudaMalloc((void **)weights_d, weights_N * sizeof(float));
    cudaMalloc((void **)biases_d, biases_N * sizeof(float));
    cudaMalloc((void **)delta_weights_d, weights_N * sizeof(float));
    cudaMalloc((void **)delta_biases_d, biases_N * sizeof(float));
    /**************************************************************************
     * 3. Transfer trainable parameters
    **************************************************************************/
    unsigned tmp_w = 0;
    unsigned tmp_b = 0;
    weights_N = 0;
    biases_N = 0;
    idx = 0;
    for(unsigned int i = 0; i < layers.size(); i++)
        if(!strcmp(layers.at(i).c_str(), "fcn") or
           !strcmp(layers.at(i).c_str(), "conv"))
        {
            tmp_w = W_sizes[idx][0] * W_sizes[idx][1] *
                    W_sizes[idx][2] * W_sizes[idx][3];
            cudaMemcpy(&((*weights_d)[weights_N]), weights[idx],
                       tmp_w * sizeof(float), cudaMemcpyHostToDevice);
            weights_N += tmp_w;
            tmp_b = b_sizes[idx][0];
            cudaMemcpy(&((*biases_d)[biases_N]), biases[idx],
                       tmp_b * sizeof(float), cudaMemcpyHostToDevice);
            biases_N += tmp_b;
            idx += 1;
        }
}

/******************************************************************************
 * transfer_trainable_parameters_to_cpu: transferring all trainable parameters
 * from the gpu to the cpu (kernels' weights and biases)
 *
 * Arguments:
 *      layers: vector of layers that are part of network
 *      weights: pointer to the pointers to the fcn or/and cnn kernels' weights
 *      biases: pointer to the pointers to the fcn or/and cnn biases
 *      W_sizes: pointer to the kernels' sizes
 *      b_sizes: pointer to the biases' sizes
 *      weights_d: pointer to the pointer to the weights at gpu
 *      biases_d: pointer to the pointer to the biases at gpu
 *****************************************************************************/
void transfer_trainable_parameters_to_cpu_(std::vector<std::string> layers,
                                           float **weights, unsigned **W_sizes,
                                           float **biases, unsigned **b_sizes,
                                           float **weights_d, float **biases_d)
{

    /**************************************************************************
     * 1. Transfer trainable parameters
    **************************************************************************/
    unsigned tmp_w = 0;
    unsigned tmp_b = 0;
    unsigned weights_N = 0;
    unsigned biases_N = 0;
    unsigned idx = 0;
    for(unsigned int i = 0; i < layers.size(); i++)
        if(!strcmp(layers.at(i).c_str(), "fcn") or
           !strcmp(layers.at(i).c_str(), "conv"))
        {
            tmp_w = W_sizes[idx][0] * W_sizes[idx][1] *
                    W_sizes[idx][2] * W_sizes[idx][3];
            cudaMemcpy(weights[idx], &weights_d[0][weights_N],
                       tmp_w * sizeof(float), cudaMemcpyDeviceToHost);
            weights_N += tmp_w;

            tmp_b = b_sizes[idx][0];
            cudaMemcpy(biases[idx], &biases_d[0][biases_N],
                       tmp_b * sizeof(float), cudaMemcpyDeviceToHost);
            biases_N += tmp_b;

            idx += 1;
        }
}

/******************************************************************************
 * propagate_forward_gpu_train: propagate training images trough the network
 * and save all neuron inputs and outputs for the further training
 *
 * Arguments:
 *      train_imgs: input training images
 *      train_S: size of the input data
 *      train_neuron_in_d: pointer to the pointer to the array where  neuron
 *          inputs would be placed on the gpu
 *      train_neuron_out_d: pointer to the pointer to the array where neuron
 *          outputs would be placed on the gpu
 *      layers: vector of model's layers
 *      weights_d: pointer to the pointer to the array of kernels' weights
 *          on the gpu
 *      W_sizes: pointer to the kernels' sizes
 *      biases_d: pointer to the pointer to the array of biases on the gpu
 *      b_sizes: pointer to the biases' sizes
 *      pool_sizes: pool strides' sizes
 *****************************************************************************/
void propagate_forward_gpu_train(float *train_imgs, unsigned *train_S,
                                 float **train_neuron_in_d,
                                 float **train_neuron_out_d,
                                 std::vector<std::string> layers,
                                 float **weights_d, unsigned **W_sizes,
                                 float **biases_d, unsigned **b_sizes,
                                 unsigned **pool_sizes)
{
    /**************************************************************************
     *  1. Allocate and transfer training data to gpu
     *************************************************************************/
    unsigned *train_split = new unsigned[layers.size() + 1];
    data_buffer_split(train_S, layers, W_sizes, pool_sizes, train_split);
    unsigned int neuron_out_len = 0;
    unsigned int neuron_in_len  = 0;

    for(unsigned int i = 0; i < (layers.size() + 1); i++)
    {
        neuron_out_len += train_split[i];
        if(i > 0)
            neuron_in_len += train_split[i];
    }
    neuron_out_len *= train_S[3];
    neuron_in_len *= train_S[3];
    cudaMalloc((void **)train_neuron_out_d, neuron_out_len * sizeof(float));
    cudaMemcpy(train_neuron_out_d[0], train_imgs,
               train_split[0] * train_S[3] * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMalloc((void **)train_neuron_in_d, neuron_in_len * sizeof(float));

    /**************************************************************************
     *  2. Propagate training data trough layers and store neuron inputs and
     *  outputs
     *************************************************************************/
    long *start_ni = new long[layers.size() + 1];
    long *start_no = new long[layers.size() + 1];
    long *start_w = new long[layers.size() + 1];
    long *start_b = new long[layers.size() + 1];
    start_ni[0] = 0;
    start_no[0] = 0;
    start_w[0] = 0;
    start_b[0] = 0;

    for(unsigned int i = 0; i < layers.size(); i++)
    {
        if(!strcmp(layers.at(i).c_str(), "fcn"))
        {

            dim3 th_per_block(BLOCK_SIZE, BLOCK_SIZE);

            unsigned n_blocks_1 = W_sizes[i][1] / BLOCK_SIZE;
            if (W_sizes[i][1] > n_blocks_1 * BLOCK_SIZE)
                n_blocks_1 += 1;
            unsigned n_blocks_2 = train_S[3] / BLOCK_SIZE;
            if (train_S[3] > n_blocks_2 *BLOCK_SIZE)
                n_blocks_2 += 1;
            dim3 blc_per_grid(n_blocks_1, n_blocks_2);

            propagate_forward_fcn_gpu_train<<<blc_per_grid, th_per_block>>>(
                    train_neuron_in_d[0], train_neuron_out_d[0],
                    weights_d[0], biases_d[0], train_S[3],
                    W_sizes[i][0], W_sizes[i][1],
                    start_ni[i], start_no[i], start_w[i], start_b[i]);

            start_ni[i + 1] = (start_ni[i] + train_S[3] * W_sizes[i][1]);
            start_no[i + 1] = (start_no[i] + train_S[3] * W_sizes[i][0]);
            start_w[i + 1] = (start_w[i] + W_sizes[i][0] * W_sizes[i][1]);
            start_b[i + 1] = (start_b[i] + W_sizes[i][1]);
        }
    }
    /**************************************************************************
     *  3. Release memory
     *************************************************************************/
    delete [] train_split;
    delete [] start_ni;
    delete [] start_no;
    delete [] start_w;
    delete [] start_b;
}

/******************************************************************************
 * propagate_forward_gpu_test: propagate testing images trough the network
 * and save all neuron outputs
 *
 * Arguments:
 *      test_imgs: input testing images
 *      test_S: size of the input data
 *      train_neuron_out_d: pointer to the pointer to the array where neuron
 *          outputs would be placed on the gpu
 *      layers: vector of model's layers
 *      weights_d: pointer to the pointer to the array of kernels' weights
 *          on the gpu
 *      W_sizes: pointer to the kernels' sizes
 *      biases_d: pointer to the pointer to the array of biases on the gpu
 *      b_sizes: pointer to the biases' sizes
 *      pool_sizes: pool strides' sizes
 *      scores: pointer to the array where the output scores would be stored
 *****************************************************************************/
void propagate_forward_gpu_test(float *test_imgs, unsigned *test_S,
                                float **test_neuron_out_d,
                                std::vector<std::string> layers,
                                float **weights_d, unsigned **W_sizes,
                                float **biases_d, unsigned **b_sizes,
                                unsigned **pool_sizes,
                                float *scores)
{
    /**************************************************************************
     *  1. Allocate and transfer training data to gpu
     *************************************************************************/
    unsigned *test_split = new unsigned[layers.size() + 1];
    data_buffer_split(test_S, layers, W_sizes, pool_sizes, test_split);
    unsigned int neuron_out_len = 0;

    for(unsigned int i = 0; i < (layers.size() + 1); i++)
        neuron_out_len += test_split[i];
    neuron_out_len *= test_S[3];

    cudaMalloc((void **)test_neuron_out_d, neuron_out_len * sizeof(float));
    cudaMemcpy(test_neuron_out_d[0], test_imgs,
               test_split[0] * test_S[3] * sizeof(float),
               cudaMemcpyHostToDevice);

    /**************************************************************************
     *  2. Propagate training data trough layers and store neuron inputs and
     *  outputs
     *************************************************************************/
    long *start_ni = new long[layers.size() + 1];
    long *start_no = new long[layers.size() + 1];
    long *start_w = new long[layers.size() + 1];
    long *start_b = new long[layers.size() + 1];
    start_ni[0] = 0;
    start_no[0] = 0;
    start_w[0] = 0;
    start_b[0] = 0;

    for(unsigned int i = 0; i < layers.size(); i++)
    {
        if(!strcmp(layers.at(i).c_str(), "fcn"))
        {

            dim3 th_per_block(BLOCK_SIZE, BLOCK_SIZE);

            unsigned n_blocks_1 = W_sizes[i][1] / BLOCK_SIZE;
            if (W_sizes[i][1] > n_blocks_1 * BLOCK_SIZE)
                n_blocks_1 += 1;
            unsigned n_blocks_2 = test_S[3] / BLOCK_SIZE;
            if (test_S[3] > n_blocks_2 *BLOCK_SIZE)
                n_blocks_2 += 1;
            dim3 blc_per_grid(n_blocks_1, n_blocks_2);

            propagate_forward_fcn_gpu_test<<<blc_per_grid, th_per_block>>>(
                    test_neuron_out_d[0],
                    weights_d[0], biases_d[0], test_S[3],
                    W_sizes[i][0], W_sizes[i][1],
                    start_ni[i], start_no[i], start_w[i], start_b[i]);

            start_ni[i + 1] = (start_ni[i] + test_S[3] * W_sizes[i][1]);
            start_no[i + 1] = (start_no[i] + test_S[3] * W_sizes[i][0]);
            start_w[i + 1] = (start_w[i] + W_sizes[i][0] * W_sizes[i][1]);
            start_b[i + 1] = (start_b[i] + W_sizes[i][1]);
        }
    }
    cudaMemcpy(scores, &test_neuron_out_d[0][start_no[layers.size()]],
               test_S[3] * b_sizes[layers.size() - 1][0] * sizeof(float),
               cudaMemcpyDeviceToHost);

    /**************************************************************************
     *  3. Release memory
     *************************************************************************/
    cudaFree(test_neuron_out_d[0]);
    cudaFree(test_neuron_out_d);
    delete [] test_split;
    delete [] start_ni;
    delete [] start_no;
    delete [] start_w;
    delete [] start_b;
}

/******************************************************************************
 * compute_error_gpu: propagate testing images trough the network
 * and save all neuron outputs
 *
 * Arguments:
 *      data_gt: data ground truth
 *      data_S: data size
 *      data_neuron_in_d: pointer to the pointer to the array where  neuron
 *          inputs would be placed on the gpu
 *      data_neuron_out_d: pointer to the pointer to the array where neuron
 *          outputs would be placed on the gpu
 *      weights_d: pointer to the pointer to the weights at gpu
 *      delta_weights_d: pointer to the pointer to the weights' updates at gpu
 *      W_sizes: pointer to the kernels' sizes
 *      biases_d: pointer to the pointer to the biases at gpu
 *      delta_biases_d: pointer to the pointer to the biases' updates at gpu
 *      b_sizes: pointer to the biases' sizes
 *      pool_sizes: pool strides' sizes
 *****************************************************************************/
float compute_error_gpu(float *data_gt, unsigned *data_S,
                        float **data_neuron_inputs_d,
                        float **data_neuron_outputs_d,
                        std::vector<std::string> layers,
                        float **weights_d, float **delta_weights_d,
                        unsigned **W_sizes,
                        float **biases_d, float **delta_biases_d,
                        unsigned **b_sizes,
                        unsigned **pool_sizes)
{

    /**************************************************************************
     * 1. Count number of kernels's weights, biases, input and output places
     *************************************************************************/
    long weights_N = 0;
    long biases_N = 0;
    long inputs_N = 0;
    long delta_N = 0;
    long outputs_N = data_S[0] * data_S[1] * data_S[2] * data_S[3];
    unsigned idx = 0;
    for(unsigned int i = 0; i < layers.size(); i++)
        if(!strcmp(layers.at(i).c_str(), "fcn") or
           !strcmp(layers.at(i).c_str(), "conv"))
        {
            weights_N += (W_sizes[idx][0] * W_sizes[idx][1] *
                          W_sizes[idx][2] * W_sizes[idx][3]);
            biases_N += b_sizes[idx][0];

            inputs_N += data_S[3] * W_sizes[idx][1];
            idx += 1;
        }
    outputs_N += inputs_N;
    delta_N = biases_N * data_S[3];
    /**************************************************************************
     * 2. Transfer ground truth to the gpu
     *************************************************************************/
    float *data_gt_d;
    cudaMalloc((void **)&data_gt_d,
               data_S[3] * W_sizes[layers.size() - 1][1] * sizeof(float));
    cudaMemcpy(data_gt_d, data_gt,
               data_S[3] * W_sizes[layers.size() - 1][1] * sizeof(float),
               cudaMemcpyHostToDevice);
    /**************************************************************************
     * 3. Compute the average error
     *************************************************************************/
    float *avg_error = new float[1];
    avg_error[0] = 0;
    float *avg_error_d;
    cudaMalloc((void **)&avg_error_d, sizeof(float));
    cudaMemcpy(avg_error_d, avg_error, sizeof(float), cudaMemcpyHostToDevice);

    evaluate_error<<<data_S[3], W_sizes[layers.size() - 1][1]>>>
            (data_gt_d, data_S[3], W_sizes[layers.size() - 1][1],
             data_neuron_outputs_d[0], outputs_N,
             avg_error_d);
    cudaMemcpy(avg_error, avg_error_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(data_gt_d);
    cudaFree(data_neuron_inputs_d[0]);
    cudaFree(data_neuron_outputs_d[0]);
    cudaFree(data_neuron_inputs_d);
    cudaFree(data_neuron_outputs_d);

    float avg_p = avg_error[0];
    delete [] avg_error;

    return avg_p;
}

/******************************************************************************
 * propagate_backwards_gpu_train: propagate training images trough the network
 * and save all neuron inputs and outputs for the further training
 *
 * Arguments:
 *      data_gt: data ground truth
 *      data_S: data size
 *      train_neuron_in_d: pointer to the pointer to the array where  neuron
 *          inputs would be placed on the gpu
 *      train_neuron_out_d: pointer to the pointer to the array where neuron
 *          outputs would be placed on the gpu
 *      layers: vector of model's layers
 *      weights_d: pointer to the pointer to the weights at gpu
 *      delta_weights_d: pointer to the pointer to the weights' updates at gpu
 *      W_sizes: pointer to the kernels' sizes
 *      biases_d: pointer to the pointer to the biases at gpu
 *      delta_biases_d: pointer to the pointer to the biases' updates at gpu
 *      b_sizes: pointer to the biases' sizes
 *      pool_sizes: pool strides' sizes
 *      learning_rate: rate by which trainable parameters will be updated
 *****************************************************************************/
float propagate_backwards_gpu_train(float *data_gt, unsigned *data_S,
                                    float **train_neuron_inputs_d,
                                    float **train_neuron_outputs_d,
                                    std::vector<std::string> layers,
                                    float **weights_d,
                                    float **delta_weights_d,
                                    unsigned **W_sizes,
                                    float **biases_d,
                                    float **delta_biases_d,
                                    unsigned **b_sizes,
                                    unsigned **pool_sizes,
                                    float learning_rate)
{

    /**************************************************************************
     * 1. Allocate memory for weights and biases on GPU and initialize
     *  them with zeros.
     *************************************************************************/
    long weights_N = 0;
    long biases_N = 0;
    long inputs_N = 0;
    long delta_N = 0;
    long outputs_N = data_S[0] * data_S[1] * data_S[2] * data_S[3];
    unsigned idx = 0;
    for(unsigned int i = 0; i < layers.size(); i++)
        if(!strcmp(layers.at(i).c_str(), "fcn") or
           !strcmp(layers.at(i).c_str(), "conv"))
        {
            weights_N += (W_sizes[idx][0] * W_sizes[idx][1] *
                          W_sizes[idx][2] * W_sizes[idx][3]);
            biases_N += b_sizes[idx][0];

            inputs_N += data_S[3] * W_sizes[idx][1];
            idx += 1;
        }
    outputs_N += inputs_N;
    delta_N = biases_N * data_S[3];
    /**************************************************************************
     * 2. Transfer ground truth to gpu
     *************************************************************************/
    float *data_gt_d;
    cudaMalloc((void **)&data_gt_d,
               data_S[3] * W_sizes[layers.size() - 1][1] * sizeof(float));
    cudaMemcpy(data_gt_d, data_gt,
               data_S[3] * W_sizes[layers.size() - 1][1] * sizeof(float),
               cudaMemcpyHostToDevice);
    /**************************************************************************
     * 3.
     *************************************************************************/
    float *delta_x_d;
    cudaMalloc((void **)&delta_x_d, biases_N * data_S[3] * sizeof(float));

    cost_function_backprop<<<data_S[3], W_sizes[layers.size() - 1][1]>>>
            (data_gt_d,
             train_neuron_inputs_d[0], inputs_N,
             train_neuron_outputs_d[0], outputs_N,
             delta_x_d, delta_N,
             data_S[3], W_sizes[layers.size() - 1][1]);

    unsigned N_tot = delta_N;

    for(unsigned i = (layers.size() - 1); i > 0; i--)
    {
        dim3 th_per_block(BLOCK_SIZE, BLOCK_SIZE);
        unsigned n_blocks_1 = W_sizes[i][0] / BLOCK_SIZE;
        if (W_sizes[i][1] > n_blocks_1 * BLOCK_SIZE)
            n_blocks_1 += 1;

        unsigned n_blocks_2 = data_S[3] / BLOCK_SIZE;
        if (data_S[3] > n_blocks_2 *BLOCK_SIZE)
            n_blocks_2 += 1;

        dim3 blc_per_grid(n_blocks_1, n_blocks_2);

        weights_N -= (W_sizes[i][0] * W_sizes[i][1]);
        delta_N -= (data_S[3] * W_sizes[i][1]);
        inputs_N -= (data_S[3] * W_sizes[i][1]);

        backpropagate_fcn_gpu_train<<<blc_per_grid, th_per_block>>>
                (weights_d[0], weights_N,
                 delta_x_d, delta_N,
                 train_neuron_inputs_d[0], inputs_N,
                 data_S[3], W_sizes[i][0], W_sizes[i][1]);
    }
    /**************************************************************************
     * 4. Compute error
     *************************************************************************/
    float *avg_error = new float[1];
    avg_error[0] = 0;
    float *avg_error_d;
    cudaMalloc((void **)&avg_error_d, sizeof(float));
    cudaMemcpy(avg_error_d, avg_error, sizeof(float), cudaMemcpyHostToDevice);

    evaluate_error<<<data_S[3], W_sizes[layers.size() - 1][1]>>>
            (data_gt_d, data_S[3], W_sizes[layers.size() - 1][1],
             train_neuron_outputs_d[0], outputs_N,
             avg_error_d);
    cudaMemcpy(avg_error, avg_error_d, sizeof(float), cudaMemcpyDeviceToHost);
    /**************************************************************************
     * 5. Update weights and biases
     *************************************************************************/
    weights_N = 0;
    biases_N = 0;
    outputs_N = 0;
    delta_N = 0;
    for(unsigned int i = 0; i < layers.size(); i++)
    {
        if(!strcmp(layers.at(i).c_str(), "fcn"))
        {
            dim3 th_per_block(BLOCK_SIZE, BLOCK_SIZE);

            unsigned n_blocks_1 = W_sizes[i][1] / BLOCK_SIZE;
            if (W_sizes[i][1] > n_blocks_1 * BLOCK_SIZE)
                n_blocks_1 += 1;
            unsigned n_blocks_2 = W_sizes[i][0] / BLOCK_SIZE;
            if (W_sizes[i][0] > n_blocks_2 *BLOCK_SIZE)
                n_blocks_2 += 1;
            dim3 blc_per_grid(n_blocks_1, n_blocks_2);

            update_weights_and_biases<<<blc_per_grid, th_per_block>>>
                    (weights_d[0], biases_d[0], weights_N, biases_N,
                     W_sizes[i][0], W_sizes[i][1],
                     train_neuron_outputs_d[0], outputs_N,
                     delta_x_d, delta_N,
                     data_S[3], learning_rate);

            weights_N += (W_sizes[i][0] *  W_sizes[i][1]);
            biases_N += W_sizes[i][1];
            outputs_N += W_sizes[i][0] * data_S[3];
            delta_N += W_sizes[i][1] * data_S[3];
        }
    }
    cudaFree(data_gt_d);
    cudaFree(delta_x_d);
    cudaFree(train_neuron_inputs_d[0]);
    cudaFree(train_neuron_outputs_d[0]);
    cudaFree(train_neuron_inputs_d);
    cudaFree(train_neuron_outputs_d);

    float avg_p = avg_error[0];
    delete [] avg_error;

    return avg_p;
}


