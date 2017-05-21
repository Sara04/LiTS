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

void training_buffer(unsigned *train_S, std::vector<std::string> layers,
                     unsigned **W_sizes, unsigned **pool_sizes,
                     unsigned *train_split)
{

    //1. Adding input training data
    unsigned idx = 0;
    unsigned idx_w = 0;
    unsigned idx_p = 0;
    unsigned int w_in = train_S[0];
    unsigned int h_in = train_S[1];
    unsigned int d_in = train_S[2];
    train_split[idx] = w_in * h_in * d_in;
    idx += 1;
    for(unsigned int i = 0; i < layers.size(); i++)
    {
        if(!strcmp(layers.at(i).c_str(), "fcn"))
        {
            train_split[idx] = W_sizes[idx_w][1];
            idx_w += 1;
            idx += 1;
        }

        if(!strcmp(layers.at(i).c_str(), "conv"))
        {
            w_in = (w_in - W_sizes[idx_w][0] + 1);
            h_in = (h_in - W_sizes[idx_w][1] + 1);
            d_in  = W_sizes[idx_w][3];
            train_split[idx] = w_in * h_in * d_in;
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

void transfer_trainable_parameters_to_gpu(std::vector<std::string> layers,
                                          float **weights, unsigned **W_sizes,
                                          float **biases, unsigned **b_sizes,
                                          float **weights_d, float **biases_d,
                                          float **delta_weights_d,
                                          float **delta_biases_d)
{
    // 1. Count trainable weights and biases
    // _______________________________________________________________________
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

    cudaMalloc((void **)weights_d, weights_N * sizeof(float));
    cudaMalloc((void **)biases_d, biases_N * sizeof(float));
    cudaMalloc((void **)delta_weights_d, weights_N * sizeof(float));
    cudaMalloc((void **)delta_biases_d, biases_N * sizeof(float));

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
    training_buffer(train_S, layers, W_sizes, pool_sizes, train_split);
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
    /*************************************************************************/

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

        if(!strcmp(layers.at(i).c_str(), "cnn"))
        {

        }
        if(!strcmp(layers.at(i).c_str(), "pool"))
        {

        }
    }

    /*************************************************************************/
    /* verify data.*/
    /*
    std::ofstream f_w;
    std::ofstream f_b;
    std::ofstream f_td_o;
    std::ofstream f_td_i;

    f_w.open("weights.txt");
    f_b.open("biases.txt");
    f_td_o.open("train_data_out.txt");
    f_td_i.open("train_data_in.txt");

    float *weights = new float[start_w[layers.size()]];
    float *biases = new float[start_b[layers.size()]];
    float *train_in = new float[start_ni[layers.size()]];
    float *train_out = new float[start_no[layers.size()]];

    cudaMemcpy(weights, weights_d[0], start_w[layers.size()] * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(biases, biases_d[0], start_b[layers.size()] * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(train_in, train_neuron_in_d[0], start_ni[layers.size()] * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(train_out, train_neuron_out_d[0], neuron_out_len * sizeof(float),
               cudaMemcpyDeviceToHost);

    for(unsigned int i = 0; i < start_w[layers.size()]; i++)
        f_w<<std::fixed<<std::setprecision(20)<<weights[i]<<std::endl;
    for(unsigned int i = 0; i < start_b[layers.size()]; i++)
        f_b<<std::fixed<<std::setprecision(20)<<biases[i]<<std::endl;
    for(unsigned int i = 0; i < start_ni[layers.size()]; i++)
        f_td_i<<std::fixed<<std::setprecision(20)<<train_in[i]<<std::endl;
    for(unsigned int i = 0; i < neuron_out_len; i++)
        f_td_o<<std::fixed<<std::setprecision(20)<<train_out[i]<<std::endl;

    f_w.close();
    f_b.close();
    f_td_o.close();
    f_td_i.close();
    */
    /**************************************************************************
     *  3. Release memory
     *************************************************************************/
    delete [] train_split;
    delete [] start_ni;
    delete [] start_no;
    delete [] start_w;
    delete [] start_b;
}


void propagate_backwards_gpu_train(float *data_gt, unsigned *data_S,
                                   float **train_neuron_inputs_d,
                                   float **train_neuron_outputs_d,
                                   std::vector<std::string> layers,
                                   float **weights_d,
                                   float **delta_weights_d,
                                   unsigned **W_sizes,
                                   float **biases_d,
                                   float **delta_biases_d,
                                   unsigned **b_sizes,
                                   unsigned **pool_sizes)
{

    /**************************************************************************
     * 1. Allocate memory for weights and biases on GPU and initialize
     *  them with zeros.
     *************************************************************************/
    /*************************************************************************/
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
    /*************************************************************************/
    /**************************************************************************
     * 2. Transfer ground truth to gpu
     *************************************************************************/
    float *data_gt_d;
    cudaMalloc((void **)&data_gt_d,
               data_S[3] * W_sizes[layers.size() - 1][1] * sizeof(float));
    cudaMemcpy(data_gt_d, data_gt,
               data_S[3] * W_sizes[layers.size() - 1][1] * sizeof(float),
               cudaMemcpyHostToDevice);

    /*************************************************************************/
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
    /*************************************************************************/
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
    std::cout<<"avg error:"<<*avg_error<<std::endl;
    /**************************************************************************
     * 5. Update weights and biases
     *************************************************************************/
    weights_N = 0;
    biases_N = 0;
    outputs_N = 0;
    delta_N = 0;
    float lr = 0.01;
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

            std::cout<<std::endl;


            update_weights_and_biases<<<blc_per_grid, th_per_block>>>
                    (weights_d[0], biases_d[0], weights_N, biases_N,
                     W_sizes[i][0], W_sizes[i][1],
                     train_neuron_outputs_d[0], outputs_N,
                     delta_x_d, delta_N,
                     data_S[3], lr);

            weights_N += (W_sizes[i][0] *  W_sizes[i][1]);
            biases_N += W_sizes[i][1];
            outputs_N += W_sizes[i][0] * data_S[3];
            delta_N += W_sizes[i][1] * data_S[3];

        }
    }
    /*************************************************************************/

    /*
    float *delta_x = new float[N_tot];
    for(unsigned int i = 0; i < N_tot; i++)
        delta_x[i] = 0.0;

    cudaMemcpy(delta_x, delta_x_d, N_tot * sizeof(float),
               cudaMemcpyDeviceToHost);
    std::ofstream f_delta;
    f_delta.open("delta.txt");
    for(unsigned int i = 0; i < N_tot; i++)
        f_delta<<std::fixed<<std::setprecision(20)<<delta_x[i]<<std::endl;
    f_delta.close();

    std::ofstream f_gt;
    f_gt.open("ground_truth.txt");
    for(unsigned int i = 0; i < data_S[3]; i++)
        f_gt<<data_gt[i]<<std::endl;
    f_gt.close();
    delete [] delta_x;

    float *weight_updates = new float[weights_N];
    cudaMemcpy(weight_updates, weights_d[0], weights_N * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::ofstream f_wu;
    f_wu.open("weights_update.txt");
    for(unsigned int i = 0; i < weights_N; i++)
        f_wu<<weight_updates[i]<<std::endl;
    f_wu.close();
    delete [] weight_updates;
    */
    cudaFree(data_gt_d);
    cudaFree(delta_x_d);
    cudaFree(train_neuron_inputs_d[0]);
    cudaFree(train_neuron_outputs_d[0]);
    cudaFree(train_neuron_inputs_d);
    cudaFree(train_neuron_outputs_d);
}


