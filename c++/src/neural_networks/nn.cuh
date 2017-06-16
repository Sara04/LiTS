/*
 * nn.cuh
 *
 *  Created on: Apr 18, 2017
 *      Author: sara
 */

#ifndef NN_CUH_
#define NN_CUH_

#include <iostream>
#include <vector>

/******************************************************************************
 * Methods used for nn model training and testing
 *****************************************************************************/

void data_buffer_split(unsigned *data_S, std::vector<std::string> layers,
                       unsigned **W_sizes, unsigned **pool_sizes,
                       unsigned *data_split);


void transfer_trainable_parameters_to_gpu_(std::vector<std::string> layers,
                                           float **weights, unsigned **W_sizes,
                                           float **biases, unsigned **b_sizes,
                                           float **weights_d, float **biases_d,
                                           float **delta_weights_d,
                                           float **delta_biases_d);

void transfer_trainable_parameters_to_cpu_(std::vector<std::string> layers,
                                           float **weights, unsigned **W_sizes,
                                           float **biases, unsigned **b_sizes,
                                           float **weights_d, float **biases_d);

void propagate_forward_gpu_train(float *train_imgs, unsigned *train_s,
                                 float **train_neuron_inputs_d,
                                 float **train_neuron_outputs_d,
                                 std::vector<std::string> layers,
                                 float **weights_d,
                                 unsigned **W_sizes,
                                 float **biases_d,
                                 unsigned **b_sizes,
                                 unsigned **pool_sizes,
                                 unsigned na);

void propagate_forward_gpu_test(float *test_imgs, unsigned *test_S,
                                float **test_neuron_out_d,
                                std::vector<std::string> layers,
                                float **weights_d, unsigned **W_sizes,
                                float **biases_d, unsigned **b_sizes,
                                unsigned **pool_sizes,
                                float *scores,
                                unsigned na);

float compute_error_gpu(float *data_gt,
                        unsigned *data_S,
                        float **train_neuron_inputs_d,
                        float **train_neuron_outputs_d,
                        std::vector<std::string> layers,
                        float **weights_d,
                        float **delta_weights_d,
                        unsigned **W_sizes,
                        float **biases_d,
                        float **delta_biases_d,
                        unsigned **b_sizes,
                        unsigned **pool_sizes);

float propagate_backwards_gpu_train(float *data_gt,
                                    unsigned *data_S,
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
                                    float learning_rate,
                                    unsigned na);

#endif /* NN_CUH_ */
