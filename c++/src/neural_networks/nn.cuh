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


/*Method used to initial and transfer trainable parameters of the network
 * to the gpu.
 */
void transfer_trainable_parameters_to_gpu(std::vector<std::string> layers,
                                          float **weights, unsigned **W_sizes,
                                          float **biases, unsigned **b_sizes,
                                          float **weights_d, float **biases_d,
                                          float **delta_weights_d,
                                          float **delta_biases_d);

void training_buffer(unsigned *train_S, std::vector<std::string> layers,
                     unsigned **W_sizes, unsigned **pool_sizes,
                     unsigned *train_split);

void transfer_trainable_parameters_to_gpu(std::vector<std::string> layers,
                                          float **weights, unsigned **W_sizes,
                                          float **biases, unsigned **b_sizes);

void propagate_forward_gpu_train(float *train_imgs, unsigned *train_s,
                                 float **train_neuron_inputs_d,
                                 float **train_neuron_outputs_d,
                                 std::vector<std::string> layers,
                                 float **weights_d,
                                 unsigned **W_sizes,
                                 float **biases_d,
                                 unsigned **b_sizes,
                                 unsigned **pool_sizes);

void propagate_backwards_gpu_train(float *data_gt,
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

#endif /* NN_CUH_ */
