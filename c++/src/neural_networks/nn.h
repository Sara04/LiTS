/*
 * nn.cuh
 *
 *  Created on: Apr 18, 2017
 *      Author: sara
 */

#ifndef NN_H_
#define NN_H_

#include <random>
#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <fstream>
#include <sstream>

namespace fs = boost::filesystem;

/******************************************************************************
/* NN a  class for the nn model creation, training and testing
 *
 * It has means of model structure creation, model weights initialization,
 * model transfer to gpu, model weights update and model testing
 *
 * Attributes:
 *
 *      layers: vector of neural network layers
 *          possibilities: fcn, cnn, pool
 *
 *      weights: array of pointers to the kernels' weights on the cpu
 *      weights_d: pointer to the pointer to the kernels' weights on the gpu
 *      delta_weights_d: pointer to the pointer to the kernels' updates of
 *          weights on the gpu
 *      W_sizes: pointer to the kernels' sizes
 *
 *      biases: array of the pointers to the biases on the cpu
 *      biases_d: pointer to the pointer to the biases on the gpu
 *      delta_biases_d: pointer to the pointer to the updates of the biases
 *          on the gpu
 *      b_sizes: pointer to the biases' sizes (lengths)
 *
 *      pool_size: array of the pointers to the pool strides' sizes
 *
 *      neuron_inputs_d: pointer to the neuron inputs on gpu
 *      neuron_outputs_d: pointer to the neuron outputs on gpu
 *
 *
 * Methods:
 *
 *      NN: constructors (empty, with random initialization, by loading)
 *      ~NN: destructor
 *
 *      operator= : overload of the assignment operator
 *
 *      get_layers: get vector of existing layers
 *      get_weights: get pointers to the kernels' weights
 *      get_weight_sizes: get pointer to the kernels' sizes
 *      get_biases: get pointer to the biases
 *      get_bias_sizes: get pointer to the biases' sizes (lengths)
 *      get_pool_sizes: get pointer to the pool strides' sizes
 *
 *      init_weights: initial kernels' weights
 *      init_biases: initial biases
 *
 *      transfer_trainable_parameters_to_gpu: transfer trainable parameters
 *          from cpu to gpu
 *      transfer_trainable_parameters_to_cpu: transfer trainable parameters
 *          from gpu to cpu
 *
 *      propagate_forward_train: propagate data trough the network forward
 *          for the training process
 *      propagate_forward_test: propagate data trough the network forward
 *          for the testing process
 *      compute_error: compute the classification/regression error given
 *          the ground truths
 *      propagate_backwards_train: propagate errors trough the network
 *          backwards for the training process
 *
 *      save_model: save network configuration, kernels' weights and biases
 *
 *****************************************************************************/

class NN
{
private:

    std::vector<std::string> layers;

    float **weights;
    float *weights_d;
    float *delta_weights_d;
    unsigned **W_sizes;

    float **biases;
    float *biases_d;
    float *delta_biases_d;
    unsigned **b_sizes;

    unsigned **pool_sizes;

    float *neuron_inputs_d;
    float *neuron_outputs_d;

public:

    NN();
    NN(std::vector<std::string> layers_,
       unsigned **W_sizes_, unsigned **b_sizes_);
    NN(std::string model_path);
    ~NN();

    NN operator=(const NN &nn);

    std::vector<std::string> get_layers()const;
    float ** get_weights()const;
    unsigned ** get_weight_sizes()const;
    float ** get_biases()const;
    unsigned ** get_bias_sizes()const;
    unsigned ** get_pool_sizes()const;

    void init_weights(float *W, unsigned *S);
    void init_biases(float *b, unsigned *S);

    void transfer_trainable_parameters_to_gpu();
    void transfer_trainable_parameters_to_cpu();

    void propagate_forward_train(float *data, unsigned *data_S);
    void propagate_forward_test(float *data, unsigned *data_S, float *scores);
    float compute_error(float *data_gt, unsigned *data_S);
    float propagate_backwards_train(float *data_gt, unsigned *data_S,
                                    float learning_rate);

    void save_model(std::string model_path);
};

#endif /* NN_H_ */
