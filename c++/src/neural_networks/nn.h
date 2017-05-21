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
 *      weights: array of pointers to the layer weights on the cpu
 *      weights_d: pointer to the weights to the gpu
 *      delta_weights_d: pointer to the weight updates on the gpu
 *      W_sizes: pointer to the kernels' sizes
 *
 *      biases: array of the pointers to the layer biases on the cpu
 *      biases_d: pointer to the biases on the gpu
 *      delta_biases_d: pointer to the update of biases on the gpu
 *      b_sizes: pointer to the biases' sizes
 *
 *      pool_size: array of the pointers to the pool strides' sizes
 *
 *      training_data_d: pointer to the training data on the gpu
 *
 *
 * Methods:
 *
 *      NN: constructors
 *      ~NN: destructor
 *      operator= : overload of the assignement operator
 *
 *      get_layers: get vector of existing layers
 *      get_weights: get pointer to the array of pointers to the layer weights
 *      get_weight_sizes: get pointer to the array of pointer to the kernels'
 *          sizes
 *      get_biases: get pointer to the array of pointers to the layer biases
 *      get_bias_sizes: get pointer to the array of pointers to the biases'
 *          sizes
 *      get_pool_sizes: get pointer to the array of pointers to the strides'
 *          sizes
 *
 *      init_weights: initial kernel weights
 *      init_biases: initial biases
 *
 *      transfer_trainable_parameters: transfer trainable parameters to the
 *          gpu (weights and biases)
 *
 *      propagate_forward: propagate data trough the network forward
 *      propagate_backwards: propagate error trough the network backwards
 *
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
    ~NN();

    NN operator=(NN &nn);

    std::vector<std::string> get_layers();
    float ** get_weights();
    unsigned ** get_weight_sizes();
    float ** get_biases();
    unsigned ** get_bias_sizes();
    unsigned ** get_pool_sizes();

    void init_weights(float *W, unsigned *S);
    void init_biases(float *b, unsigned *S);

    void transfer_trainable_parameters();

    void propagate_forward_train(float *data, unsigned *data_S);
    void propagate_backwards_train(float *data_gt, unsigned *data_S);
};



#endif /* NN_H_ */
