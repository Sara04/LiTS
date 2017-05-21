/*
 * nn.cpp
 *
 *  Created on: Apr 18, 2017
 *      Author: sara
 */

#include "nn.cuh"
#include "nn.h"
#include <cstring>

NN::NN(){}

NN::NN(std::vector<std::string> layers_,
       unsigned **W_sizes_, unsigned **b_sizes_)
{

    unsigned int N_w = 0;
    for(unsigned int i = 0; i < layers_.size(); i++)
    {
        std::string layer_name(layers_.at(i));
        layers.push_back(layer_name);

        if(!strcmp(layer_name.c_str(), "fcn") or
           !strcmp(layer_name.c_str(), "conv"))
            N_w += 1;

    }

    weights = new float*[N_w];
    W_sizes = new unsigned*[N_w];
    biases = new float*[N_w];
    b_sizes = new unsigned*[N_w];
    for(unsigned int i = 0; i < N_w; i++)
    {
        W_sizes[i] = new unsigned[4];
        b_sizes[i] = new unsigned[1];

        W_sizes[i] = W_sizes_[i];
        b_sizes[i] = b_sizes_[i];

        weights[i] = new float[W_sizes[i][0] * W_sizes[i][1] *
                               W_sizes[i][2] * W_sizes[i][3]];

        biases[i] = new float[b_sizes[i][0]];

        init_weights(weights[i], W_sizes[i]);
        init_biases(biases[i], b_sizes[i]);
    }
}

NN::~NN(){}

NN NN::operator=(NN &nn)
{
    layers = nn.get_layers();
    weights = nn.get_weights();
    W_sizes = nn.get_weight_sizes();
    biases = nn.get_biases();
    b_sizes = nn.get_bias_sizes();
    pool_sizes = nn.get_pool_sizes();
    return *this;
}

std::vector<std::string> NN::get_layers()
{
    return layers;
}
float ** NN::get_weights()
{
    return weights;
}
unsigned ** NN::get_weight_sizes()
{
    return W_sizes;
}
float ** NN::get_biases()
{
    return biases;
}
unsigned ** NN::get_bias_sizes()
{
    return b_sizes;
}
unsigned ** NN::get_pool_sizes()
{
    return pool_sizes;
}

void NN::init_weights(float *W, unsigned *S)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0,0.1);
    for(unsigned int i = 0; i < S[0] * S[1] * S[2] * S[3]; i++)
        W[i] = distribution(generator);
}

void NN::init_biases(float *b, unsigned *S)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0,0.1);
    for(unsigned int i = 0; i < S[0]; i++)
        b[i] = distribution(generator);
}

void NN::transfer_trainable_parameters()
{
    transfer_trainable_parameters_to_gpu(layers,
                                         weights, W_sizes,
                                         biases, b_sizes,
                                         &weights_d, &biases_d,
                                         &delta_weights_d,
                                         &delta_biases_d);

}


void NN::propagate_forward_train(float *data, unsigned *data_S)
{
    propagate_forward_gpu_train(data, data_S,
                                &neuron_inputs_d, &neuron_outputs_d,
                                layers,
                                &weights_d, W_sizes,
                                &biases_d, b_sizes,
                                pool_sizes);

    delete []data;
}


void NN::propagate_backwards_train(float *data_gt, unsigned *data_S)
{

    propagate_backwards_gpu_train(data_gt, data_S,
                                  &neuron_inputs_d, &neuron_outputs_d,
                                  layers,
                                  &weights_d, &delta_weights_d, W_sizes,
                                  &biases_d, &delta_biases_d, b_sizes,
                                  pool_sizes);
    delete [] data_gt;
}
