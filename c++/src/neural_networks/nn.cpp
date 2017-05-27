/*
 * nn.cpp
 *
 *  Created on: Apr 18, 2017
 *      Author: sara
 */

#include "nn.cuh"
#include "nn.h"
#include <cstring>

/******************************************************************************
 * empty constructor
 *****************************************************************************/
NN::NN(){}

/******************************************************************************
 * constructor: configuring network and random initialization of kernels'
 * weight and biases
 *
 * Arguments:
 *      layers: vectors of layers' names
 *      W_sizes_: pointer to the kernels' sizes
 *      b_sizes_: pointer to the biases' sizes (lengths)
 *****************************************************************************/
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

/******************************************************************************
 * constructor: loading existing network, loading configuration from config.txt
 * file and kernels' weights and biases from weights.bin and biases.bin binary
 * files
 *
 * Arguments:
 *      model_path: path where pre-trained model and its configuration are
 *      saved
 *****************************************************************************/
NN::NN(std::string model_path)
{
    std::string config_path = model_path + "config.txt";
    if(!fs::exists(config_path))
    {
        std::cout<<"Configuration file does not exist!"<<std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        std::ifstream config_file;
        config_file.open(config_path);
        std::string line;
        std::vector<std::string> weights_str;
        std::vector<std::string> biases_str;
        unsigned N_w = 0;
        while(std::getline(config_file, line))
            if(!strcmp(line.c_str(), "fcn") or !strcmp(line.c_str(), "conv"))
            {
                layers.push_back(line);

                std::string w_line;
                std::getline(config_file, w_line);
                std::string b_line;
                std::getline(config_file, b_line);

                weights_str.push_back(w_line);
                biases_str.push_back(b_line);

                N_w += 1;
            }

        weights = new float*[N_w];
        W_sizes = new unsigned*[N_w];
        biases = new float*[N_w];
        b_sizes = new unsigned*[N_w];
        for(unsigned i = 0; i < N_w; i++)
        {
            W_sizes[i] = new unsigned[4];
            b_sizes[i] = new unsigned[1];

            std::stringstream w_stream(weights_str.at(i));
            std::stringstream b_stream(biases_str.at(i));

            for(unsigned j = 0; j < 4; j++)
                w_stream >> W_sizes[i][j];
            b_stream >> b_sizes[i][0];
        }
    }

    std::string weights_path = model_path + "weights.bin";
    if(!fs::exists(weights_path))
    {
        std::cout<<"Weights binary file does not exist!"<<std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        std::ifstream w_in(weights_path, std::ios::in |std::ios::binary);
        for(unsigned i = 0; i < layers.size(); i++)
        {
            unsigned s = 1;
            for(unsigned j = 0; j < 4; j++)
                s *= W_sizes[i][j];
            weights[i] = new float[s];
            w_in.read((char *)weights[i], sizeof(float) * s);
        }
    }

    std::string biases_path = model_path + "biases.bin";
    if(!fs::exists(biases_path))
    {
        std::cout<<"Biases binary file does not exist!"<<std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        std::ifstream b_in(biases_path, std::ios::in |std::ios::binary);
        for(unsigned i = 0; i < layers.size(); i++)
        {
            biases[i] = new float[b_sizes[i][0]];
            b_in.read((char *)biases[i], sizeof(float) * b_sizes[i][0]);
        }
    }
}

/******************************************************************************
 * destructor: to be done
 *****************************************************************************/
NN::~NN(){}

/******************************************************************************
 * operator= : assignment operator
 *
 * Argument:
 *      &nn: reference to the NN object to be assigned
 *****************************************************************************/
NN NN::operator=(const NN &nn)
{
    layers = nn.get_layers();
    weights = nn.get_weights();
    W_sizes = nn.get_weight_sizes();
    biases = nn.get_biases();
    b_sizes = nn.get_bias_sizes();
    pool_sizes = nn.get_pool_sizes();
    return *this;
}

/******************************************************************************
 * get_layers: get vector of existing layers of the network
 *****************************************************************************/
std::vector<std::string> NN::get_layers() const
{
    return layers;
}

/******************************************************************************
 * get_weights: get pointer to the kernels' weights
 *****************************************************************************/
float ** NN::get_weights() const
{
    return weights;
}

/******************************************************************************
 * get_weight_sizes: get pointer to the kernels' sizes
 *****************************************************************************/
unsigned ** NN::get_weight_sizes() const
{
    return W_sizes;
}

/******************************************************************************
 * get_biases: get pointer to the biases
 *****************************************************************************/
float ** NN::get_biases() const
{
    return biases;
}

/******************************************************************************
 * get_biases_sizes: get pointer to the biases' sizes (lengths)
 *****************************************************************************/
unsigned ** NN::get_bias_sizes() const
{
    return b_sizes;
}

/******************************************************************************
 * get_pool_sizes: get pointer to the pool strides' sizes
 *****************************************************************************/
unsigned ** NN::get_pool_sizes() const
{
    return pool_sizes;
}

/******************************************************************************
 * init_weights: initialize weights of given size in random manner
 *
 * Arguments:
 *      W: pointer to the kernel's weights
 *      S: pointer to the kernel's size
 *****************************************************************************/
void NN::init_weights(float *W, unsigned *S)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0,0.1);
    for(unsigned int i = 0; i < S[0] * S[1] * S[2] * S[3]; i++)
        W[i] = distribution(generator);
}

/******************************************************************************
 * init_biases: initialize biases of given size in random manner
 *
 * Arguments:
 *      b: pointer to the bias array
 *      S: pointer to the bias array size (length)
 *****************************************************************************/
void NN::init_biases(float *b, unsigned *S)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0,0.1);
    for(unsigned int i = 0; i < S[0]; i++)
        b[i] = distribution(generator);
}

/******************************************************************************
 * transfer_trainable_parameters_to_gpu: transfer all kernels' weights and
 * biases from the cpu to the gpu, where they will stay during the training
 *****************************************************************************/
void NN::transfer_trainable_parameters_to_gpu()
{
    transfer_trainable_parameters_to_gpu_(layers,
                                          weights, W_sizes,
                                          biases, b_sizes,
                                          &weights_d, &biases_d,
                                          &delta_weights_d,
                                          &delta_biases_d);

}

/******************************************************************************
 * transfer_trainable_parameters_to_cpu: transfer all kernels' weights and
 * biases from gpu to the cpu, normally after the training
 *****************************************************************************/
void NN::transfer_trainable_parameters_to_cpu()
{
    transfer_trainable_parameters_to_cpu_(layers,
                                          weights, W_sizes,
                                          biases, b_sizes,
                                          &weights_d, &biases_d);
}

/******************************************************************************
 * propagate_forward_train: call gpu based function to propagate training data
 * trough the network and stores neuron inputs and outputs that are needed
 * for the further training
 *
 * Arguments:
 *      data: input training data
 *      data_S: input training data size
 *****************************************************************************/
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

/******************************************************************************
 * propagate_forward_test: call gpu based function to propagate testing data
 * trough the network and stores only neuron outputs, since they are needed
 * for final scores computation
 *
 * Arguments:
 *      data: input testing data
 *      data_S: input testing data size
 *      scores: array where to store network output
 *****************************************************************************/
void NN::propagate_forward_test(float *data, unsigned *data_S, float *scores)
{
    propagate_forward_gpu_test(data, data_S,
                               &neuron_outputs_d,
                               layers,
                               &weights_d, W_sizes,
                               &biases_d, b_sizes,
                               pool_sizes,
                               scores);
    delete []data;
}

/******************************************************************************
 * compute_error: compute model's performance given the ground truth for the
 * current data that is on gpu
 *
 * Arguments:
 *      data_gt: ground truth
 *      data_S: data size
 *****************************************************************************/
float NN::compute_error(float *data_gt, unsigned *data_S)
{
    float error;
    error = compute_error_gpu(data_gt, data_S,
                              &neuron_inputs_d,
                              &neuron_outputs_d,
                              layers,
                              &weights_d,
                              &delta_weights_d,
                              W_sizes,
                              &biases_d,
                              &delta_biases_d,
                              b_sizes,
                              pool_sizes);
    return error;
}

/******************************************************************************
 * propagate_backwards_train: propagate backwards trough the network computed
 * error
 *
 * Arguments:
 *      data_gt: ground truth
 *      data_S: data size
 *      learning_rate: rate by which trainable parameters are updated
 *****************************************************************************/
float NN::propagate_backwards_train(float *data_gt, unsigned *data_S,
                                    float learning_rate)
{

    float train_error;
    train_error = propagate_backwards_gpu_train(data_gt, data_S,
                                                &neuron_inputs_d,
                                                &neuron_outputs_d,
                                                layers,
                                                &weights_d,
                                                &delta_weights_d,
                                                W_sizes,
                                                &biases_d,
                                                &delta_biases_d,
                                                b_sizes,
                                                pool_sizes,
                                                learning_rate);
    delete [] data_gt;
    return train_error;
}

/******************************************************************************
 * save_model: save trainable parameters - kernels' weights and biases
 *
 * Arguments:
 *      model_path: path where to store network configuration, weights and
 *      biases
 *****************************************************************************/
void NN::save_model(std::string model_path)
{
    std::string config_path = model_path + "config.txt";
    if(!fs::exists(config_path))
    {
        std::ofstream config_file;
        config_file.open(config_path);
        for(unsigned i = 0; i < layers.size(); i++)
        {
            config_file<<layers.at(i)<<std::endl;
            for(unsigned j = 0; j < 4; j++)
                config_file<<W_sizes[i][j]<<" ";
            config_file<<std::endl;
            config_file<<b_sizes[i][0]<<std::endl;
        }
    }

    std::string weights_path = model_path + "weights.bin";
    if(!fs::exists(weights_path))
    {
        std::ofstream w_out(weights_path, std::ios::out |std::ios::binary);
        for(unsigned i = 0; i < layers.size(); i++)
        {
            unsigned s = 1;
            for(unsigned j = 0; j < 4; j++)
                s *= W_sizes[i][j];
            w_out.write((char *)weights[i], sizeof(float) * s);
        }
    }

    std::string biases_path = model_path + "biases.bin";
    if(!fs::exists(biases_path))
    {
        std::ofstream b_out(biases_path, std::ios::out |std::ios::binary);
        for(unsigned i = 0; i < layers.size(); i++)
            b_out.write((char *)biases[i], sizeof(float) * b_sizes[i][0]);
    }
}
