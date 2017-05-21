/*
 * nn_kernels.cuh
 *
 *  Created on: Apr 27, 2017
 *      Author: sara
 */

#ifndef NN_KERNELS_CUH_
#define NN_KERNELS_CUH_

#define BLOCK_SIZE 16

__device__
float neuron_activation(float neuron_input)
{
    //return 1.0 / (1.0 + exp(-1. * neuron_input));

    if(neuron_input > 0)
        return neuron_input;
    else
        return 0;
}

__device__
float neuron_activation_derivative(float neuron_input)
{
    //float na = neuron_activation(neuron_input);
    //return na * (1. - na);
    if(neuron_input > 0)
        return 1;
    else
        return 0;
}

__global__
void cost_function_backprop(float *data_gt,
                           float *train_neuron_inputs, unsigned N_inputs,
                           float *train_neuron_outputs, unsigned N_outputs,
                           float *biases_x, unsigned biases_x_N,
                           unsigned N_samples,
                           unsigned out_feat_len)
{
    unsigned sample_idx = blockIdx.x;
    unsigned feat_idx = threadIdx.x;

    if (sample_idx < N_samples and feat_idx < out_feat_len)
    {
        unsigned gt_idx = sample_idx * out_feat_len + feat_idx;
        unsigned biases_x_idx = biases_x_N -
                                N_samples * out_feat_len +
                                sample_idx * out_feat_len +
                                feat_idx;
        unsigned in_x_idx = N_inputs -
                            N_samples * out_feat_len +
                            sample_idx * out_feat_len +
                            feat_idx;
        unsigned out_x_idx = N_outputs -
                             N_samples * out_feat_len +
                             sample_idx * out_feat_len +
                             feat_idx;

        biases_x[biases_x_idx] =
                (train_neuron_outputs[out_x_idx] - data_gt[gt_idx]) *
                neuron_activation_derivative(train_neuron_inputs[in_x_idx]);
    }
}


__global__
void evaluate_error(float *data_gt,
                    unsigned N_samples,
                    unsigned out_feat_len,
                    float *train_neuron_outputs,
                    unsigned N_outputs,
                    float *avg_error)
{
    unsigned idx_s = blockIdx.x;
    unsigned idx_f = threadIdx.x;

    if(idx_s == 0 and idx_f == 0)
    {
        float label;
        for(unsigned i = 0; i < N_samples; i++)
        {
            unsigned idx = N_outputs - N_samples * out_feat_len + i;

            if(train_neuron_outputs[idx] >= 0.5)
                label = 1.0;
            else
                label = 0.0;
            if(data_gt[i] == label)
                *avg_error += 1;

        }

        *avg_error /= N_samples;
    }
}
__global__
void propagate_forward_fcn_gpu_train(float *train_neuron_inputs,
                                     float *train_neuron_outputs,
                                     float *weights_d,
                                     float *biases_d,
                                     unsigned N_feats,
                                     unsigned N_input,
                                     unsigned N_output,
                                     long start_ni,
                                     long start_no,
                                     long start_w,
                                     long start_b)
{
    unsigned rt = threadIdx.y;
    unsigned ct = threadIdx.x;
    unsigned rb = blockIdx.y;
    unsigned cb = blockIdx.x;

    float v = 0.0;

    unsigned Nb = N_input / BLOCK_SIZE;
    if(N_input > Nb * BLOCK_SIZE)
        Nb += 1;

    for(unsigned blc_c=0; blc_c < Nb; blc_c++)
    {
        __shared__ float TD[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float TW[BLOCK_SIZE][BLOCK_SIZE];

        long d_c_idx = BLOCK_SIZE * blc_c + ct;
        long w_c_idx = BLOCK_SIZE * cb + ct;

        if(d_c_idx < N_input)
            TD[rt][ct] = train_neuron_outputs[start_no +
                                              N_input * BLOCK_SIZE * rb +
                                              N_input * rt +
                                              BLOCK_SIZE * blc_c + ct];
        else
            TD[rt][ct] = 0.0;

        if(w_c_idx < N_output and d_c_idx < N_input)
            TW[rt][ct] = weights_d[start_w +
                                   N_output * BLOCK_SIZE * blc_c +
                                   N_output * rt +
                                   BLOCK_SIZE * cb + ct];
        else
            TW[rt][ct] = 0.0;

        __syncthreads();

        unsigned b = BLOCK_SIZE;
        if((blc_c + 1) * BLOCK_SIZE > N_input)
            b = N_input - blc_c * BLOCK_SIZE;

        for(unsigned i = 0; i < b; i++)
            v += TD[rt][i] * TW[i][ct];

        __syncthreads();
    }
    if(BLOCK_SIZE * cb + ct < N_output)
    {
        unsigned n_out_idx = start_no +
                             N_feats * N_input +
                             rb * BLOCK_SIZE * N_output + rt * N_output +
                             cb * BLOCK_SIZE + ct;
        unsigned n_in_idx = start_ni +
                            rb * BLOCK_SIZE * N_output + rt * N_output +
                            cb * BLOCK_SIZE + ct;

        train_neuron_inputs[n_in_idx] =
                    v + biases_d[start_b + cb * BLOCK_SIZE + ct];

        train_neuron_outputs[n_out_idx] =
                neuron_activation(train_neuron_inputs[n_in_idx]);
    }
}

__global__
void backpropagate_fcn_gpu_train(float *weights_d, long start_w,
                                 float *delta_x_d, long start_dx,
                                 float *train_neuron_inputs_d,
                                 long start_in,
                                 unsigned N_feats,
                                 unsigned in_len,
                                 unsigned out_len)
{
    unsigned rt = threadIdx.y;
    unsigned ct = threadIdx.x;
    unsigned rb = blockIdx.y;
    unsigned cb = blockIdx.x;

    float v = 0.0;

    unsigned Nb = out_len / BLOCK_SIZE;
    if(out_len > Nb * BLOCK_SIZE)
        Nb += 1;

    for(unsigned blc_c=0; blc_c < Nb; blc_c++)
    {
        __shared__ float delta[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float WT[BLOCK_SIZE][BLOCK_SIZE];

        long d_c_idx = BLOCK_SIZE * blc_c + ct;
        long w_r_idx = BLOCK_SIZE * blc_c + rt;

        if(d_c_idx < out_len)
            delta[rt][ct] = delta_x_d[start_dx +
                                      out_len * BLOCK_SIZE * rb +
                                      out_len * rt +
                                      BLOCK_SIZE * blc_c + ct];
        else
            delta[rt][ct] = 0.0;

        if(w_r_idx < out_len)
            WT[rt][ct] = weights_d[start_w +
                                   out_len * BLOCK_SIZE * cb +
                                   out_len * ct +
                                   BLOCK_SIZE * blc_c + rt];
        else
            WT[rt][ct] = 0.0;

        __syncthreads();

        unsigned b = BLOCK_SIZE;
        if((blc_c + 1) * BLOCK_SIZE > out_len)
            b = out_len - blc_c * BLOCK_SIZE;

        for(unsigned i = 0; i < b; i++)
            v += delta[rt][i] * WT[i][ct];

        __syncthreads();
    }
    if((BLOCK_SIZE * rb + rt) < N_feats and (BLOCK_SIZE * cb + ct) < in_len)
    {
        unsigned idx = (rb * BLOCK_SIZE + rt) * in_len + cb * BLOCK_SIZE + ct;
        delta_x_d[start_dx - N_feats * in_len + idx] = v *
                neuron_activation_derivative(train_neuron_inputs_d[start_in - N_feats * in_len + idx]);
    }
}

__global__
void update_weights_and_biases(float *weights_d,
                               float *biases_d,
                               unsigned start_w,
                               unsigned start_b,
                               unsigned in_feat_len,
                               unsigned out_feat_len,
                               float *train_neuron_outputs,
                               unsigned start_in,
                               float *delta_d,
                               unsigned start_d,
                               unsigned N_samples,
                               float lr)
{
    unsigned rt = threadIdx.y;
    unsigned ct = threadIdx.x;
    unsigned rb = blockIdx.y;
    unsigned cb = blockIdx.x;
    if((BLOCK_SIZE * rb + rt) < in_feat_len and (BLOCK_SIZE * cb + ct) < out_feat_len)
    {
        float dw = 0.0;
        float db = 0.0;

        unsigned Nb = N_samples / BLOCK_SIZE;
        if(N_samples > Nb * BLOCK_SIZE)
            Nb += 1;

        for(unsigned blc_c=0; blc_c < Nb; blc_c++)
        {
            __shared__ float neuron_out[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float deltas[BLOCK_SIZE][BLOCK_SIZE];

            long d_c_idx = BLOCK_SIZE * cb + ct;
            long no_r_idx = BLOCK_SIZE * rb + rt;
            long no_c_idx = BLOCK_SIZE * blc_c + ct;

            if(no_r_idx < in_feat_len and no_c_idx < N_samples)
                neuron_out[rt][ct] = train_neuron_outputs[start_in +
                                                          in_feat_len * BLOCK_SIZE * blc_c +
                                                          in_feat_len * ct +
                                                          BLOCK_SIZE * rb + rt];
            else
                neuron_out[rt][ct] = 0.0;

            if(d_c_idx < out_feat_len)
                deltas[rt][ct] = delta_d[start_d +
                                         out_feat_len * BLOCK_SIZE * blc_c +
                                         out_feat_len * rt +
                                         BLOCK_SIZE * cb + ct];
            else
                deltas[rt][ct] = 0.0;

            __syncthreads();

            unsigned b = BLOCK_SIZE;
            if((blc_c + 1) * BLOCK_SIZE > N_samples)
                b = N_samples - blc_c * BLOCK_SIZE;

            for(unsigned i = 0; i < b; i++)
            {
                dw += neuron_out[rt][i] * deltas[i][ct];
                db += deltas[i][ct];
            }

            __syncthreads();
        }
        unsigned idx_w = (rb * BLOCK_SIZE + rt) * out_feat_len + cb * BLOCK_SIZE + ct;
        unsigned idx_b = cb * BLOCK_SIZE + ct;
        atomicAdd(&weights_d[start_w + idx_w], - lr * dw / N_samples);
        atomicAdd(&biases_d[start_b + idx_b], - lr * db / N_samples);
    }
}
#endif /* NN_KERNELS_CUH_ */
