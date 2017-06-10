/*
 * activation_functions.cuh
 *
 *  Created on: Jun 10, 2017
 *      Author: sara
 */

#ifndef ACTIVATION_FUNCTIONS_CUH_
#define ACTIVATION_FUNCTIONS_CUH_

__device__
float _ReLU_neuron_activation(float neuron_input)
{
    if(neuron_input > 0)
        return neuron_input;
    else
        return 0;
}

__device__
float _ReLU_neuron_activation_derivative(float neuron_input)
{
    if(neuron_input > 0)
        return 1;
    else
        return 0;
}

__device__
float _logistic_neuron_activation(float neuron_input)
{
	return 1 / (1 + exp(-neuron_input));
}

__device__
float _logistic_neuron_activation_derivative(float neuron_input)
{
	float na = _logistic_neuron_activation(neuron_input);

	return na * (1 - na);
}

__device__
float _tanh_neuron_activation(float neuron_input)
{
	return 2 / (1 + exp(-2 * neuron_input)) -1;
}

__device__
float _tanh_neuron_activation_derivative(float neuron_input)
{
	float na = _tanh_neuron_activation(neuron_input);

	return (1 - pow(na, 2));
}

__device__
float _gaussian_neuron_activation(float neuron_input)
{
	return exp(-pow(neuron_input, 2));
}

__device__
float _gaussian_neuron_activation_derivative(float neuron_input)
{
	float na = _gaussian_neuron_activation(neuron_input);

	return (-2 * neuron_input * na);
}

__device__
float neuron_activation(float neuron_input, unsigned t=2)
{
	if(t == 1)
		return _ReLU_neuron_activation(neuron_input);
	else if(t == 2)
		return _logistic_neuron_activation(neuron_input);
	else if(t == 3)
		return _tanh_neuron_activation(neuron_input);
	else if(t == 4)
		return _gaussian_neuron_activation(neuron_input);
	else
		return 0;
}

__device__
float neuron_activation_derivative(float neuron_input, unsigned t=2)
{
	if(t == 1)
		return _ReLU_neuron_activation_derivative(neuron_input);
	else if(t == 2)
		return _logistic_neuron_activation_derivative(neuron_input);
	else if(t == 3)
		return _tanh_neuron_activation_derivative(neuron_input);
	else if(t == 4)
		return _gaussian_neuron_activation_derivative(neuron_input);
	else
		return 0;
}

#endif /* ACTIVATION_FUNCTIONS_CUH_ */
