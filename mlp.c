#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

// TODO: wrap static variables in a "state", struct
size_t layer_count;
size_t layer_sizes[5];

float active_weights[100 * 1000];
float active_biases[10 * 1000];

size_t total_z_size;
size_t total_weight_size;
// each z is the weighted sum of the previous layer's activations, plus a bias
size_t weight_indexes[4];
size_t z_indexes[4];
float z_cache[10 * 1000];

float* get_z(size_t l, size_t j) {
	return &z_cache[z_indexes[l - 1] + j];
}

float *get_weight(size_t l, size_t j, size_t k) {
	return &active_weights[weight_indexes[l - 1] + j * layer_sizes[l - 1] + k];
}

float *get_bias(size_t l, size_t j) {
	return &active_biases[z_indexes[l - 1] + j];
}

// math
float dot(float *v1, float *v2, size_t len) {
	float sum = 0.0f;
	for (int i = 0; i < len; ++i) {
		printf("%f + %f\n", v1[i], v2[i]);
		sum += v1[i] * v2[i];
	}
	return sum;
}

float af(float n) {
	return n / (1 + fabsf(n));
}

float af_derivative(float n) {
	float tmp = (1 + fabsf(n));
	return 1.0f / (tmp * tmp);
}

float cf(float a, float y) {
	return (a - y) * (a - y);
}

float cf_derivative(float a, float y) {
	return 2 * (a - y);
}

// mlp

void feedforward(float *input) {
	int l = 1;
	for (int j = 0; j < layer_sizes[l]; ++j) {
		printf("adding bias: %f\n", *get_bias(l, j));
		*get_z(l, j) = *get_bias(l, j) + dot(input, get_weight(l, j, 0), layer_sizes[l - 1]);
	}

	for (l = 2; l < layer_count; ++l) {
		float prev_activations[layer_sizes[l - 1]];
		for (int k = 0; k < layer_sizes[l - 1]; ++k) {
			prev_activations[k] = af(*get_z(l - 1, k));
		}
		for (int j = 0; j < layer_sizes[l]; ++j) {
			printf("adding bias: %f\n", *get_bias(l, j));
			*get_z(l, j) = *get_bias(l, j) + dot(prev_activations, get_weight(l, j, 0), layer_sizes[l - 1]);
		}
	}
}

void init(size_t lc, size_t *ls) {
	layer_count = lc;

	total_z_size = 0;
	total_weight_size = 0;
	for (int i = 0; i < lc; ++i) {
		layer_sizes[i] = ls[i];
		if (i > 0) {
			z_indexes[i - 1] = total_z_size;
			weight_indexes[i - 1] = total_weight_size;
			total_z_size += ls[i];
			total_weight_size += ls[i] * ls[i - 1];
		}
	}
}

void set_weights_and_biases(float *weights, float *biases) {
	for (int i = 0; i < total_weight_size; ++i) {
		active_weights[i] = weights[i];
	}

	for (int i = 0; i < total_z_size; ++i) {
		active_biases[i] = biases[i];
	}
}

// testmain
#define NUM_LAYERS 4
int main() {
	size_t ls[NUM_LAYERS] = { 3, 2, 2, 1 };
	init(NUM_LAYERS, ls);

	// set input
	float w[] = {
		0.36, 0.75, 0.74, 0.37, 0.74, 0.16,
		0.75, 0.87, 0.22, 0.83,
		0.14, 0.92
	};
	float b[] = { 0.11, 0.78, 0.85, 0.47, 0.61 };
	float in[] = { 0.71, 0.98, 0.50 };
	set_weights_and_biases(w, b);

	// activate network
	feedforward(in);

	// print results
	printf("output:\n");
	for (int i = 1; i < NUM_LAYERS; ++i) {
		printf("l%d = [", i);
		for (int j = 0; j < layer_sizes[i]; ++j) {
			printf("%f, ", *get_z(i, j));
		}
		printf("]\n");
	}
	return 0;
}