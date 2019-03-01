#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>


char asciimap[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.";

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

size_t get_weight_index(size_t l, size_t j, size_t k) {
	return weight_indexes[l - 1] + j * layer_sizes[l - 1] + k;
}

size_t get_z_index(size_t l, size_t j) {
	return z_indexes[l - 1] + j;
}

float *get_z(size_t l, size_t j) {
//	printf("you requested the z value at index %zu\n", get_z_index(l, j));
	return &z_cache[get_z_index(l, j)];
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
		sum += v1[i] * v2[i];
	}
	return sum;
}

float af(float n) {
	return (1 + (n / (1 +fabsf(n)))) / 2;
}

float af_derivative(float n) {
	float tmp = (1 + fabsf(n));
	return 1.0f / (2 * tmp * tmp);
}

float cf(float a, float y) {
	return (a - y) * (a - y);
}

float cf_derivative(float a, float y) {
	return 2 * (a - y);
}


void print_digit(float *input) {
	printf("\n");
	for (int i = 0; i < 28; ++i) {
		for(int j = 0; j < 28; ++j) {
			putchar(asciimap[(int)(input[i * 28 + j] * (sizeof(asciimap) - 2))]);
		}
		printf("\n");
	}
}
// mlp

void feedforward(float *input) {
	//print_digit(input);

	int l = 1;
	for (int j = 0; j < layer_sizes[l]; ++j) {
	//	printf("adding bias: %f\n", *get_bias(l, j));
		*get_z(l, j) = *get_bias(l, j) + dot(input, get_weight(l, j, 0), layer_sizes[l - 1]);
	}

	for (l = 2; l < layer_count; ++l) {
		float prev_activations[layer_sizes[l - 1]];
		for (int k = 0; k < layer_sizes[l - 1]; ++k) {
			prev_activations[k] = af(*get_z(l - 1, k));
		}
		for (int j = 0; j < layer_sizes[l]; ++j) {
		//	printf("adding bias: %f\n", *get_bias(l, j));
			*get_z(l, j) = *get_bias(l, j) + dot(prev_activations, get_weight(l, j, 0), layer_sizes[l - 1]);
		}
	}

	// for (int i = 0; i < layer_sizes[layer_count - 1]; ++i) {
	//  	printf("%f, ", af(*get_z(layer_count - 1, i)));
	// }
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

void update_gradients(float *in, float *y, float *weight_gradient, float *bias_gradient) {
	// caclulate error for each neuron
	// printf("^correct output layer for that image looks like... { ");
	// for (int i = 0; i < layer_sizes[layer_count -1]; ++i) {
	// 	printf("%.2f, ", y[i]);
	// }
	// 	printf("}\n");


	float error[total_z_size];

	int l = layer_count - 1;
	// output layer
	for (int j = 0; j < layer_sizes[l]; ++j) {
		float z = *get_z(l, j);
		float a = af(z);
//		printf("cf_D: %d => %f\n", j, cf_derivative(a, y[j]) * af_derivative(z));
		error[get_z_index(l, j)] = cf_derivative(a, y[j]) * af_derivative(z);
	}

	// other layers
	for (l -= 1; l > 0; --l) {
		for (int k = 0; k < layer_sizes[l]; ++k) {
			float sum = 0.0f;
			for (int j = 0; j < layer_sizes[l + 1]; ++j) {
				sum += *get_weight(l + 1, j, k) * error[get_z_index(l + 1, j)];
			}
			error[get_z_index(l, k)] = sum * af_derivative(*get_z(l, k));
		}
	}
	
	// for (int i = 0; i < total_z_size; ++i) {
	// 	printf("%f\n", error[i]);
	// }

	// update gradients
	for (int l = layer_count - 1; l > 0; --l) {
		for (int j = 0; j < layer_sizes[l]; ++j) {
			bias_gradient[get_z_index(l, j)] += error[get_z_index(l, j)];
	//		printf("it's all fake %f\n", bias_gradient[get_z_index(l, j)]);
			for (int k = 0; k < layer_sizes[l - 1]; ++k) {
				float a = l > 1 ? af(*get_z(l - 1, k)) : in[k];
				weight_gradient[get_weight_index(l, j, k)] += a * error[get_z_index(l, j)];
			}
		}
	}
}

void descend_gradient(float *in, float *y, size_t len) {
	float weight_gradient[total_weight_size];
	float bias_gradient[total_z_size];
	// zero initialze weights and biases
	for (int i = 0; i < total_weight_size; ++i) {
		weight_gradient[i] = 0.0f;
	}
	for (int i = 0; i < total_z_size; ++i) {
		bias_gradient[i] = 0.0f;
	}
	for (int i = 0; i < len; ++i) {
		//printf("next image should be a %u", y[i]);
		feedforward(&in[i * layer_sizes[0]]);
		update_gradients(&in[i * layer_sizes[0]], &y[i * layer_sizes[layer_count - 1]], weight_gradient, bias_gradient);
	}

	// printf("gradients: ");
	// for (int l = layer_count - 1; l > 0; --l) {
	// 	for (int j = 0; j < layer_sizes[l]; ++j) {
	// 		for (int k = 0; k < layer_sizes[l - 1]; ++k) {
	// 			printf("%d, %d, %d) => %f\n", l, j, k, weight_gradient[get_weight_index(l, j, k)]);
	// 		}
	// 	}
	// }

	// descend gradient
	#define SPEED 10.0
	for (int l = layer_count - 1; l > 0; --l) {
		for (int j = 0; j < layer_sizes[l]; ++j) {
			*get_bias(l, j) -= SPEED * bias_gradient[get_z_index(l, j)] / (float)len;
			for (int k = 0; k < layer_sizes[l - 1]; ++k) {
				*get_weight(l, j, k) -= SPEED * weight_gradient[get_weight_index(l, j, k)] / (float)len;
			}
		}
	}
}

// testmain
#define TRAINING_MODE
int main(int argc, char **argv) {
	int32_t entries = 6000;
	int32_t rows = 28;
	int32_t columns = 28;
	// init network
	size_t ls[4] = { rows * columns, 16, 16, 10 };
	init(4, ls);

	// yup
	float *images = malloc(entries * rows * columns * sizeof(float));
	float *labels = malloc(entries * layer_sizes[layer_count - 1] * sizeof(float));

	// read in image data
	FILE *fp;
	fp = fopen(argv[1], "rb");

	fseek(fp, 16, SEEK_SET);
	printf("%d\n", entries * rows * columns);
	for (int i = 0; i < entries * rows * columns; ++i) {
		images[i] = (float)fgetc(fp) / 255.0;
	}

	// read in label data
	fp = fopen(argv[2], "rb");

	fseek(fp, 8, SEEK_SET);
	for (int i = 0; i < entries; ++i) {
		unsigned char num = fgetc(fp);
		for (int j = 0; j < layer_sizes[layer_count - 1]; ++j) {
			//printf("%u, ", num);
			float tmp = (j == num) ? 1.0f : 0.0f;
			labels[i * layer_sizes[layer_count - 1] + j] = tmp;
		}
	}

	// set weights and biases to random floats between -3 and 3
	for (int l = layer_count - 1; l > 0; --l) {
		for (int j = 0; j < layer_sizes[l]; ++j) {
			float rn = 1.0 - (float)rand()/(float)(RAND_MAX/6);
			*get_z(l, j) = rn;
			for (int k = 0; k < layer_sizes[l - 1]; ++k) {
				float rn = 1.0 - (float)rand()/(float)(RAND_MAX/6);
				*get_weight(l, j, k) = rn;
			}
		}
	}
#ifdef TRAINING_MODE
	#define BATCH_SIZE 1
	for (int epoch = 0; epoch < 100; ++epoch) {
		for (int i = 0; i < 1; ++i) {
			descend_gradient(&images[BATCH_SIZE * i * layer_sizes[0]], &labels[i * layer_sizes[layer_count - 1]], BATCH_SIZE);
		}

		// calculate cost
		float total_cost = 0.0f;
		for (int j = 0; j < BATCH_SIZE; ++j) {
			feedforward(&images[(0 + j) * layer_sizes[0]]);
			for (int i = 0; i < layer_sizes[layer_count - 1]; ++i) {
				total_cost += cf(af(*get_z(layer_count - 1, i)), labels[j * layer_sizes[layer_count - 1] + i]);
			}
			total_cost /= 10.0;
		}
		total_cost /= (float)BATCH_SIZE;
		printf("epoch %d: cost = %f\n", epoch, total_cost);
	}
#else

#endif

	for(int i = 0; i < 5; ++i) {
		int which = (rand() % BATCH_SIZE);
		feedforward(&images[which * layer_sizes[0]]);
		print_digit(&images[which * layer_sizes[0]]);
		printf("my guess is: { ");
		for (int i = 0; i < layer_sizes[layer_count - 1]; ++i) {
			printf("%f, ", af(*get_z(layer_count - 1, i)));
		}
		printf("}\n");
	}

}