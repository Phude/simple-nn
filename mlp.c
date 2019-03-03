#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#define MAX_LAYERS 10
#define NDEBUG
char asciimap[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.";

// TODO: wrap static variables in a "state", struct
size_t layer_count;
size_t layer_sizes[MAX_LAYERS];

size_t weight_count;
size_t weight_indexes[MAX_LAYERS];
float weights[100 * 1000];
size_t bias_indexes[MAX_LAYERS];
size_t bias_count;
float biases[10 * 1000];

float current_input[10 * 1000];
float z[10000];
//float activations[10 * 1000];

// math
// void linear_transform(float *result, float *a, float *x, float *b, size_t in_len, size_t out_len) {
// 	for (int j = 0; j < out_len; ++j) {
// 		for (int i = 0; i < in_len; ++i) {
// 			result[j] += a[j * in_len + i] * x[i];
// 		}
// 		result[j] += b[j];
// 	}
// }

// float *vec_add(float *result, float *v1, float *v2, size_t len) {
// 	for (int i = 0; i < len; ++i) {
// 		result[i] = v1[i] + v2[i];
// 	}
// }


// my reLUs are leaky
#define LEAK 50.0
float reLU(float n) {
	if(n >= 0.0)
		return n;
	else
		return n / LEAK;
}

float reLU_derivative(float n) {
	if (n >= 0.0)
		return 1.0;
	else
		return 1.0 / LEAK;
}

float sigmoid(float n) {
	double ex = exp(n);
	double r = ex / (1 + ex);
	return r;
}

float sigmoid_derivative(float n) {
	double ex = exp(n);
	double r = ex / ((1 + ex) * (1 + ex));
	return r;
}

float cf(float a, float y) {
	return 2 * (a - y) * (a - y);
}

float cf_derivative(float a, float y) {
	return (a - y);
}

// not math

size_t get_weight_index(size_t l, size_t j, size_t k) {
	return weight_indexes[l - 1] + j * layer_sizes[l - 1] + k;
}

size_t get_z_index(size_t l, size_t j) {
	return bias_indexes[l - 1] + j;
}

float *get_z(size_t l, size_t j) {
//	printf("you requested the z value at index %zu\n", get_z_index(l, j));
	return &z[get_z_index(l, j)];
}

float *get_weight(size_t l, size_t j, size_t k) {
	return &weights[weight_indexes[l - 1] + j * layer_sizes[l - 1] + k];
}

float *get_bias(size_t l, size_t j) {
	return &biases[bias_indexes[l - 1] + j];
}

// TODO: it feels a little strange that this in the only get_* functions that returns a value as opposed to a pointer
float get_a(size_t l, size_t j) {
	return reLU(*get_z(l, j));
}

float get_da(size_t l, size_t j) {
	return reLU_derivative(*get_z(l, j));
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
// void feedforward() {
// //	print_digit(input);

// 	float *a = activations + layer_size[0];
// 	float *prev_a = a - layer_size[0]; // == activations
// 	for (int l = 1; l < layer_count; ++l) {
// 		for (int j = 0; j < layer_sizes[l]; ++j) {
// 			float new_z = 0.0f; 
// 			for (int k = 0; k < layer_sizes[l - 1]; ++k) {
// 				new_z += activations[k] * *get_weight(l, j, k);
// 				prev_a[i] = af(*get_z(l, i));
// 			}

// 			*get_z(l, j) = new_z + *get_bias(l, j);
// 		}

// 		for (int i = 0; i < layer_sizes[l]; ++i)
// 	}

// 	printf("{ ");
// 	for (int i = 0; i < layer_sizes[layer_count - 1]; ++i) {
// 	 	printf("%0.2f, ", af(*get_z(layer_count - 1, i)));
// 	}
// 	printf("}\n");
// }

double randn(double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
	{
	  call = !call;
	  return (mu + sigma * (double) X2);
	}
 
  do
	{
	  U1 = -1 + ((double) rand () / RAND_MAX) * 2;
	  U2 = -1 + ((double) rand () / RAND_MAX) * 2;
	  W = pow (U1, 2) + pow (U2, 2);
	}
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}

float dot(float *v1, float *v2, size_t len) {
	float sum = 0.0f;
	for (int i = 0; i < len; ++i) {
		sum += v1[i] * v2[i];
	}
	return sum;
}

void feedforward(float *input) {
//	print_digit(input);
// #ifndef NDEBUG
// 		printf("feedforwarding...\ninput data = { ");
// 		for (int i = 0; i < layer_sizes[0]; ++i) {
// 			printf("%f, ", input[i]);
// 		}
// 		printf("}\n");
// 		printf("weights = { ");
// 		for (int i = 0; i < weight_count; ++i) {
// 			printf("%f, ", weights[i]);
// 		}
// 		printf("}\n");
// 		printf("biases = { ");
// 		for (int i = 0; i < bias_count; ++i) {
// 			printf("%f, ", biases[i]);
// 		}
// 		printf("}\n");
// #endif

		int l = 1;
		for (int j = 0; j < layer_sizes[l]; ++j) {
			*get_z(l, j) = *get_bias(l, j) + dot(input, get_weight(l, j, 0), layer_sizes[l - 1]);
		}

		for (l = 2; l < layer_count; ++l) {
			float prev_activations[layer_sizes[l - 1]];
			for (int k = 0; k < layer_sizes[l - 1]; ++k) {
				prev_activations[k] = get_a(l - 1, k);
			}
			for (int j = 0; j < layer_sizes[l]; ++j) {
				*get_z(l, j) = *get_bias(l, j) + dot(prev_activations, get_weight(l, j, 0), layer_sizes[l - 1]);
				//printf("adding bias: %f\n", *get_bias(l, j));
		}
	}
#ifndef NDEBUG
		printf("z = { ");
		for (int i = 0; i < bias_count; ++i) {
			 printf("%f, ", z[i]);
		}
		printf("}\n");
#endif
}

void update_gradients(float *in, float *y, float *weight_gradient, float *bias_gradient) {
	// caclulate error for each neuron
	float error[bias_count];

	int l = layer_count - 1;
	// output layer
	for (int j = 0; j < layer_sizes[l]; ++j) {
		float z = *get_z(l, j);
		float a = get_a(l, j);
//		printf("cf_D: %d => %f\n", j, cf_derivative(a, y[j]) * af_derivative(z));
		error[get_z_index(l, j)] = cf_derivative(a, y[j]) * get_da(l, j);
	}

	// other layers
	for (l -= 1; l > 0; --l) {
		for (int k = 0; k < layer_sizes[l]; ++k) {
			float sum = 0.0f;
			for (int j = 0; j < layer_sizes[l + 1]; ++j) {
				sum += *get_weight(l + 1, j, k) * error[get_z_index(l + 1, j)];
			}
			error[get_z_index(l, k)] = sum * get_da(l, k);
		}
	}
	
#ifndef NDEBUG
	for (int i = 0; i < bias_count; ++i) {
		printf("error: %f\n", error[i]);
	}
#endif

	// update gradients
	for (int l = layer_count - 1; l > 0; --l) {
		for (int j = 0; j < layer_sizes[l]; ++j) {
			bias_gradient[get_z_index(l, j)] += error[get_z_index(l, j)];
//			printf("new bias gradient at neuron %d in layer %d == %f\n", j, l, bias_gradient[get_z_index(l, j)]);
			for (int k = 0; k < layer_sizes[l - 1]; ++k) {
				float a = l > 1 ? get_a(l -1, k) : in[k];
				weight_gradient[get_weight_index(l, j, k)] += a * error[get_z_index(l, j)];
//				printf("new weight gradient from neuron %d to %d in layer %d == %f\n", k, j, l, weight_gradient[get_weight_index(l, j, k)]);
			}
		}
	}
}

void descend_gradient(float *in, float *y, size_t len) {
	float weight_gradient[weight_count];
	float bias_gradient[bias_count];
	// zero initialze weights and bias gradients
	for (int i = 0; i < weight_count; ++i) {
		weight_gradient[i] = 0.0f;
	}
	for (int i = 0; i < bias_count; ++i) {
		bias_gradient[i] = 0.0f;
	}
	for (int i = 0; i < len; ++i) {
#ifndef NDEBUG
		printf("training example %d:\n", i);
		printf("in: ");
		for (int j = 0; j < layer_sizes[0]; ++j) {
			printf("%f, ", in[layer_sizes[0] * i + j]);
		}
		printf("\nout: ");
		for (int j = 0; j < layer_sizes[layer_count - 1]; ++j) {
			printf("%f, ", y[layer_sizes[layer_count - 1] * i + j]);
		}
		printf("\n");
#endif
		feedforward(&in[i * layer_sizes[0]]);
		update_gradients(&in[i * layer_sizes[0]], &y[i * layer_sizes[layer_count - 1]], weight_gradient, bias_gradient);
	}

#ifndef NDEBUG
	printf("gradients: ");
	for (int l = layer_count - 1; l > 0; --l) {
		for (int j = 0; j < layer_sizes[l]; ++j) {
			for (int k = 0; k < layer_sizes[l - 1]; ++k) {
				printf("%d, %d, %d) => %f\n", l, j, k, weight_gradient[get_weight_index(l, j, k)]);
			}
		}
	}
#endif
	// descend gradient
	#define SPEED 0.005f
	#define DECAY 0.0001f
	for (int l = layer_count - 1; l > 0; --l) {
		for (int j = 0; j < layer_sizes[l]; ++j) {
			*get_bias(l, j) -= (SPEED / (float)len) * bias_gradient[get_z_index(l, j)];
			for (int k = 0; k < layer_sizes[l - 1]; ++k) {
				*get_weight(l, j, k) *= (1.0f - (SPEED * DECAY));
				*get_weight(l, j, k) -= (SPEED / (float)len) * weight_gradient[get_weight_index(l, j, k)];
			}
		}
	}
}

void init(size_t lc, size_t *ls) {
	srand(time(NULL));

	layer_count = lc;

	bias_count = 0;
	weight_count = 0;
	for (int i = 0; i < lc; ++i) {
		layer_sizes[i] = ls[i];
		if (i > 0) {
			bias_indexes[i - 1] = bias_count;
			weight_indexes[i - 1] = weight_count;
			bias_count += ls[i];
			weight_count += ls[i] * ls[i - 1];
		}
	}

	// initialize weights and biases
	for (int l = layer_count - 1; l > 0; --l) {
		for (int j = 0; j < layer_sizes[l]; ++j) {
			*get_bias(l, j) = randn(0.0, 1.0);
			for (int k = 0; k < layer_sizes[l - 1]; ++k) {
				float sd = 1.0 / sqrt(layer_sizes[l - 1]);
				*get_weight(l, j, k) = randn(0.0, sd);
			}
		}
	}
}

void swap_vec(float *vecptr, size_t ia, size_t ib, size_t item_size) {
	float *a;
	float *b;
	for (int i = 0; i < item_size; ++i) {
		a = &vecptr[ia * item_size + i];
		b = &vecptr[ib * item_size + i];
		float tmp = *a;
		*a = *b;
		*b = tmp;
	}
}

void test(float *tdata, float *labels, size_t len) {
	printf("running tests...\n");
	float score = 0;
	for (int i = 0; i < len; ++i) {
		print_digit(&tdata[layer_sizes[0] * i]);
		feedforward(&tdata[layer_sizes[0] * i]);

		// inteprate output layer
		uint8_t guess;
		float highest_val = 0.0f;
		for (int j = 0; j < layer_sizes[layer_count - 1]; ++j) {
			float a = get_a(layer_count - 1, j);
			if (a > highest_val) {
				highest_val = a;
				guess = j;
			}
		}
		printf("my guess for this training example is %d\n", guess);

		if (guess == labels[i])
			score += 1.0;
	}
	score /= len;
	printf(" score = %f\n", score);
}

#define TRAINING_SET "./mnist/training-images"
#define TRAINING_LABELS "./mnist/training-labels"
#define TESTING_SET "./mnist/testing-images"
#define TESTING_LABELS "./mnist/testing-labels"

// testmain
#define TRAINING_MODE
int main(int argc, char **argv) {

	int32_t entries = 60000;
	int32_t rows = 28;
	int32_t columns = 28;

	// init network
	size_t ls_xor[] = { 2, 4, 4, 1 };
	size_t ls_digits[] = {28 * 28, 15, 10 };

	init(3, ls_digits);

	float *images = malloc(entries * layer_sizes[0] * sizeof(float));
	float *labels = malloc(entries * layer_sizes[layer_count - 1] * sizeof(float));

	// read in training image data
	FILE *fp;
	fp = fopen(TRAINING_SET, "rb");

	fseek(fp, 16, SEEK_SET);
	for (int i = 0; i < entries * rows * columns; ++i) {
		images[i] = (float)fgetc(fp) / 255.0;
	}

	// read in training label data

	fp = fopen(TRAINING_LABELS, "rb");

	fseek(fp, 8, SEEK_SET);
	for (int i = 0; i < entries; ++i) {
		unsigned char num = fgetc(fp);
		for (int j = 0; j < layer_sizes[layer_count - 1]; ++j) {
			//printf("%u, ", num);
			float tmp = (j == num) ? 1.0f : 0.0f;
			labels[i * layer_sizes[layer_count - 1] + j] = tmp;
		}
	}

	// ===============================fml
	#define TESTING_ENTRIES 10 * 1000
	float *testing_images = malloc(TESTING_ENTRIES * layer_sizes[0] * sizeof(float));
	float *testing_labels = malloc(TESTING_ENTRIES * sizeof(float));
	// read in testing image data
	fp = fopen(TESTING_SET, "rb");

	fseek(fp, 16, SEEK_SET);
	for (int i = 0; i < TESTING_ENTRIES * rows * columns; ++i) {
		testing_images[i] = (float)fgetc(fp) / 255.0;
	}

	// read in testing label data
	fp = fopen(TESTING_LABELS, "rb");

	fseek(fp, 8, SEEK_SET);
	for (int i = 0; i < TESTING_ENTRIES; ++i) {
		testing_labels[i] = (float)fgetc(fp);
	}

#define TRAINING_DATA_COUNT 60 * 1000
#ifdef TRAINING_MODE
	#define BATCH_SIZE 5
	for (int epoch = 0; epoch < 30; ++epoch) {
		// shuffle training data (and labels)
		printf("shuffling training data...");
		for (int i = TRAINING_DATA_COUNT - 1; i > 1; --i) {
			int swap_with = rand() % (i + 1);
			swap_vec(images, i, swap_with, layer_sizes[0]);
			swap_vec(labels, i, swap_with, layer_sizes[layer_count - 1]);
		}
		printf("done");

		// update gradient in using the average gradient of a batch of length BATCH_SIZE
		size_t in_offset = 0;
		size_t label_offset = 0;
		for (int i = 0; i < TRAINING_DATA_COUNT / BATCH_SIZE; ++i) {
			descend_gradient(images + in_offset, labels + label_offset, BATCH_SIZE);	
			in_offset += BATCH_SIZE * layer_sizes[0];
			label_offset += BATCH_SIZE * layer_sizes[layer_count - 1];
		}

		// estimate current strength
		for(int i = 0; i < 5; ++i) {
			int which = (rand() % TRAINING_DATA_COUNT);
			feedforward(&images[which * layer_sizes[0]]);
			print_digit(&images[which * layer_sizes[0]]);
				printf("correct output layer for that image looks like... { ");
				for (int j = 0; j < layer_sizes[layer_count - 1]; ++j) {
					printf("%.2f, ", labels[which * layer_sizes[layer_count - 1] + j]);
				}
					printf("}\n");
			printf("my guess is: { ");
			for (int i = 0; i < layer_sizes[layer_count - 1]; ++i) {
				printf("%0.2f, ", get_a(layer_count - 1, i));
			}
			printf("}\n");
		}
		printf("epoch %d: complete\n", epoch);
	}
#else

#endif

	if (argc == 1) {
		for(int i = 0; i < 10; ++i) {
			int which = (rand() % BATCH_SIZE);
			feedforward(&images[which * layer_sizes[0]]);
			printf("bits to xor: %0.1f, %0.1f\n", images[which * 2], images[which * 2 + 1]);
			printf("my guess is: %f\n", get_a(layer_count - 1, 0));
		}
	}
	else {
		for(int i = 0; i < 5; ++i) {
			int which = (rand() % TRAINING_DATA_COUNT);
			feedforward(&images[which * layer_sizes[0]]);
			print_digit(&images[which * layer_sizes[0]]);
			printf("my guess is: { ");
			for (int i = 0; i < layer_sizes[layer_count - 1]; ++i) {
				printf("%f, ", get_a(layer_count - 1, i));
			}
			printf("}\n");
		}
	}

	test(testing_images, testing_labels, TESTING_ENTRIES);
}