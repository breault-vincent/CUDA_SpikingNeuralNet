#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <array>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <random>
#include <time.h>
#include <Windows.h>

using namespace std;


__global__ void propagateSignal(float*layer1, float*flatMatrix, float*layer2, int col, int lenghtOfRow){
	//test kernel, not used in simulation.

	
	int row = threadIdx.x;
	printf("%d \n", row);
	float sum = 0.0f;
	for (col = 0; col < lenghtOfRow; col++){
		sum += layer1[row] * flatMatrix[(row*lenghtOfRow) + col];
	}
	layer2[row] = sum;
	printf("%f \n", layer2[row]);
	sum = 0.0f;
}

__global__ void propagateSignalBlock(float**flatMatrix, float**theNetwork, int col, int lenghtOfRow){
	//test kernel, not used in simulation.
	int layer = blockIdx.x;
	int row = threadIdx.x;
	
	float sum = 0.0f;
	for (col = 0; col < lenghtOfRow; col++){
		sum += theNetwork[layer * 2][row] * flatMatrix[layer][(row*lenghtOfRow) + col];
	}
	theNetwork[layer*2+1][row] += sum;
	sum = 0.0f;
}

__global__ void propagateSignalBlockOfConnection(float**flatMatrix, float**theNetwork, float* randomInput, float threshold, int col, int lenghtOfRow){

	/*
	Each block calculates the propagation of information for a pair of connected layers,
	along each connection matrix. The blocks are further divided into 100 threads, each calculating
	the signal propagated into the matrix by one neuron and then propagated to the next layer.
	*/
	int connectionMatrix = blockIdx.x;
	int row = threadIdx.x;
	
	int fired = 0;

	float sum = 0.0f;
	if (connectionMatrix < 2){
		theNetwork[0][row] += randomInput[row];
		if (theNetwork[0][row] >= threshold){
			theNetwork[0][row] = threshold;
			for (col = 0; col < lenghtOfRow; col++){
				theNetwork[connectionMatrix + 1][col] += theNetwork[0][row] * flatMatrix[connectionMatrix][(row*lenghtOfRow) + col];
			}
			theNetwork[0][row] = 0.0f;
		}
	}

	else if (connectionMatrix == 2){
		if (theNetwork[1][row] >= threshold){
			theNetwork[1][row] = threshold;
			for (col = 0; col < lenghtOfRow; col++){
				theNetwork[2][col] += theNetwork[1][row] * flatMatrix[connectionMatrix][(row*lenghtOfRow) + col];
			}
			theNetwork[1][row] = 0.0f;
		}
	}
	else if (connectionMatrix == 3){
		if (theNetwork[2][row] >= threshold){
			theNetwork[2][row] = threshold;
			for (col = 0; col < lenghtOfRow; col++){
				theNetwork[1][col] += theNetwork[2][row] * flatMatrix[connectionMatrix][(row*lenghtOfRow) + col];
			}
			theNetwork[2][row] = 0.0f;
		}
	}
	
	else if (connectionMatrix == 4){
		if (theNetwork[1][row] >= threshold){
			theNetwork[1][row] = threshold;
			for (col = 0; col < lenghtOfRow; col++){
				theNetwork[3][col] += theNetwork[1][row] * flatMatrix[connectionMatrix][(row*lenghtOfRow) + col];
			}
			theNetwork[1][row] = 0.0f;
		}
	}
	else if (connectionMatrix == 5){
		if (theNetwork[2][row] >= threshold){
			theNetwork[2][row] = threshold;
			for (col = 0; col < lenghtOfRow; col++){
				theNetwork[3][col] += theNetwork[2][row] * flatMatrix[connectionMatrix][(row*lenghtOfRow) + col];
			}
			theNetwork[1][row] = 0.0f;
		}
	}

	
	else if (connectionMatrix >= 6){
		if (theNetwork[connectionMatrix - 6][row] >= threshold){
			theNetwork[connectionMatrix - 6][row] = threshold;
			for (col = 0; col < lenghtOfRow; col++){
				theNetwork[connectionMatrix - 6][col] += theNetwork[connectionMatrix - 6][row] * flatMatrix[connectionMatrix][(row*lenghtOfRow) + col];
			}
			theNetwork[connectionMatrix - 6][row] = 0.0f;
		}
	}
}

__global__ void printLayerInfo(float**theNetwork, float lenghtOfRow){
	
	int layer = threadIdx.x;
	float avg = 0.0f;
	float active = 0.0f;
	for (int i = 0; i < lenghtOfRow; i++){
		avg += theNetwork[layer][i];;
		if (theNetwork[layer][i] >= 1.0f){
			active += 1.0f;
		}
	}
	avg = avg / lenghtOfRow;
	printf("Layer %d has an average activation of %f and it has %f active neurons. \n", layer, avg, active);
}

int main()
{
	cudaError_t cudaStatus;

	// Setting up random number generator
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0, 1);

	// Initialisation of parameters!
	// Size of each group of neurons
	const int layerSize = 100;
	// Threshold for neuron fireing
	const float threshold = 1.0f;
	// Rate of connection in group
	const float rateInGroup = 0.1;
	// Rate of connection our group
	const float rateOutGroup = 0.01;
	// Connection weight factor
	const float factor = 0.2;



	// Array of layers on host machine
	float ** host_theNetwork = new float*[4];

	for (int i = 0; i < 4; i++)
	{
		host_theNetwork[i] = new float[layerSize];
		for (int x = 0; x < layerSize; x++)
		{
			host_theNetwork[i][x] = distribution(generator);
		}
	}
	// Array of pointers on device for layers
	float ** dev_host_theNetwork = new float*[4];
	for (int i = 0; i < 4; i++)
	{
		
		cudaStatus = cudaMalloc((void**)&dev_host_theNetwork[i], layerSize * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_host_theNetwork[i], host_theNetwork[i], layerSize * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

	}
	
	float ** device_theNetwork;
	cudaStatus = cudaMalloc((void**)&device_theNetwork, 4 * sizeof(float*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(device_theNetwork, dev_host_theNetwork, 4 * sizeof(float*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}



	// Array of pointers on device for connections
	float ** host_theConnections = new float*[10];
	// Create 10 flat matrices
	for (int i = 0; i < 10; i++)
	{
		host_theConnections[i] = new float[10000];
		for (int x = 0; x < 10000; x++)
		{
			if (i < 6 && (distribution(generator) < rateOutGroup)){
				host_theConnections[i][x] = distribution(generator)*factor;
			}
			else if (i >= 6 && (distribution(generator) < rateInGroup)){
				host_theConnections[i][x] = distribution(generator)*factor;
			}
			else{
				host_theConnections[i][x] = 0.0f;
			}

			
		}
	}

	// Array of pointers on device for connection
	float ** dev_host_theConnections = new float*[10];
	for (int i = 0; i < 10; i++)
	{
		cudaStatus = cudaMalloc((void**)&dev_host_theConnections[i], 10000 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_host_theConnections[i], host_theConnections[i], 10000 * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

	}

	float ** device_theConnections;
	cudaStatus = cudaMalloc((void**)&device_theConnections, 10 * sizeof(float*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(device_theConnections, dev_host_theConnections, 10 * sizeof(float*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Array for random signal coming into input layer
	float randomInput[layerSize] = { 0 };
	float *dev_randomInput;
	
	for (int i = 0; i < layerSize; i++){
		randomInput[i] = distribution(generator)*0.5;

	}

	cudaStatus = cudaMalloc((void**)&dev_randomInput, layerSize * sizeof(float*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_randomInput, randomInput, layerSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Choose which GPU to run on, change this on a multi-GPU system
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!");
		goto Error;
	}

	


	// Kernel calls
	int number = 0;
	while (number < 10000){



		printLayerInfo << <1, 4 >> >(device_theNetwork, layerSize);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed!");
			goto Error;
		}
		propagateSignalBlockOfConnection << <10, layerSize >> >(device_theConnections, device_theNetwork, dev_randomInput, threshold, layerSize, layerSize);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed!");
			goto Error;
		}
		printf("This is iteration # %d. The simulation can be stopped at any time by pressing CTRL+C. \n", number);
		
		

		Sleep(250);
		number += 1;
	}

	// Retrieving processed data after processing
	cudaStatus = cudaMemcpy(dev_host_theNetwork, device_theNetwork, 4 * sizeof(float*), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	for (int i = 0; i < 4; i++){
		cudaStatus = cudaMemcpy(host_theNetwork[i], dev_host_theNetwork[i], layerSize * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		
	}
	
	Error:
	    cudaFree(device_theConnections);
	    cudaFree(device_theNetwork);
	    cudaFree(dev_host_theConnections);
		cudaFree(dev_host_theNetwork);
	    
	    return cudaStatus;
    return 0;
}