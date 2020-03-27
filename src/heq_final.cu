
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
 
#include "config.h"

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     


#define BLOCK_SIZE_1D 256
#define DOWN_SAMPLE 64

__constant__ double alpha; //depreciated

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}
/*******************************************************/
/*                                                     */
/*******************************************************/





                
// Add GPU kernel and functions
// turn hist to shared
// also declared register is useless since variables are automatically stored
__global__ void kernel(unsigned char *input, 
                       int           *hist)
{
  if( !(threadIdx.x & DOWN_SAMPLE)){
	    register int location = blockIdx.x * blockDim.x+threadIdx.x;
	    atomicAdd(&(hist[input[location]]),1);
  }
}




//// turn hist to shared memory	
//// also might wanna do it differently
__global__ void cum_calc( int     *hist)
{
         register int location = threadIdx.x;
 	 register int space = 1;
	 register int temp = 0;
	 register int neighbor = 0;

	 //use shared memory
	  __shared__ int shared_hist[256];
	  shared_hist[location] = hist[location];

	  __syncthreads();

	  // use prefix sum (reduce 256 iterations to 8)
	  for (register int i = 0; i < 8; i++)
	  {
	    temp = shared_hist[location];
	    neighbor = 0;

	    if (location >= space)
	    {
	      neighbor = shared_hist[location - space];
	      shared_hist[location] = temp + neighbor;
	    }
	    // technically pointless as the runtime would be dictated by number of max iterations, which is 8
	    else
	    {
		break;
	    }

	    space = space * 2;

	    __syncthreads();
	  }

	 //write to result
	 hist[location] = shared_hist[location];
	
		  
}


__global__ void calc_diff( int   *hist)
{
	hist[256] = hist[255]-hist[0];
}

//// turn hist to shared memory	
//// also might wanna do it differently
__global__ void norm_cum( int   *hist)
{

         register int location = threadIdx.x;

	 hist[location] = hist[location] - hist[0];
	 hist[location] = (long int)hist[location]*255/hist[256];	  
}




///// switching to use shared memory??	  
__global__ void equalize_output(unsigned char *input,
				int           *lookup)
{
    register int location = blockIdx.x * blockDim.x+threadIdx.x;
    input[location] = lookup[input[location]];

}







__global__ void  print_hist(int *hist)
{
    register int location = blockIdx.x*TILE_SIZE+threadIdx.x;
    printf("pixel intensity: %d, value: %d\n",location,hist[location]);
}



__global__ void warmup(unsigned char *input, 
                       unsigned char *output){

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
	
    output[location] = 0;

}





















// NOTE: The data passed on is already padded
void gpu_function(unsigned char *data,  
                  unsigned int height, 
                  unsigned int width){
    
	unsigned char *input_gpu;
	int           *hist; 

	int  size_img = width*height;


	dim3 dimGrid(((size_img-1) / BLOCK_SIZE_1D)+1);
	dim3 dimBlock(BLOCK_SIZE_1D);


	// Allocate arrays in GPU memory

	checkCuda(cudaMalloc((void**)&input_gpu , size_img*sizeof(unsigned char)));
        checkCuda(cudaMalloc((void**)&hist      , 257*sizeof(int)));
    
        checkCuda(cudaMemset(hist, 0 , 257*sizeof(int)));  // set histogram to 0

    	
	// Copy data to GPU
        // constant memory reduce memroy transfer time?
	checkCuda(cudaMemcpyToSymbol(input_gpu, 
				     data, 
				     size_img*sizeof(char), 
				     cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

	// begin timing
	#ifdef CUDA_TIMING
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
        // kernel calls
        kernel<<<dimGrid, dimBlock>>>(input_gpu, 
                                      hist);
        //print_hist<<<16,16>>>(hist);   
                  
        cum_calc<<<1,256 >>>(hist);
 	calc_diff<<<1,1>>>(hist);
        norm_cum<<<1,256 >>>(hist);
 
	//print_hist<<<16,16>>>(hist);
        equalize_output<<<dimGrid, dimBlock>>>(input_gpu,
		                               hist);

        // From here on, no need to change anything
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
	
	#ifdef CUDA_TIMING
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			     input_gpu, 
			     size_img*sizeof(unsigned char), 
			     cudaMemcpyDeviceToHost));

	// Free resources and end the program (not required as cuda frees its memory upon exiting the executables
        // Good practice to have but for the competition we are gonna ignore this (and we will definately not run out of memory 
	//checkCuda(cudaFree(input_gpu));
	//checkCuda(cudaFree(hist));

}





















void gpu_warmup(unsigned char *data, 
                unsigned int height, 
                unsigned int width){
    
	unsigned char *input_gpu;
	unsigned char *output_gpu;

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);

	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;

	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;

	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));

	checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));

	// Copy data to GPU
	checkCuda(cudaMemcpy(input_gpu, 
			     data, 
			     size*sizeof(char), 
			     cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());
        
	// Execute algorithm
        
	dim3 dimGrid(gridXSize, gridYSize);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	warmup<<<dimGrid, dimBlock>>>(input_gpu, 
		                      output_gpu);
		                         
	checkCuda(cudaDeviceSynchronize());

	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
		  output_gpu, 
		  size*sizeof(unsigned char), 
		  cudaMemcpyDeviceToHost));
                        
    // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));

}

