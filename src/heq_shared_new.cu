
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
                
// Add GPU kernel and functions
// turn hist to shared
__global__ void kernel(unsigned char *input, 
                       int           *hist,
		       unsigned int   size){

    register int location = blockIdx.x * blockDim.x+threadIdx.x;

	
    // calculate histogram 
    if (location < size){
		atomicAdd(&(hist[input[location]]),1);
	}
	

  

}

//// turn hist to shared memory	
//// for loop prefix sum has more consistant result
__global__ void cum_calc( unsigned char *input,
			  int           *hist,
			  unsigned int   size){
	

         register int location = threadIdx.x;
	 register int space = 1;
	 register int temp = 0;
	 register int neighbor = 0;
	 //use shared memory
	  __shared__ int Cache[256];
	  Cache[location] = hist[location];

	  __syncthreads();

	  for (register int i = 0; i < 8; i++)
	  {
	    temp = Cache[location];
	    neighbor = 0;
	    if ((location - space) >= 0)
	    {
	      neighbor = Cache[location - space];
	    }


	    if (location >= space)
	    {
	      Cache[location] = temp + neighbor;
	    }

	    space = space * 2;

	    __syncthreads();
	  }

	  //write to result
	  hist[location] = Cache[location]*255/size;

		  
}



///// port lookup to shared memor
__global__ void equalize_output(  unsigned char *input,
				  int           *lookup,
				  unsigned int   size){

   
    register int location = blockIdx.x * blockDim.x+threadIdx.x;
    
    if (location < size){
		input[location] = lookup[input[location]];
     }
    

}









__global__ void  print_hist(int *hist)
{
    register int location =threadIdx.x;


    
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
	
    unsigned int size_img = width*height;
    
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	int size = XSize*YSize;



	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu , size*sizeof(unsigned char)));
        checkCuda(cudaMalloc((void**)&hist  , 256*sizeof(int)));
    
        checkCuda(cudaMemset(hist, 0 , 256*sizeof(int)));

    	
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice));

        
	checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

	  int gridSize1D = 1 + (( size_img - 1) / BLOCK_SIZE_1D);
	  dim3 dimGrid1D(gridSize1D);
	  dim3 dimBlock1D(BLOCK_SIZE_1D);

	// Kernel Call
	#ifdef CUDA_TIMING
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
        printf("here:/n");
        // Add more kernels and functions as needed here
        kernel<<<dimGrid1D, dimBlock1D>>>(input_gpu, 
                                          hist,
					  size_img);
       //print_hist<<<1,256>>>(hist);
                                      
        cum_calc<<<1,256 >>>(    input_gpu, 
                                      hist,
                                      size_img);
 
        equalize_output<<<dimGrid1D, dimBlock1D>>>(input_gpu,
                                      hist,
                                      size_img);
        
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
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));

    // Free resources and end the program
	checkCuda(cudaFree(input_gpu));

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

