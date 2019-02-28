#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <opencv2/core/core.hpp> 
using namespace std;
using namespace cv;
#define TILE_DIM    16
#define BLOCK_ROWS  16
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static void
XMulTransposedR(const Mat& srcmat, Mat& dstmat, const Mat& deltamat, double scale){
	int i, j, k;
	const double* src = srcmat.ptr<double>();
	double* dst = dstmat.ptr<double>();
	double* tdst = dst;
	Size size = srcmat.size();

	const size_t A_ROW = size.height;
	const size_t A_COL = size.width;
	const size_t C_ROW = size.width;
	const size_t C_COL = size.width;

	double *d_A, *d_C;    //存储于显存中的矩阵
	cudaMalloc((void**)&d_A, sizeof(double)*A_ROW*A_COL); //在显存中开辟空间
	//cudaMalloc((void**)&d_B, sizeof(double)*B_ROW*B_COL);
	cudaMalloc((void**)&d_C, sizeof(double)*C_ROW*C_COL);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaMemcpy(d_A, src, sizeof(double)*A_ROW*A_COL, cudaMemcpyHostToDevice); //数据从内存拷贝到显存
	double a = 1, b = 0;
	cublasDgemm(
		handle,               //         调用cuBLAS库时的句柄。                                                                       
		CUBLAS_OP_N,          //       是否转置矩阵A                                                                              
		CUBLAS_OP_T,          //        是否转置矩阵B                                                                             
		A_COL,                //        number of rows of matrix op(A) and C.    int m,                                                   
		A_COL,                //        number of columns of matrix op(B) and C.         int n,                                                           
		A_COL,                //        number of columns of op(A) and rows of op(B).       int k,                                                         
		&a,                   //      <type> scalar used for multiplication.标量 α 的指针，可以是主机指针或设备指针，只需要计算矩阵乘法时命 α = 1.0f；不再单独说明                                    
		d_A,                  //       <type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
		//       矩阵（数组）A 的指针，必须是设备指针；不再单独说明                                                           
		A_COL,                //      矩阵 A 的主维（leading dimension）     lda = num_col_A = num_row_AT = N;   leading dimension of two-dimensional array used to store the matrix A.                                                    
		d_A,                  //       矩阵（数组）B 的指针，必须是设备指针；不再单独说明          
		//       <type> array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
		A_COL,                //      矩阵 B 的主维     ldb = num_col_B = num_row_BT = N;  leading dimension of two-dimensional array used to store matrix B.                                                                        
		&b,                   //       标量 β 的指针，可以是主机指针或设备指针，只需要计算矩阵乘法时命 β = 0.0f；不再单独说明<type> scalar used for multiplication. If beta==0, C does not have to be a valid input.                             
		d_C,                  //  矩阵（数组）C 的指针，必须是设备指针；不再单独说明　 <type> array of dimensions ldc x n with ldc>=max(1,m).                                                              
		C_COL                 //    矩阵 C 的主维　　ldc = num_row_C = N;leading dimension of a two-dimensional array used to store the matrix C.
		);
	cudaMemcpy(dst, d_C, sizeof(double)*C_ROW*C_COL, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_C);
}
