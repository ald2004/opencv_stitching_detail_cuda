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

	double *d_A, *d_C;    //�洢���Դ��еľ���
	cudaMalloc((void**)&d_A, sizeof(double)*A_ROW*A_COL); //���Դ��п��ٿռ�
	//cudaMalloc((void**)&d_B, sizeof(double)*B_ROW*B_COL);
	cudaMalloc((void**)&d_C, sizeof(double)*C_ROW*C_COL);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaMemcpy(d_A, src, sizeof(double)*A_ROW*A_COL, cudaMemcpyHostToDevice); //���ݴ��ڴ濽�����Դ�
	double a = 1, b = 0;
	cublasDgemm(
		handle,               //         ����cuBLAS��ʱ�ľ����                                                                       
		CUBLAS_OP_N,          //       �Ƿ�ת�þ���A                                                                              
		CUBLAS_OP_T,          //        �Ƿ�ת�þ���B                                                                             
		A_COL,                //        number of rows of matrix op(A) and C.    int m,                                                   
		A_COL,                //        number of columns of matrix op(B) and C.         int n,                                                           
		A_COL,                //        number of columns of op(A) and rows of op(B).       int k,                                                         
		&a,                   //      <type> scalar used for multiplication.���� �� ��ָ�룬����������ָ����豸ָ�룬ֻ��Ҫ�������˷�ʱ�� �� = 1.0f�����ٵ���˵��                                    
		d_A,                  //       <type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
		//       �������飩A ��ָ�룬�������豸ָ�룻���ٵ���˵��                                                           
		A_COL,                //      ���� A ����ά��leading dimension��     lda = num_col_A = num_row_AT = N;   leading dimension of two-dimensional array used to store the matrix A.                                                    
		d_A,                  //       �������飩B ��ָ�룬�������豸ָ�룻���ٵ���˵��          
		//       <type> array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
		A_COL,                //      ���� B ����ά     ldb = num_col_B = num_row_BT = N;  leading dimension of two-dimensional array used to store matrix B.                                                                        
		&b,                   //       ���� �� ��ָ�룬����������ָ����豸ָ�룬ֻ��Ҫ�������˷�ʱ�� �� = 0.0f�����ٵ���˵��<type> scalar used for multiplication. If beta==0, C does not have to be a valid input.                             
		d_C,                  //  �������飩C ��ָ�룬�������豸ָ�룻���ٵ���˵���� <type> array of dimensions ldc x n with ldc>=max(1,m).                                                              
		C_COL                 //    ���� C ����ά����ldc = num_row_C = N;leading dimension of a two-dimensional array used to store the matrix C.
		);
	cudaMemcpy(dst, d_C, sizeof(double)*C_ROW*C_COL, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_C);
}
