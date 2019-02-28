#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "driver_types.h"
#include <opencv2/core/core.hpp> 
//#include "stit.hpp"
using namespace std;
using namespace cv;
#define LOG_STITCHING_MSG(msg) for(;;) { std::cout << msg; std::cout.flush(); break; }
#define LOG_(_level, _msg)                     \
    for(;;)                                    \
			    {                                          \
        using namespace std;                   \
        if ((_level) >= ::cv::detail::stitchingLogLevel()) \
						        {                                      \
            LOG_STITCHING_MSG(_msg);           \
						        }                                      \
    break;                                 \
			    }
#define LOG(msg) LOG_(1, msg)
#define LOGLN(msg) LOG(msg << std::endl)


#define CALL_HAL(name, fun, ...) \
{ \
    int res = __CV_EXPAND(fun(__VA_ARGS__)); \
    if (res == CV_HAL_ERROR_OK) \
        return; \
	    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res)); \
}


inline int hal_ni_SVD64f(double* src, size_t src_step, double* w, double* u, size_t u_step, double* vt, size_t vt_step, int m, int n, int flags) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
#define cv_hal_SVD64f hal_ni_SVD64f
template<typename T> struct VBLAS
{
	int dot(const T*, const T*, int, T*) const { return 0; }
	int givens(T*, T*, int, T, T) const { return 0; }
	int givensx(T*, T*, int, T, T, T*, T*) const { return 0; }
};

//a.T dot a
template<typename sT, typename dT> static int
MulTransposedR_CUDA(const Mat& srcmat, Mat& dstmat, const Mat& deltamat, double scale){
	const sT* src = srcmat.ptr<sT>();
	dT* dst = dstmat.ptr<dT>();
	Size size = srcmat.size();
	
	const size_t A_ROW = size.height;
	const size_t A_COL = size.width;
	const size_t C_ROW = size.width;
	const size_t C_COL = size.width;

	double *d_A, *d_C;   
	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		LOGLN(status);
		exit(0);
	}
	cudaError_t error_t;
	/*cudaError_t error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("zzzzzzzzzzzzzzzzzzaaaaaaaaaaaaaaaaaaaa cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	if (cudaMalloc((void**)&d_A, sizeof(sT)*A_ROW*A_COL) != cudaSuccess){
		LOGLN("MulTransposedR_CUDA !!!! host memory allocation error (d_A)");
		LOGLN("A_ROW" << A_ROW << "A_COL" << A_COL << "C_ROW" << C_ROW << "C_COL" << C_COL);
		//A_ROW 322446 A_COL 724 B_ROW 322446 B_COL 1 C_ROW 724 C_COL 1
		return	EXIT_FAILURE;
	}
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("xxxxxxxxxxxxxxxxxxxxxxaaaaaaaaaaaaaaaaaaaa cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	if (cudaMalloc((void**)&d_C, sizeof(dT)*C_ROW*C_COL)!=cudaSuccess){
		LOGLN("MulTransposedR_CUDA !!!! host memory allocation error (d_C)");
		LOGLN("A_ROW" << A_ROW << "A_COL" << A_COL << "C_ROW" << C_ROW << "C_COL" << C_COL);
		//A_ROW 322446 A_COL 724 B_ROW 322446 B_COL 1 C_ROW 724 C_COL 1
		return	EXIT_FAILURE;
	}
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("cccccccccccccccccccccccccaaaaaaaaaaaaaaaaaaaa cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	
	cudaMemcpy(d_A, src, sizeof(sT)*A_ROW*A_COL, cudaMemcpyHostToDevice); //数据从内存拷贝到显存
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("vvvvvvvvvvvvvvvvvvvvaaaaaaaaaaaaaaaaaaaa cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	double a = 1, b = 0;
	status=cublasDgemm(
		handle,               //         调用cuBLAS库时的句柄。                                                                       
		CUBLAS_OP_N,          //       是否转置矩阵A                                                                              
		CUBLAS_OP_T,          //        是否转置矩阵B                                                                             
		A_COL,                //        number of rows of matrix op(A) and C.    int m,                                                   
		A_COL,                //        number of columns of matrix op(B) and C.         int n,                                                           
		A_ROW,                //        number of columns of op(A) and rows of op(B).       int k,                                                         
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
	if (status != CUBLAS_STATUS_SUCCESS) {
	LOGLN("!!!! cublasDgemm["<<status<<"]");
	return EXIT_FAILURE;
	}
	error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("bbbbbbbbbbbbbbbbbbbbbbbbbnnnnnnnnnnnnnnnnnnnaaaaaaaaaaaaaaaaaaaa cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}
	cudaMemcpy(dst, d_C, sizeof(sT)*C_ROW*C_COL, cudaMemcpyDeviceToHost);
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("mmmmmmmmmmmmmmmmmmmmmmmmmmmaaaaaaaaaaaaaaaaaaaa cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	cudaFree(d_A);
	cudaFree(d_C);
	cublasDestroy(handle);
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("111111111aaaaaaaaaaaaaaaaaaaaaacudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	/*cudaError_t err_reset = cudaDeviceReset();
	if (err_reset != cudaSuccess){
		LOGLN("cudaDeviceReset faild status is " << err_reset);
		exit(EXIT_FAILURE);
	}*/
	//cudaThreadExit();
	return CUBLAS_STATUS_SUCCESS;
}
void XmulTransposed(InputArray _src, OutputArray _dst, bool ata)
{
	int64 t = getTickCount();
	Mat src = _src.getMat();
	const int gemm_level = 100; // boundary above which GEMM is faster.
	int stype = src.type();
	//dtype = std::max(std::max(CV_MAT_DEPTH(dtype >= 0 ? dtype : stype), delta.depth()), CV_32F);
	CV_Assert(src.channels() == 1);
	//LOGLN("4444444444444444444444444aaaaaaaaaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();

	//LOGLN("44444444444444444444444444bbbbbbbbbbbbbbbbbbbbbbbbbbbb, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	int dsize = ata ? src.cols : src.rows;
	_dst.create(dsize, dsize, stype);
	//LOGLN("44444444444444444444ccccccccccccccccccccc, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	Mat dst = _dst.getMat(); Mat x;
	MulTransposedR_CUDA<double, double>(src, dst, x, 1);



	/*writeMatToFile(src, "d:\\srcmat.matrix");
	writeMatToFile(dst, "d:\\dstmat.matrix");
	writeMatToFile(delta, "d:\\delta.matrix");
	exit(0);*/
}


static void subMatrix(const cv::Mat& src, cv::Mat& dst, const std::vector<uchar>& cols,
	const std::vector<uchar>& rows) {
	int nonzeros_cols = cv::countNonZero(cols);
	cv::Mat tmp(src.rows, nonzeros_cols, CV_64FC1);

	for (int i = 0, j = 0; i < (int)cols.size(); i++)
	{
		if (cols[i])
		{
			src.col(i).copyTo(tmp.col(j++));
		}
	}

	int nonzeros_rows = cv::countNonZero(rows);
	dst.create(nonzeros_rows, nonzeros_cols, CV_64FC1);
	for (int i = 0, j = 0; i < (int)rows.size(); i++)
	{
		if (rows[i])
		{
			tmp.row(i).copyTo(dst.row(j++));
		}
	}
}
template<typename _Tp> void
JacobiSVDImpl_(_Tp* At, size_t astep, _Tp* _W, _Tp* Vt, size_t vstep,int m, int n, int n1, double minval, _Tp eps)
{
	VBLAS<_Tp> vblas;
	AutoBuffer<double> Wbuf(n);
	double* W = Wbuf.data();
	int i, j, k, iter, max_iter = std::max(m, 30);
	_Tp c, s;
	double sd;
	astep /= sizeof(At[0]);
	vstep /= sizeof(Vt[0]);

	for (i = 0; i < n; i++)
	{
		for (k = 0, sd = 0; k < m; k++)
		{
			_Tp t = At[i*astep + k];
			sd += (double)t*t;
		}
		W[i] = sd;

		if (Vt)
		{
			for (k = 0; k < n; k++)
				Vt[i*vstep + k] = 0;
			Vt[i*vstep + i] = 1;
		}
	}

	for (iter = 0; iter < max_iter; iter++)
	{
		bool changed = false;

		for (i = 0; i < n - 1; i++)
			for (j = i + 1; j < n; j++)
			{
				_Tp *Ai = At + i*astep, *Aj = At + j*astep;
				double a = W[i], p = 0, b = W[j];

				for (k = 0; k < m; k++)
					p += (double)Ai[k] * Aj[k];

				if (std::abs(p) <= eps*std::sqrt((double)a*b))
					continue;

				p *= 2;
				double beta = a - b, gamma = hypot((double)p, beta);
				if (beta < 0)
				{
					double delta = (gamma - beta)*0.5;
					s = (_Tp)std::sqrt(delta / gamma);
					c = (_Tp)(p / (gamma*s * 2));
				}
				else
				{
					c = (_Tp)std::sqrt((gamma + beta) / (gamma * 2));
					s = (_Tp)(p / (gamma*c * 2));
				}

				a = b = 0;
				for (k = 0; k < m; k++)
				{
					_Tp t0 = c*Ai[k] + s*Aj[k];
					_Tp t1 = -s*Ai[k] + c*Aj[k];
					Ai[k] = t0; Aj[k] = t1;

					a += (double)t0*t0; b += (double)t1*t1;
				}
				W[i] = a; W[j] = b;

				changed = true;

				if (Vt)
				{
					_Tp *Vi = Vt + i*vstep, *Vj = Vt + j*vstep;
					k = vblas.givens(Vi, Vj, n, c, s);

					for (; k < n; k++)
					{
						_Tp t0 = c*Vi[k] + s*Vj[k];
						_Tp t1 = -s*Vi[k] + c*Vj[k];
						Vi[k] = t0; Vj[k] = t1;
					}
				}
			}
		if (!changed)
			break;
	}

	for (i = 0; i < n; i++)
	{
		for (k = 0, sd = 0; k < m; k++)
		{
			_Tp t = At[i*astep + k];
			sd += (double)t*t;
		}
		W[i] = std::sqrt(sd);
	}

	for (i = 0; i < n - 1; i++)
	{
		j = i;
		for (k = i + 1; k < n; k++)
		{
			if (W[j] < W[k])
				j = k;
		}
		if (i != j)
		{
			std::swap(W[i], W[j]);
			if (Vt)
			{
				for (k = 0; k < m; k++)
					std::swap(At[i*astep + k], At[j*astep + k]);

				for (k = 0; k < n; k++)
					std::swap(Vt[i*vstep + k], Vt[j*vstep + k]);
			}
		}
	}

	for (i = 0; i < n; i++)
		_W[i] = (_Tp)W[i];

	if (!Vt)
		return;

	RNG rng(0x12345678);
	for (i = 0; i < n1; i++)
	{
		sd = i < n ? W[i] : 0;

		for (int ii = 0; ii < 100 && sd <= minval; ii++)
		{
			// if we got a zero singular value, then in order to get the corresponding left singular vector
			// we generate a random vector, project it to the previously computed left singular vectors,
			// subtract the projection and normalize the difference.
			const _Tp val0 = (_Tp)(1. / m);
			for (k = 0; k < m; k++)
			{
				_Tp val = (rng.next() & 256) != 0 ? val0 : -val0;
				At[i*astep + k] = val;
			}
			for (iter = 0; iter < 2; iter++)
			{
				for (j = 0; j < i; j++)
				{
					sd = 0;
					for (k = 0; k < m; k++)
						sd += At[i*astep + k] * At[j*astep + k];
					_Tp asum = 0;
					for (k = 0; k < m; k++)
					{
						_Tp t = (_Tp)(At[i*astep + k] - sd*At[j*astep + k]);
						At[i*astep + k] = t;
						asum += std::abs(t);
					}
					asum = asum > eps * 100 ? 1 / asum : 0;
					for (k = 0; k < m; k++)
						At[i*astep + k] *= asum;
				}
			}
			sd = 0;
			for (k = 0; k < m; k++)
			{
				_Tp t = At[i*astep + k];
				sd += (double)t*t;
			}
			sd = std::sqrt(sd);
		}

		s = (_Tp)(sd > minval ? 1 / sd : 0.);
		for (k = 0; k < m; k++)
			At[i*astep + k] *= s;
	}
}

template <typename fptype> static inline int decodeSVDParameters(const fptype* U, const fptype* Vt, int m, int n, int n1)
{
	int halSVDFlag = 0;
	if (Vt == NULL)
		halSVDFlag = CV_HAL_SVD_NO_UV;
	else if (n1 <= 0 || n1 == n)
	{
		halSVDFlag = CV_HAL_SVD_SHORT_UV;
		if (U == NULL)
			halSVDFlag |= CV_HAL_SVD_MODIFY_A;
	}
	else if (n1 == m)
	{
		halSVDFlag = CV_HAL_SVD_FULL_UV;
		if (U == NULL)
			halSVDFlag |= CV_HAL_SVD_MODIFY_A;
	}
	return halSVDFlag;
}
/* y[0:m,0:n] += diag(a[0:1,0:m]) * x[0:m,0:n] */
template<typename T1, typename T2, typename T3> static void MatrAXPY(int m, int n, const T1* x, int dx, const T2* a, int inca, T3* y, int dy)
{
	int i;
	for (i = 0; i < m; i++, x += dx, y += dy)
	{
		T2 s = a[i*inca];
		int j = 0;
#if CV_ENABLE_UNROLLED
		for (; j <= n - 4; j += 4)
		{
			T3 t0 = (T3)(y[j] + s*x[j]);
			T3 t1 = (T3)(y[j + 1] + s*x[j + 1]);
			y[j] = t0;
			y[j + 1] = t1;
			t0 = (T3)(y[j + 2] + s*x[j + 2]);
			t1 = (T3)(y[j + 3] + s*x[j + 3]);
			y[j + 2] = t0;
			y[j + 3] = t1;
		}
#endif
		for (; j < n; j++)
			y[j] = (T3)(y[j] + s*x[j]);
	}
}
template<typename T> static void SVBkSbImpl_(int m, int n, const T* w, int incw,const T* u, int ldu, bool uT,const T* v, int ldv, bool vT,const T* b, int ldb, int nb,T* x, int ldx, double* buffer, T eps)
{
	double threshold = 0;
	int udelta0 = uT ? ldu : 1, udelta1 = uT ? 1 : ldu;
	int vdelta0 = vT ? ldv : 1, vdelta1 = vT ? 1 : ldv;
	int i, j, nm = std::min(m, n);

	if (!b)
		nb = m;

	for (i = 0; i < n; i++)
		for (j = 0; j < nb; j++)
			x[i*ldx + j] = 0;

	for (i = 0; i < nm; i++)
		threshold += w[i*incw];
	threshold *= eps;

	// v * inv(w) * uT * b
	for (i = 0; i < nm; i++, u += udelta0, v += vdelta0)
	{
		double wi = w[i*incw];
		if ((double)std::abs(wi) <= threshold)
			continue;
		wi = 1 / wi;

		if (nb == 1)
		{
			double s = 0;
			if (b)
				for (j = 0; j < m; j++)
					s += u[j*udelta1] * b[j*ldb];
			else
				s = u[0];
			s *= wi;

			for (j = 0; j < n; j++)
				x[j*ldx] = (T)(x[j*ldx] + s*v[j*vdelta1]);
		}
		else
		{
			if (b)
			{
				for (j = 0; j < nb; j++)
					buffer[j] = 0;
				MatrAXPY(m, nb, b, ldb, u, udelta1, buffer, 0);
				for (j = 0; j < nb; j++)
					buffer[j] *= wi;
			}
			else
			{
				for (j = 0; j < nb; j++)
					buffer[j] = u[j*udelta1] * wi;
			}
			MatrAXPY(n, nb, buffer, 0, v, vdelta1, x, ldx);
		}
	}
}



void SVD64f(double* At, size_t astep, double* W, double* U, size_t ustep, double* Vt, size_t vstep, int m, int n, int n1)
{
	CALL_HAL(SVD64f, cv_hal_SVD64f, At, astep, W, U, ustep, Vt, vstep, m, n, decodeSVDParameters(U, Vt, m, n, n1))
		JacobiSVDImpl_(At, astep, W, Vt, vstep, m, n, !Vt ? 0 : n1 < 0 ? n : n1, DBL_MIN, DBL_EPSILON * 10);
}


static void JacobiSVD(double* At, size_t astep, double* W, double* Vt, size_t vstep, int m, int n, int n1 = -1)
{
	SVD64f(At, astep, W, NULL, astep, Vt, vstep, m, n, n1);
}

static void
SVBkSb(int m, int n, const double* w, size_t wstep,
const double* u, size_t ustep, bool uT,
const double* v, size_t vstep, bool vT,
const double* b, size_t bstep, int nb,
double* x, size_t xstep, uchar* buffer)
{
	SVBkSbImpl_(m, n, w, wstep ? (int)(wstep / sizeof(w[0])) : 1,
		u, (int)(ustep / sizeof(u[0])), uT,
		v, (int)(vstep / sizeof(v[0])), vT,
		b, (int)(bstep / sizeof(b[0])), nb,
		x, (int)(xstep / sizeof(x[0])),
		(double*)alignPtr(buffer, sizeof(double)), DBL_EPSILON * 2);
}

bool Xsolve(InputArray _src, InputArray _src2arg, OutputArray _dst, int method)
{
	bool result = true;
	Mat src = _src.getMat(), _src2 = _src2arg.getMat();
	int type = src.type();
	bool is_normal = (method & DECOMP_NORMAL) != 0;

	CV_Assert(type == _src2.type() && (type == CV_32F || type == CV_64F));

	method &= ~DECOMP_NORMAL;
	CV_Check(method, method == DECOMP_LU || method == DECOMP_SVD || method == DECOMP_EIG ||
		method == DECOMP_CHOLESKY || method == DECOMP_QR,
		"Unsupported method, see #DecompTypes");
	CV_Assert((method != DECOMP_LU && method != DECOMP_CHOLESKY) ||
		is_normal || src.rows == src.cols);


	int m = src.rows, m_ = m, n = src.cols, nb = _src2.cols;
	int64 t = getTickCount();
	size_t esz = CV_ELEM_SIZE(type), bufsize = 0;
	size_t vstep = alignSize(n*esz, 16);
	size_t astep = method == DECOMP_SVD && !is_normal ? alignSize(m*esz, 16) : vstep;
	AutoBuffer<uchar> buffer;

	Mat src2 = _src2;
	_dst.create(src.cols, src2.cols, src.type());
	Mat dst = _dst.getMat();

	if (m < n)
		CV_Error(CV_StsBadArg, "The function can not solve under-determined linear systems");

	if (m == n)
		is_normal = false;
	else if (is_normal)
	{
		m_ = n;
		if (method == DECOMP_SVD)
			method = DECOMP_EIG;
	}
	//LOGLN("aaaaaaaaaaa11111111111111aaaaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	t = getTickCount();
	size_t asize = astep*(method == DECOMP_SVD || is_normal ? n : m);
	bufsize += asize + 32;

	if (is_normal)
		bufsize += n*nb*esz;
	if (method == DECOMP_SVD || method == DECOMP_EIG)
		bufsize += n * 5 * esz + n*vstep + nb*sizeof(double) + 32;

	buffer.allocate(bufsize);
	uchar* ptr = alignPtr(buffer.data(), 16);
	//LOGLN("aaaaaaaaaaa2222222222222222222222222221aaaaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	t = getTickCount();
	Mat a(m_, n, type, ptr, astep);

	if (is_normal)
		XmulTransposed(src, a, true);
	else if (method != DECOMP_SVD)
		src.copyTo(a);
	else
	{

		a = Mat(n, m_, type, ptr, astep);
		cuda::transpose(src, a);
		//cuda::transpose();
	}
	ptr += asize;
	//LOGLN("aaaaaaaaaaa33333333333333333333333333333333331aaaaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	t = getTickCount();
	if (!is_normal)
	{
		if (method == DECOMP_LU || method == DECOMP_CHOLESKY)
			src2.copyTo(dst);
	}
	else
	{
		// a'*b
		if (method == DECOMP_LU || method == DECOMP_CHOLESKY)
			cuda::gemm(src, src2, 1, Mat(), 0, dst, GEMM_1_T);
		else
		{

			t = getTickCount();
			Mat tmp(n, nb, type, ptr);
			ptr += n*nb*esz;
			cuda::gemm(src, src2, 1, Mat(), 0, tmp, GEMM_1_T);
			src2 = tmp;
			LOGLN("aaaaaaaaaaa55555555555555555555555551aaaaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
			//t = getTickCount();
		}
	}

	//LOGLN("aaaaaaaaaaa4444444444444444444444441aaaaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	t = getTickCount();
	
		ptr = alignPtr(ptr, 16);
		Mat v(n, n, type, ptr, vstep), w(n, 1, type, ptr + vstep*n), u;
		ptr += n*(vstep + esz);
		JacobiSVD(a.ptr<double>(), a.step, w.ptr<double>(), v.ptr<double>(), v.step, m_, n);
		u = a;
		SVBkSb(m_, n, w.ptr<double>(), 0, u.ptr<double>(), u.step, true,
				v.ptr<double>(), v.step, true, src2.ptr<double>(),
				src2.step, nb, dst.ptr<double>(), dst.step, ptr);
		result = true;
	
	//LOGLN("aaaaaaaaaaa666666666666666666666666666aaaaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	if (!result)
		dst = Scalar(0);

	return result;
}

void CvLevMarq::step()
{

	//int64 t = getTickCount();
	const double LOG10 = log(10.);
	double lambda = exp(lambdaLg10*LOG10);
	int nparams = param->rows;
	//LOGLN("------aaaaaaaa---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	Mat _JtJ = cvarrToMat(JtJ);
	Mat _mask = cvarrToMat(mask);
	//LOGLN("------bbbbbbbbbbbb---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	int nparams_nz = countNonZero(_mask);
	if (!JtJN || JtJN->rows != nparams_nz) {
		// prevent re-allocation in every step
		JtJN.reset(cvCreateMat(nparams_nz, nparams_nz, CV_64F));
		JtJV.reset(cvCreateMat(nparams_nz, 1, CV_64F));
		JtJW.reset(cvCreateMat(nparams_nz, 1, CV_64F));
	}
	//LOGLN("------ccccccccccccccc---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	Mat _JtJN = cvarrToMat(JtJN);
	Mat _JtErr = cvarrToMat(JtJV);
	Mat_<double> nonzero_param = cvarrToMat(JtJW);
	//LOGLN("------ddddddddddddd---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	subMatrix(cvarrToMat(JtErr), _JtErr, std::vector<uchar>(1, 1), _mask);
	//LOGLN("------eeeeeeeeeeeeeeee---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	subMatrix(_JtJ, _JtJN, _mask, _mask);
	//LOGLN("------ffffffffff---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	/*if (!err)
		completeSymm(_JtJN, completeSymmFlag);*/
	//LOGLN("------gggggggggggggg---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	_JtJN.diag() *= 1. + lambda;
	//LOGLN("------hhhhhhhhhhhhhhh---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	Xsolve(_JtJN, _JtErr, nonzero_param, solveMethod);//DECOMP_SVD      = 1,
	//LOGLN("solveMethod is " << solveMethod << "_JtJN" << _JtJN.type() << "solveMethod" << solveMethod);
	//LOGLN("------iiiiiiiiiiiiiii---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	int j = 0;
	for (int i = 0; i < nparams; i++)
		param->data.db[i] = prevParam->data.db[i] - (mask->data.ptr[i] ? nonzero_param(j++) : 0);
	//LOGLN("------jjjjjjjjjjj---------, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


//cvGEMM(J, err, 1, 0, 0, JtErr, CV_GEMM_A_T);
//GEMM_1_T  transposes src1
template<typename sT, typename dT> static int 
Xgemm(InputArray _A, InputArray _B, double alpha,
	InputArray matC, double beta, OutputArray _D, int flags)
{
	/*cudaError_t err_reset = cudaDeviceReset();
	if (err_reset != cudaSuccess){
		LOGLN("cudaDeviceReset faild status is "<<err_reset);
		exit(EXIT_FAILURE);
	}*/
	cudaError_t error_t;
	/*error_t= cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("111111111bbbbbbbbbbbbbbbbbbbbcudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	/*int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}
	if (deviceCount == 0)
	{
		LOGLN("There are no available device(s) that support CUDA");
		return	EXIT_FAILURE;
	}*/
	
	//gemm(A, B, alpha, C, beta, D, flags);
	Mat A = _A.getMat(), B = _B.getMat();
	const sT* src = A.ptr<sT>();
	const sT* srcB = B.ptr<sT>();
	const int A_ROW = A.rows;
	const int A_COL = A.cols;
	const int B_ROW = B.rows;
	const int B_COL = B.cols;
	const int C_ROW = A.cols;
	const int C_COL = B.cols;

	int len = 0, type = A.type();//CV_64F
	_D.create(C_ROW, C_COL, type);
	Mat D = _D.getMat();
	double* dst = D.ptr<double>();
	
	

	CV_Assert_N(type == B.type(), (type == CV_32FC1 || type == CV_64FC1 || type == CV_32FC2 || type == CV_64FC2));
	
	double *d_A, *d_B, *d_C;    //存储于显存中的矩阵
	error_t = cudaMalloc((void**)&d_A, sizeof(sT)*A_ROW*A_COL);
	if ((error_t != cudaSuccess) && (error_t != 6)){
		error_t = cudaThreadSynchronize();
		if (error_t != cudaSuccess){
			LOGLN("XcvGEMM !!!! host memory allocation error (d_A)");
			LOGLN("A_ROW" << A.rows << "A_COL" << A.cols << "B_ROW" << B.rows << "B_COL" << B.cols << "C_ROW" << A.cols << "C_COL" << B.cols);
			//A_ROW 322929 A_COL 724 B_ROW 322929 B_COL 1 C_ROW 724 C_COL 1
			LOGLN("A.type()" << A.type() << "sizeof(double)" << sizeof(sT) << "error_t" << error_t);
			return	EXIT_FAILURE;
		}
		
	}
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("111ddddddddddddddddddddd111111bbbbbbbbbbbbbbbbbbbbcudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	error_t = cudaMalloc((void**)&d_B, sizeof(sT)*B_ROW*B_COL);
	if ((error_t != cudaSuccess) && (error_t != 6)){
		LOGLN("!!!! host memory allocation error (d_B)");
		LOGLN("A_ROW" << A.rows << "A_COL" << A.cols << "B_ROW" << B.rows << "B_COL" << B.cols << "C_ROW" << A.cols << "C_COL" << B.cols);
		return	EXIT_FAILURE;
	}
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("1111gggggggggggggggggggg11111bbbbbbbbbbbbbbbbbbbbcudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	error_t =cudaMalloc((void**)&d_C, sizeof(dT)*C_ROW*C_COL);
	if ((error_t != cudaSuccess) && (error_t != 6)){
		LOGLN("!!!! host memory allocation error (d_C)");
		LOGLN("A_ROW" << A.rows << "A_COL" << A.cols << "B_ROW" << B.rows << "B_COL" << B.cols << "C_ROW" << A.cols << "C_COL" << B.cols);
		return	EXIT_FAILURE;
	}
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("222222222222222cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	
	if (status != CUBLAS_STATUS_SUCCESS) {
		LOGLN("!!!! CUBLAS initialization error status is [" << status<<"]");
		return EXIT_FAILURE;
	}
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("hhhhhhhhhhhhhhhhhhhhh111111111bbbbbbbbbbbbbbbbbbbbcudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	if (cudaMemcpy(d_A, src, sizeof(sT)*A_ROW*A_COL, cudaMemcpyHostToDevice) != cudaSuccess){
		LOGLN("!!!! host cudaMemcpy allocation error (d_A)");
		return	EXIT_FAILURE;
	}
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("iiiiiiiiiiiiiiiiiiii111111111bbbbbbbbbbbbbbbbbbbbcudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	if (cudaMemcpy(d_B, srcB, sizeof(sT)*B_ROW*B_COL, cudaMemcpyHostToDevice) != cudaSuccess){
		LOGLN("!!!! host cudaMemcpy allocation error (d_B)");
		return	EXIT_FAILURE;
	}
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("33333333333cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	double a = 1, b = 0;
	status = cublasDgemm(
		handle,               //         调用cuBLAS库时的句柄。                                                                       
		CUBLAS_OP_N,          //       是否转置矩阵A                                                                              
		CUBLAS_OP_T,          //        是否转置矩阵B                                                                             
		B_COL,                //        number of rows of matrix op(A) and C.    int m,                                                   
		A_COL,                //        number of columns of matrix op(B) and C.         int n,                                                           
		B_ROW,                //        number of columns of op(A) and rows of op(B).       int k,                                                         
		&a,                   //      <type> scalar used for multiplication.标量 α 的指针，可以是主机指针或设备指针，只需要计算矩阵乘法时命 α = 1.0f；不再单独说明                                    
		d_B,                  //       <type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
		//       矩阵（数组）A 的指针，必须是设备指针；不再单独说明                                                           
		B_COL,                //      矩阵 A 的主维（leading dimension）     lda = num_col_A = num_row_AT = N;   leading dimension of two-dimensional array used to store the matrix A.                                                    
		d_A,                  //       矩阵（数组）B 的指针，必须是设备指针；不再单独说明          
		//       <type> array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
		A_COL,                //      矩阵 B 的主维     ldb = num_col_B = num_row_BT = N;  leading dimension of two-dimensional array used to store matrix B.                                                                        
		&b,                   //       标量 β 的指针，可以是主机指针或设备指针，只需要计算矩阵乘法时命 β = 0.0f；不再单独说明<type> scalar used for multiplication. If beta==0, C does not have to be a valid input.                             
		d_C,                  //  矩阵（数组）C 的指针，必须是设备指针；不再单独说明　 <type> array of dimensions ldc x n with ldc>=max(1,m).                                                              
		C_COL                 //    矩阵 C 的主维　　ldc = num_row_C = N;leading dimension of a two-dimensional array used to store the matrix C.
		);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		LOGLN("cublasDgemm: Failed status: [" << status << "]");
		return EXIT_FAILURE;
	}
	//cublasDgemm(
	//	handle,               //         调用cuBLAS库时的句柄。                                                                       
	//	CUBLAS_OP_N,          //       是否转置矩阵A                                                                              
	//	CUBLAS_OP_T,          //        是否转置矩阵B                                                                             
	//	A_COL,                //        number of rows of matrix op(A) and C.    int m,                                                   
	//	B_COL,                //        number of columns of matrix op(B) and C.         int n,                                                           
	//	A_ROW,                //        number of columns of op(A) and rows of op(B).       int k,                                                         
	//	&a,                   //      <type> scalar used for multiplication.标量 α 的指针，可以是主机指针或设备指针，只需要计算矩阵乘法时命 α = 1.0f；不再单独说明                                    
	//	d_A,                  //       <type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
	//	//       矩阵（数组）A 的指针，必须是设备指针；不再单独说明                                                           
	//	A_COL,                //      矩阵 A 的主维（leading dimension）     lda = num_col_A = num_row_AT = N;   leading dimension of two-dimensional array used to store the matrix A.                                                    
	//	d_B,                  //       矩阵（数组）B 的指针，必须是设备指针；不再单独说明          
	//	//       <type> array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
	//	B_COL,                //      矩阵 B 的主维     ldb = num_col_B = num_row_BT = N;  leading dimension of two-dimensional array used to store matrix B.                                                                        
	//	&b,                   //       标量 β 的指针，可以是主机指针或设备指针，只需要计算矩阵乘法时命 β = 0.0f；不再单独说明<type> scalar used for multiplication. If beta==0, C does not have to be a valid input.                             
	//	d_C,                  //  矩阵（数组）C 的指针，必须是设备指针；不再单独说明　 <type> array of dimensions ldc x n with ldc>=max(1,m).                                                              
	//	C_ROW                 //    矩阵 C 的主维　　ldc = num_row_C = N;leading dimension of a two-dimensional array used to store the matrix C.
	//	);
	error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("444444444444444cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}

	if (cudaMemcpy(dst, d_C, sizeof(dT)*C_ROW*C_COL, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		LOGLN("cudaMemcpy call failed with code dst, d_C");
		exit(EXIT_FAILURE);
	}
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("kkkkkkkkkkkkkkkkkkkkkkkkk111111111bbbbbbbbbbbbbbbbbbbbcudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	if (cudaFree(d_A) != cudaSuccess)
	{
		LOGLN("cudaFree call failed with code d_A");
		exit(EXIT_FAILURE);
	}
	if (cudaFree(d_B) != cudaSuccess)
	{
		LOGLN("cudaFree call failed with code d_B");
		exit(EXIT_FAILURE);
	}
	if (cudaFree(d_C) != cudaSuccess)
	{
		LOGLN("cudaFree call failed with code d_C");
		exit(EXIT_FAILURE);
	}
	cublasDestroy(handle);
	/*error_t = cudaDeviceSynchronize();
	if ((error_t != cudaSuccess)){
		LOGLN("55555555555555cudaDeviceSynchronize() error_t is " << error_t);
		return	EXIT_FAILURE;
	}*/
	/*err_reset = cudaDeviceReset();
	if (err_reset != cudaSuccess){
		LOGLN("cudaDeviceReset faild status is " << err_reset);
		exit(EXIT_FAILURE);
	}*/
	
	return CUBLAS_STATUS_SUCCESS;
}