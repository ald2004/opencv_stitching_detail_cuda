#pragma once
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "cudaMatrix.hpp"



#define ENABLE_LOG 1
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


using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
int i = 0;
vector<String> img_names;
bool preview = false;
bool try_cuda = true;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "surf";
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_VERT;
double work_scale = 1;
double seam_scale = 1;
double compose_scale = 1;
double seam_work_aspect = 1;
bool is_work_scale_set = false;
bool is_seam_scale_set = false;
bool is_compose_scale_set = false;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "plane";
float match_conf = 0.3f;
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;

//redefine 
//double work_megapix = 0.08;
//double seam_megapix = 0.08;
//double work_megapix = 0.6;
double seam_megapix = 0.1;
double work_megapix = 0.1;
int expos_comp_type = ExposureCompensator::GAIN;
//int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
string seam_find_type = "dp_colorgrad";
//string seam_find_type = "gc_color";

//template<typename sT, typename dT> 

static void printUsage()
{

	cout <<
		//Mat::eye(3, 3, CV_64F)<<endl<<
		"Rotation model images stitcher.\n\n"
		"stitching_detailed img1 img2 [...imgN] [flags]\n\n"
		"Flags:\n"
		"  --preview\n"
		"      Run stitching in the preview mode. Works faster than usual mode,\n"
		"      but output image will have lower resolution.\n"
		"  --try_cuda (yes|no)\n"
		"      Try to use CUDA. The default value is 'no'. All default values\n"
		"      are for CPU mode.\n"
		"\nMotion Estimation Flags:\n"
		"  --work_megapix <float>\n"
		"      Resolution for image registration step. The default is 0.6 Mpx.\n"
		"  --features (surf|orb|sift)\n"
		"      Type of features used for images matching. The default is surf.\n"
		"  --matcher (homography|affine)\n"
		"      Matcher used for pairwise image matching.\n"
		"  --estimator (homography|affine)\n"
		"      Type of estimator used for transformation estimation.\n"
		"  --match_conf <float>\n"
		"      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
		"  --conf_thresh <float>\n"
		"      Threshold for two images are from the same panorama confidence.\n"
		"      The default is 1.0.\n"
		"  --ba (no|reproj|ray|affine)\n"
		"      Bundle adjustment cost function. The default is ray.\n"
		"  --ba_refine_mask (mask)\n"
		"      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
		"      where 'x' means refine respective parameter and '_' means don't\n"
		"      refine one, and has the following format:\n"
		"      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
		"      adjustment doesn't support estimation of selected parameter then\n"
		"      the respective flag is ignored.\n"
		"  --wave_correct (no|horiz|vert)\n"
		"      Perform wave effect correction. The default is 'horiz'.\n"
		"  --save_graph <file_name>\n"
		"      Save matches graph represented in DOT language to <file_name> file.\n"
		"      Labels description: Nm is number of matches, Ni is number of inliers,\n"
		"      C is confidence.\n"
		"\nCompositing Flags:\n"
		"  --warp (affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
		"      Warp surface type. The default is 'spherical'.\n"
		"  --seam_megapix <float>\n"
		"      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
		"  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
		"      Seam estimation method. The default is 'gc_color'.\n"
		"  --compose_megapix <float>\n"
		"      Resolution for compositing step. Use -1 for original resolution.\n"
		"      The default is -1.\n"
		"  --expos_comp (no|gain|gain_blocks)\n"
		"      Exposure compensation method. The default is 'gain_blocks'.\n"
		"  --blend (no|feather|multiband)\n"
		"      Blending method. The default is 'multiband'.\n"
		"  --blend_strength <float>\n"
		"      Blending strength from [0,100] range. The default is 5.\n"
		"  --output <result_img>\n"
		"      The default is 'result.jpg'.\n"
		"  --timelapse (as_is|crop) \n"
		"      Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names.\n"
		"  --rangewidth <int>\n"
		"      uses range_width to limit number of images to match with.\n";
}


static int parseCmdArgs(int argc, char** argv)
{
	if (argc == 1)
	{
		printUsage();
		return -1;
	}
	for (int i = 1; i < argc; ++i)
	{
		if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
		{
			printUsage();
			return -1;
		}
		else if (string(argv[i]) == "--preview")
		{
			preview = true;
		}
		else if (string(argv[i]) == "--try_cuda")
		{
			if (string(argv[i + 1]) == "no")
				try_cuda = false;
			else if (string(argv[i + 1]) == "yes")
				try_cuda = true;
			else
			{
				cout << "Bad --try_cuda flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--work_megapix")
		{
			work_megapix = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--seam_megapix")
		{
			seam_megapix = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--compose_megapix")
		{
			compose_megapix = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--result")
		{
			result_name = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--features")
		{
			features_type = argv[i + 1];
			if (features_type == "orb")
				match_conf = 0.3f;
			i++;
		}
		else if (string(argv[i]) == "--matcher")
		{
			if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
				matcher_type = argv[i + 1];
			else
			{
				cout << "Bad --matcher flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--estimator")
		{
			if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
				estimator_type = argv[i + 1];
			else
			{
				cout << "Bad --estimator flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--match_conf")
		{
			match_conf = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--conf_thresh")
		{
			conf_thresh = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--ba")
		{
			ba_cost_func = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--ba_refine_mask")
		{
			ba_refine_mask = argv[i + 1];
			if (ba_refine_mask.size() != 5)
			{
				cout << "Incorrect refinement mask length.\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--wave_correct")
		{
			if (string(argv[i + 1]) == "no")
				do_wave_correct = false;
			else if (string(argv[i + 1]) == "horiz")
			{
				do_wave_correct = true;
				wave_correct = detail::WAVE_CORRECT_HORIZ;
			}
			else if (string(argv[i + 1]) == "vert")
			{
				do_wave_correct = true;
				wave_correct = detail::WAVE_CORRECT_VERT;
			}
			else
			{
				cout << "Bad --wave_correct flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--save_graph")
		{
			save_graph = true;
			save_graph_to = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--warp")
		{
			warp_type = string(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--expos_comp")
		{
			if (string(argv[i + 1]) == "no")
				expos_comp_type = ExposureCompensator::NO;
			else if (string(argv[i + 1]) == "gain")
				expos_comp_type = ExposureCompensator::GAIN;
			else if (string(argv[i + 1]) == "gain_blocks")
				expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
			else
			{
				cout << "Bad exposure compensation method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--seam")
		{
			if (string(argv[i + 1]) == "no" ||
				string(argv[i + 1]) == "voronoi" ||
				string(argv[i + 1]) == "gc_color" ||
				string(argv[i + 1]) == "gc_colorgrad" ||
				string(argv[i + 1]) == "dp_color" ||
				string(argv[i + 1]) == "dp_colorgrad")
				seam_find_type = argv[i + 1];
			else
			{
				cout << "Bad seam finding method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "	")
		{
			if (string(argv[i + 1]) == "no")
				blend_type = Blender::NO;
			else if (string(argv[i + 1]) == "feather")
				blend_type = Blender::FEATHER;
			else if (string(argv[i + 1]) == "multiband")
				blend_type = Blender::MULTI_BAND;
			else
			{
				cout << "Bad blending method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--timelapse")
		{
			timelapse = true;

			if (string(argv[i + 1]) == "as_is")
				timelapse_type = Timelapser::AS_IS;
			else if (string(argv[i + 1]) == "crop")
				timelapse_type = Timelapser::CROP;
			else
			{
				cout << "Bad timelapse method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--rangewidth")
		{
			range_width = atoi(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--blend_strength")
		{
			blend_strength = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--output")
		{
			result_name = argv[i + 1];
			i++;
		}
		else
			img_names.push_back(argv[i]);
	}
	if (preview)
	{
		compose_megapix = 0.6;
	}
	return 0;
}

typedef int(*MulTransposedFunc)(const Mat& src, Mat& dst, const Mat& delta, double scale);

void XmulTransposed(InputArray _src, OutputArray _dst, bool ata,
	InputArray _delta, double scale, int dtype)
{
	int64 t = getTickCount();
	Mat src = _src.getMat(), delta = _delta.getMat();
	const int gemm_level = 100; // boundary above which GEMM is faster.
	int stype = src.type();
	dtype = std::max(std::max(CV_MAT_DEPTH(dtype >= 0 ? dtype : stype), delta.depth()), CV_32F);
	CV_Assert(src.channels() == 1);
	//LOGLN("4444444444444444444444444aaaaaaaaaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	if (!delta.empty())
	{
		CV_Assert_N(delta.channels() == 1,
			(delta.rows == src.rows || delta.rows == 1),
			(delta.cols == src.cols || delta.cols == 1));
		if (delta.type() != dtype)
			delta.convertTo(delta, dtype);
	}
	//LOGLN("44444444444444444444444444bbbbbbbbbbbbbbbbbbbbbbbbbbbb, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	int dsize = ata ? src.cols : src.rows;
	_dst.create(dsize, dsize, dtype);
	//LOGLN("44444444444444444444ccccccccccccccccccccc, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	Mat dst = _dst.getMat();

	if (0 && (src.data == dst.data || (stype == dtype &&
		(dst.cols >= gemm_level && dst.rows >= gemm_level &&
		src.cols >= gemm_level && src.rows >= gemm_level))))
	{
		LOGLN("dgggggggggggggggggggggggggggggggggggggggggggggggg");
		Mat src2;
		const Mat* tsrc = &src;
		if (!delta.empty())
		{

			if (delta.size() == src.size()){
				//t = getTickCount();
				subtract(src, delta, src2);
				//LOGLN("555555555555aaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
			}

			else
			{
				//t = getTickCount();
				repeat(delta, src.rows / delta.rows, src.cols / delta.cols, src2);
				//LOGLN("555555555555bbbbbbbbbbbbbbbbbbbbbbbb, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
				//t = getTickCount();
				subtract(src, src2, src2);
				//LOGLN("5555555555555555555555ccccccccccccccccccccc, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
			}
			tsrc = &src2;
		}
		//t = getTickCount();
		gemm(*tsrc, *tsrc, scale, Mat(), 0, dst, ata ? GEMM_1_T : GEMM_2_T);
		//LOGLN("555555555555ddddddddddddddddddd, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	}
	else
	{
		MulTransposedFunc func = 0;
		if (stype == CV_8U && dtype == CV_32F)
		{
			if (ata)
				func = MulTransposedR_CUDA<uchar, float>;
			
		}
		else if (stype == CV_8U && dtype == CV_64F)
		{
			if (ata)
				func = MulTransposedR_CUDA<uchar, double>;
			
		}
		else if (stype == CV_16U && dtype == CV_32F)
		{
			if (ata)
				func = MulTransposedR_CUDA<ushort, float>;
			
		}
		else if (stype == CV_16U && dtype == CV_64F)
		{
			if (ata)
				func = MulTransposedR_CUDA<ushort, double>;
			
		}
		else if (stype == CV_16S && dtype == CV_32F)
		{
			if (ata)
				func = MulTransposedR_CUDA<short, float>;
			
		}
		else if (stype == CV_16S && dtype == CV_64F)
		{
			if (ata)
				func = MulTransposedR_CUDA<short, double>;
			
		}
		else if (stype == CV_32F && dtype == CV_32F)
		{
			if (ata)
				func = MulTransposedR_CUDA<float, float>;
			
		}
		else if (stype == CV_32F && dtype == CV_64F)
		{
			if (ata)
				func = MulTransposedR_CUDA<float, double>;
			
		}
		else if (stype == CV_64F && dtype == CV_64F)
		{
			if (ata)
				func = MulTransposedR_CUDA<double, double>;
			
		}
		if (!func)
			CV_Error(CV_StsUnsupportedFormat, "");

		func(src, dst, delta, scale);
		
		
	}
	/*writeMatToFile(src, "d:\\srcmat.matrix");
	writeMatToFile(dst, "d:\\dstmat.matrix");
	writeMatToFile(delta, "d:\\delta.matrix");
	exit(0);*/
}


void cvMulTransposed(const CvArr* srcarr, CvArr* dstarr,
	int order, const CvArr* deltaarr, double scale)
{
	cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0, delta;
	if (deltaarr)
		delta = cv::cvarrToMat(deltaarr);
	XmulTransposed(src, dst, order != 0, delta, scale, dst.type());
	if (dst.data != dst0.data)
		dst.convertTo(dst0, dst0.type());
}


void XcvGEMM(const CvArr* Aarr, const CvArr* Barr, double alpha, const CvArr* Carr, double beta, CvArr* Darr, int flags)
{
	cv::Mat A = cv::cvarrToMat(Aarr), B = cv::cvarrToMat(Barr);
	cv::Mat C, D = cv::cvarrToMat(Darr);

	if (Carr)
		C = cv::cvarrToMat(Carr);

	CV_Assert_N((D.rows == ((flags & CV_GEMM_A_T) == 0 ? A.rows : A.cols)),
		(D.cols == ((flags & CV_GEMM_B_T) == 0 ? B.cols : B.rows)),
		D.type() == A.type());
	int stype = A.type();
	int xx = 0;
	if (stype == CV_32F )
	{
		//InputArray _A, InputArray _B, double alpha,InputArray matC, double beta, OutputArray _D, int flags
		xx = Xgemm<float, float>(A, B, alpha, C, beta, D, flags);

	}
	else if (stype == CV_64F )
	{
		xx = Xgemm<double, double>(A, B, alpha, C, beta, D, flags);

	}
	if (xx != CUBLAS_STATUS_SUCCESS){
		LOGLN("Xgemm error status is [" << xx<<"]");
		exit(0);
	}
	
}

bool CvLevMarq::update(const CvMat*& _param, CvMat*& matJ, CvMat*& _err)
{
	int64 t = getTickCount();
	matJ = _err = 0;

	assert(!err.empty());
	if (state == DONE)
	{
		_param = param;
		return false;
	}
	
	//t = getTickCount();
	if (state == STARTED)
	{
		_param = param;
		cvZero(J);
		cvZero(err);
		matJ = J;
		_err = err;
		state = CALC_J;
		return true;
	}
	//LOGLN("1111111111111111bbbbbbbbbbbbbbbbbbb, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	if (state == CALC_J)
	{
		//t = getTickCount();
		cvMulTransposed(J, JtJ, 1);
		//LOGLN("2222222222222aaaaaaaaaaaaaaaaaaaaaaaaaaa, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		//t = getTickCount();
		//cvGEMM(J, err, 1, 0, 0, JtErr, CV_GEMM_A_T);
		XcvGEMM(J, err, 1, 0, 0, JtErr, CV_GEMM_A_T);

		
		//LOGLN("2222222222222bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		//t = getTickCount();
		cvCopy(param, prevParam);
		//LOGLN("2222222222222ccccccccccccccccccc, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		//t = getTickCount();
		step();
		//LOGLN("2222222222222ddddddddddddddddddd, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		//t = getTickCount();
		if (iters == 0)
			prevErrNorm = cvNorm(err, 0, CV_L2);
		//LOGLN("2222222222222eeeeeeeeeeeeeeeeeeeeeeeeeeee, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		//t = getTickCount();
		_param = param;
		cvZero(err);
		_err = err;
		state = CHECK_ERR;
		//LOGLN("111111111111ffffffffffffffffffffffffffff, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		return true;
	}
	
	//LOGLN("11111111111111cccccccccccccccccccc, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	assert(state == CHECK_ERR);
	errNorm = cvNorm(err, 0, CV_L2);
	if (errNorm > prevErrNorm)
	{
		if (++lambdaLg10 <= 16)
		{
			step();
			_param = param;
			cvZero(err);
			_err = err;
			state = CHECK_ERR;
			return true;
		}
	}
	//LOGLN("11111111111111dddddddddddddddddddd, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	lambdaLg10 = MAX(lambdaLg10 - 1, -16);
	if (++iters >= criteria.max_iter || //iters>150||
		cvNorm(param, prevParam, CV_RELATIVE_L2) < criteria.epsilon)
	{
		_param = param;
		state = DONE;
		return true;
	}
	//LOGLN("1111111111111eeeeeeeeeeeeeeeeeeeee, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	//t = getTickCount();
	prevErrNorm = errNorm;
	_param = param;
	cvZero(J);
	matJ = J;
	_err = err;
	state = CALC_J;
	//LOGLN("1111111111111fffffffffffffffffffffffff, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	return true;
}

void calcDeriv(const Mat &err1, const Mat &err2, double h, Mat res)
{
#pragma omp parallel for shared(err1,err2,res,h) schedule(dynamic) 
	for (int i = 0; i < err1.rows; ++i)
		res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}


void BundleAdjusterRay::calcError(Mat &err)
{
	err.create(total_num_matches_ * 3, 1, CV_64F);
	//LOGLN("--------err" << err.dot(err) << "total_num_matches_" << total_num_matches_);
	int match_idx = 0;
	for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx)
	{
		int i = edges_[edge_idx].first;
		int j = edges_[edge_idx].second;
		double f1 = cam_params_.at<double>(i * 4, 0);
		double f2 = cam_params_.at<double>(j * 4, 0);

		double R1[9];
		Mat R1_(3, 3, CV_64F, R1);
		Mat rvec(3, 1, CV_64F);
		rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
		Rodrigues(rvec, R1_);

		double R2[9];
		Mat R2_(3, 3, CV_64F, R2);
		rvec.at<double>(0, 0) = cam_params_.at<double>(j * 4 + 1, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(j * 4 + 2, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(j * 4 + 3, 0);
		Rodrigues(rvec, R2_);

		const ImageFeatures& features1 = features_[i];
		const ImageFeatures& features2 = features_[j];
		const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];

		Mat_<double> K1 = Mat::eye(3, 3, CV_64F);
		K1(0, 0) = f1; K1(0, 2) = features1.img_size.width * 0.5;
		K1(1, 1) = f1; K1(1, 2) = features1.img_size.height * 0.5;

		Mat_<double> K2 = Mat::eye(3, 3, CV_64F);
		K2(0, 0) = f2; K2(0, 2) = features2.img_size.width * 0.5;
		K2(1, 1) = f2; K2(1, 2) = features2.img_size.height * 0.5;

		Mat_<double> H1 = R1_ * K1.inv();
		Mat_<double> H2 = R2_ * K2.inv();

		
		for (size_t k = 0; k < matches_info.matches.size(); ++k)
		{
			if (!matches_info.inliers_mask[k])
				continue;

			//const DMatch& m = matches_info.matches[k];
			const DMatch m = matches_info.matches[k];
			Point2f p1 = features1.keypoints[m.queryIdx].pt;
			double x1 = H1(0, 0)*p1.x + H1(0, 1)*p1.y + H1(0, 2);
			double y1 = H1(1, 0)*p1.x + H1(1, 1)*p1.y + H1(1, 2);
			double z1 = H1(2, 0)*p1.x + H1(2, 1)*p1.y + H1(2, 2);
			double len = std::sqrt(x1*x1 + y1*y1 + z1*z1);
			x1 /= len; y1 /= len; z1 /= len;

			Point2f p2 = features2.keypoints[m.trainIdx].pt;
			double x2 = H2(0, 0)*p2.x + H2(0, 1)*p2.y + H2(0, 2);
			double y2 = H2(1, 0)*p2.x + H2(1, 1)*p2.y + H2(1, 2);
			double z2 = H2(2, 0)*p2.x + H2(2, 1)*p2.y + H2(2, 2);
			len = std::sqrt(x2*x2 + y2*y2 + z2*z2);
			x2 /= len; y2 /= len; z2 /= len;

			double mult = std::sqrt(f1 * f2);
			err.at<double>(3 * match_idx, 0) = mult * (x1 - x2);
			err.at<double>(3 * match_idx + 1, 0) = mult * (y1 - y2);
			err.at<double>(3 * match_idx + 2, 0) = mult * (z1 - z2);

			match_idx++;
		}
	}
}


void BundleAdjusterRay::calcJacobian(Mat &jac)
{
	
	//int64 t = getTickCount();
	//cuda::GpuMat jac;
	jac.create(total_num_matches_ * 3, num_images_ * 4, CV_64F);
	
	//LOGLN("---11111111----, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	double val;
	const double step = 2e-3;
	//t = getTickCount();
    //#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_images_; ++i)
	{

		for (int j = 0; j < 4; ++j)
		{
			val = cam_params_.at<double>(i * 4 + j, 0);
			cam_params_.at<double>(i * 4 + j, 0) = val - step;
			calcError(err1_);
			cam_params_.at<double>(i * 4 + j, 0) = val + step;
			calcError(err2_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(i * 4 + j));
			cam_params_.at<double>(i * 4 + j, 0) = val;
		}
	}
	//LOGLN("---2222222222222222----, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}

bool BundleAdjusterBase::estimate(const std::vector<ImageFeatures> &features,
	const std::vector<MatchesInfo> &pairwise_matches,
	std::vector<CameraParams> &cameras)
{
	LOGLN("Bundle adjustment xxxxxxxxx");
	num_images_ = static_cast<int>(features.size());
	features_ = &features[0];
	pairwise_matches_ = &pairwise_matches[0];

	setUpInitialCameraParams(cameras);

	// Leave only consistent image pairs
	edges_.clear();
	for (int i = 0; i < num_images_ - 1; ++i)
	{
		for (int j = i + 1; j < num_images_; ++j)
		{
			const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];
			if (matches_info.confidence > conf_thresh_)
				edges_.push_back(std::make_pair(i, j));
		}
	}
	// Compute number of correspondences
	total_num_matches_ = 0;
	for (size_t i = 0; i < edges_.size(); ++i)
		total_num_matches_ += static_cast<int>(pairwise_matches[edges_[i].first * num_images_ +
		edges_[i].second].num_inliers);
	/*LOGLN("num_images_" << num_images_ << "num_params_per_cam_" << num_params_per_cam_ << "total_num_matches_" 
		<< total_num_matches_ << "num_errs_per_measurement_" << num_errs_per_measurement_ << "term_criteria_" << term_criteria_.maxCount);
	exit(0);*/
	/*num_images_154 num_params_per_cam_ 4 total_num_matches_ 79508 num_errs_per_measurement_ 3 term_criteria_ 1000*/
	/*num_images_ 90 num_params_per_cam_ 4 total_num_matches_ 50534 num_errs_per_measurement_ 3 term_criteria_ 1000*/
	BundleAdjusterBase::setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 1000, DBL_EPSILON));
	CvLevMarq solver(num_images_ * num_params_per_cam_,
		total_num_matches_ * num_errs_per_measurement_,
		term_criteria_);
	
	Mat err, jac;
	CvMat matParams = cam_params_;
	cvCopy(&matParams, solver.param);
	
	int iter = 0;
	double err_temp[10];
	for (int i = 0; i < 10; i++)
		err_temp[i] = 100;
	double *err_temp_pointer = err_temp;
	/*for (int i = 0; i < 10; i++)
		LOGLN("err_temp_pointer" << err_temp_pointer[i]);*/
	int ind = 0;
	for (;;)
	{
		//int64 t = getTickCount();
		const CvMat* _param = 0;
		CvMat* _jac = 0;
		CvMat* _err = 0;
		bool proceed = solver.update(_param, _jac, _err);
		//LOGLN("wwwwwwwwwwwwwwwwwwwwwwwwwwww, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		//t = cvGetTickCount();
		cvCopy(_param, &matParams);
		//LOGLN("xxxxxxxxxxxxxxxxx, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		//t = getTickCount();
		
		if (!proceed || !_err )
			break;
		
		if (_jac)
		{
			
			calcJacobian(jac);
			//t = getTickCount();
			CvMat tmp = jac;
			cvCopy(&tmp, _jac);
			/*jac.release();*/
			//LOGLN("ccccccccccccccc, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		}
		//LOGLN("bbbbbbbbbbbbbbbbbbb, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		//t = getTickCount();
		if (_err)
		{
			
			calcError(err);
			LOG(".");
			iter++;
			
			//err_temp[iter % 10] = sqrt(err.dot(err) / total_num_matches_);
			err_temp_pointer[ind] = sqrt(err.dot(err) / total_num_matches_);
			CvMat tmp = err;
			cvCopy(&tmp, _err);
			LOGLN("---------RMS error: " << err_temp_pointer[ind]<<"-ind-"<<ind);
			if ( err_temp_pointer[ind] < 10){
				//LOGLN("err_temp_pointer[ind]< 5" << err_temp_pointer[ind]);
				break;
			}
			/*if (iter > 10){
				LOGLN("err_temp_pointer[ind]" << err_temp_pointer[ind]);
				LOGLN("err_temp_pointer[ind - 9]" << err_temp_pointer[(ind - 9)<0 ? (ind + 1) : (ind - 9)]);
				LOGLN("ind" << (err_temp_pointer[ind] == err_temp_pointer[(ind - 9)<0 ? (ind + 1) : (ind - 9)]));
			}*/
			double temp_ep = fabs(err_temp_pointer[ind] - err_temp_pointer[(ind - 9) < 0 ? (ind + 1) : (ind - 9)]);
			if ((iter>10) && temp_ep<1){
				//LOGLN("temp_ep " << temp_ep);
				break;
			}
			/*if (iter > 10 && (err_temp_pointer[ind] == err_temp_pointer[(ind - 9)<0 ? (ind + 1) : (ind - 9)])){
				break;
			}*/
			if ((!(iter % 10)) && iter>9)
				ind =0;
			else 
				ind++;
			/*err.release();*/
			
		}
		//LOGLN("dddddddcacalcErrorcalcErrorcalcErrorlcErrorcalcErrorddddddddddddd, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		
			
		
			
	}
	LOGLN("");
	LOGLN("Bundle adjustment, final RMS error: " << std::sqrt(err.dot(err) / total_num_matches_));
	
	LOGLN("Bundle adjustment, iterations done: " << iter);

	// Check if all camera parameters are valid
	bool ok = true;
	for (int i = 0; i < cam_params_.rows; ++i)
	{
		if (cvIsNaN(cam_params_.at<double>(i, 0)))
		{
			LOGLN("cam_params_.at<double>(i, 0)" << cam_params_.at<double>(i, 0)<<"i"<<i);
			ok = false;
			break;
		}
	}
	if (!ok)
		return false;

	obtainRefinedCameraParams(cameras);

	// Normalize motion to center image
	Graph span_tree;
	std::vector<int> span_tree_centers;
	findMaxSpanningTree(num_images_, pairwise_matches, span_tree, span_tree_centers);
	Mat R_inv = cameras[span_tree_centers[0]].R.inv();
	for (int i = 0; i < num_images_; ++i)
		cameras[i].R = R_inv * cameras[i].R;

	
	return true;
}
