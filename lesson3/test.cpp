#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include <iomanip> 
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

using namespace cv;
using namespace cv::detail;
using namespace std;

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

int main(int argc, char** argv)
{

    vector<Mat> imgs;    //表示待拼接的图像矢量队列
    Mat img = imread("0.jpg");    //读取两幅图像，并存入队列
    imgs.push_back(img);
    img = imread("1.jpg");
    imgs.push_back(img);
    Ptr<Feature2D> finder;    //特征检测
    //finder = xfeatures2d::SIFT::create();
    finder = xfeatures2d::SURF::create();
    vector<ImageFeatures> features(2);

    /*(*finder)(imgs[0], features[0]);
    (*finder)(imgs[1], features[1]);*/

    computeImageFeatures(finder, imgs[0], features[0]);
    computeImageFeatures(finder, imgs[1], features[1]);
    features[0].img_idx = 0;
    features[1].img_idx = 1;
    LOGLN("Features in image #" << 0 + 1 << ": " << features[0].keypoints.size());
    LOGLN("Features in image #" << 1 + 1 << ": " << features[1].keypoints.size());
    vector<MatchesInfo> pairwise_matches;    //特征匹配
    BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //定义特征匹配器，2NN方法
    matcher(features, pairwise_matches);    //进行特征匹配

    HomographyBasedEstimator estimator;    //定义参数评估器
    vector<CameraParams> cameras;    //表示相机参数矢量队列
    estimator(features, pairwise_matches, cameras);    //相机参数评估

    cout << "相机参数的初次评估：" << endl;    //终端输出
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);    //数据类型转换
        cameras[i].R = R;
        cout << "第" << i + 1 << "个相机的内参数" << ":\n" << cameras[i].K() << endl;
        cout << "第" << i + 1 << "个相机的旋转参数" << ":\n" << R << endl;
        cout << "第" << i + 1 << "个相机的焦距" << ":\n" << cameras[i].focal << endl;
    }

    Ptr<detail::BundleAdjusterBase> adjuster;    //光束平差法，精确相机参数
    adjuster = new detail::BundleAdjusterReproj();    //重映射误差方法
    //adjuster = new detail::BundleAdjusterRay();    //射线发散误差方法

    adjuster->setConfThresh(1);    //设置匹配置信度，该值为1
    (*adjuster)(features, pairwise_matches, cameras);    //相机参数的精确评估

    cout << "相机参数的精确评估：" << endl;    //终端输出
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cout << "第" << i + 1 << "个相机的内参数" << ":\n" << cameras[i].K() << endl;
        cout << "第" << i + 1 << "个相机的旋转参数" << ":\n" << cameras[i].R << endl;
    }

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());    //复制相机的旋转参数
    waveCorrect(rmats, WAVE_CORRECT_HORIZ);    //进行波形校正
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];    //赋值

    cout << "波形校正：" << endl;    //终端输出
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cout << "第" << i + 1 << "个相机的旋转参数" << ":\n" << cameras[i].R << endl;
    }
    return 0;
}