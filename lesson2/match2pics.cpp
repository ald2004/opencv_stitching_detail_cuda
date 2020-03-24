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
    Mat img = imread("1.jpg");    //读取两幅图像，并存入队列
    imgs.push_back(img);
    img = imread("2.jpg");
    imgs.push_back(img);
    Ptr<Feature2D> finder;    //特征检测
    finder = xfeatures2d::SIFT::create();
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

    Mat dispimg;    //两幅图像合并成一幅图像显示
    dispimg.create(Size(imgs[0].cols + imgs[1].cols, max(imgs[1].rows, imgs[1].rows)), CV_8UC3);
    Mat imgROI = dispimg(Rect(0, 0, (int)(imgs[0].cols), (int)(imgs[0].rows)));
    resize(imgs[0], imgROI, Size((int)(imgs[0].cols), (int)(imgs[0].rows)));
    imgROI = dispimg(Rect((int)(imgs[0].cols), 0, (int)(imgs[1].cols), (int)(imgs[1].rows)));
    resize(imgs[1], imgROI, Size((int)(imgs[1].cols), (int)(imgs[1].rows)));

    Point2f p1, p2;    //分别表示两幅图像内的匹配点对
    for (size_t i = 0; i < pairwise_matches[1].matches.size(); ++i)    //遍历匹配点对
    {
        if (!pairwise_matches[1].inliers_mask[i])    //不是内点，则继续下一次循环
            continue;

        const DMatch& m = pairwise_matches[1].matches[i];    //得到内点的匹配点对
        p1 = features[0].keypoints[m.queryIdx].pt;
        p2 = features[1].keypoints[m.trainIdx].pt;
        p2.x += features[0].img_size.width;    //p2在合并图像上的坐标

        line(dispimg, p1, p2, Scalar::all(255));    //画直线
    }

    //在终端显示内点数量和单应矩阵
    cout << "内点数量：" << endl;
    cout << setw(10) << pairwise_matches[1].matches.size() << endl << endl;

    const double* h = reinterpret_cast<const double*>(pairwise_matches[1].H.data);
    cout << "单应矩阵：" << endl;
    cout << setw(10) << (int)(h[0] + 0.5) << setw(6) << (int)(h[1] + 0.5) << setw(6) << (int)(h[2] + 0.5) << endl;
    cout << setw(10) << (int)(h[3] + 0.5) << setw(6) << (int)(h[4] + 0.5) << setw(6) << (int)(h[5] + 0.5) << endl;
    cout << setw(10) << (int)(h[6] + 0.5) << setw(6) << (int)(h[7] + 0.5) << setw(6) << (int)(h[8] + 0.5) << endl;

    imshow("匹配显示", dispimg);    //显示匹配图像
    imwrite("test.jpg", dispimg);
    waitKey(0);

    return 0;
}