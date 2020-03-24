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

    vector<Mat> imgs;    //��ʾ��ƴ�ӵ�ͼ��ʸ������
    Mat img = imread("1.jpg");    //��ȡ����ͼ�񣬲��������
    imgs.push_back(img);
    img = imread("2.jpg");
    imgs.push_back(img);
    Ptr<Feature2D> finder;    //�������
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
    vector<MatchesInfo> pairwise_matches;    //����ƥ��
    BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //��������ƥ������2NN����
    matcher(features, pairwise_matches);    //��������ƥ��

    Mat dispimg;    //����ͼ��ϲ���һ��ͼ����ʾ
    dispimg.create(Size(imgs[0].cols + imgs[1].cols, max(imgs[1].rows, imgs[1].rows)), CV_8UC3);
    Mat imgROI = dispimg(Rect(0, 0, (int)(imgs[0].cols), (int)(imgs[0].rows)));
    resize(imgs[0], imgROI, Size((int)(imgs[0].cols), (int)(imgs[0].rows)));
    imgROI = dispimg(Rect((int)(imgs[0].cols), 0, (int)(imgs[1].cols), (int)(imgs[1].rows)));
    resize(imgs[1], imgROI, Size((int)(imgs[1].cols), (int)(imgs[1].rows)));

    Point2f p1, p2;    //�ֱ��ʾ����ͼ���ڵ�ƥ����
    for (size_t i = 0; i < pairwise_matches[1].matches.size(); ++i)    //����ƥ����
    {
        if (!pairwise_matches[1].inliers_mask[i])    //�����ڵ㣬�������һ��ѭ��
            continue;

        const DMatch& m = pairwise_matches[1].matches[i];    //�õ��ڵ��ƥ����
        p1 = features[0].keypoints[m.queryIdx].pt;
        p2 = features[1].keypoints[m.trainIdx].pt;
        p2.x += features[0].img_size.width;    //p2�ںϲ�ͼ���ϵ�����

        line(dispimg, p1, p2, Scalar::all(255));    //��ֱ��
    }

    //���ն���ʾ�ڵ������͵�Ӧ����
    cout << "�ڵ�������" << endl;
    cout << setw(10) << pairwise_matches[1].matches.size() << endl << endl;

    const double* h = reinterpret_cast<const double*>(pairwise_matches[1].H.data);
    cout << "��Ӧ����" << endl;
    cout << setw(10) << (int)(h[0] + 0.5) << setw(6) << (int)(h[1] + 0.5) << setw(6) << (int)(h[2] + 0.5) << endl;
    cout << setw(10) << (int)(h[3] + 0.5) << setw(6) << (int)(h[4] + 0.5) << setw(6) << (int)(h[5] + 0.5) << endl;
    cout << setw(10) << (int)(h[6] + 0.5) << setw(6) << (int)(h[7] + 0.5) << setw(6) << (int)(h[8] + 0.5) << endl;

    imshow("ƥ����ʾ", dispimg);    //��ʾƥ��ͼ��
    imwrite("test.jpg", dispimg);
    waitKey(0);

    return 0;
}