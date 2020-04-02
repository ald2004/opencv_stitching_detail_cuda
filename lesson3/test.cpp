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
    Mat img = imread("0.jpg");    //��ȡ����ͼ�񣬲��������
    imgs.push_back(img);
    img = imread("1.jpg");
    imgs.push_back(img);
    Ptr<Feature2D> finder;    //�������
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
    vector<MatchesInfo> pairwise_matches;    //����ƥ��
    BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //��������ƥ������2NN����
    matcher(features, pairwise_matches);    //��������ƥ��

    HomographyBasedEstimator estimator;    //�������������
    vector<CameraParams> cameras;    //��ʾ�������ʸ������
    estimator(features, pairwise_matches, cameras);    //�����������

    cout << "��������ĳ���������" << endl;    //�ն����
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);    //��������ת��
        cameras[i].R = R;
        cout << "��" << i + 1 << "��������ڲ���" << ":\n" << cameras[i].K() << endl;
        cout << "��" << i + 1 << "���������ת����" << ":\n" << R << endl;
        cout << "��" << i + 1 << "������Ľ���" << ":\n" << cameras[i].focal << endl;
    }

    Ptr<detail::BundleAdjusterBase> adjuster;    //����ƽ�����ȷ�������
    adjuster = new detail::BundleAdjusterReproj();    //��ӳ������
    //adjuster = new detail::BundleAdjusterRay();    //���߷�ɢ����

    adjuster->setConfThresh(1);    //����ƥ�����Ŷȣ���ֵΪ1
    (*adjuster)(features, pairwise_matches, cameras);    //��������ľ�ȷ����

    cout << "��������ľ�ȷ������" << endl;    //�ն����
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cout << "��" << i + 1 << "��������ڲ���" << ":\n" << cameras[i].K() << endl;
        cout << "��" << i + 1 << "���������ת����" << ":\n" << cameras[i].R << endl;
    }

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());    //�����������ת����
    waveCorrect(rmats, WAVE_CORRECT_HORIZ);    //���в���У��
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];    //��ֵ

    cout << "����У����" << endl;    //�ն����
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cout << "��" << i + 1 << "���������ת����" << ":\n" << cameras[i].R << endl;
    }
    return 0;
}