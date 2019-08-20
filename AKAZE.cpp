#include "pch.h"
#include <iostream>
#include <opencv2\opencv.hpp>

int AKAZE(const cv::Mat &img_ref, const cv::Mat &img_target, cv::Mat &img_reg, cv::Point2d &translation, float ratio_thresh = 0.65)  //return 0:succeed  1:not enough matched points
{
	std::vector<cv::KeyPoint> kpts1, kpts2;
	cv::Mat desc1, desc2;
	cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
	akaze->detectAndCompute(img_ref, cv::noArray(), kpts1, desc1);
	akaze->detectAndCompute(img_target, cv::noArray(), kpts2, desc2);

	cv::BFMatcher matcher(cv::NORM_HAMMING);
	std::vector<std::vector<cv::DMatch> > knn_matches;
	matcher.knnMatch(desc1, desc2, knn_matches, 2);

	std::vector<cv::DMatch> matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			matches.push_back(knn_matches[i][0]);
		}
	}

	std::vector<cv::KeyPoint> R_keypoint01, R_keypoint02;

	for (int i = 0; i < matches.size(); i++)
	{
		R_keypoint01.push_back(kpts1[matches[i].queryIdx]);  //img1
		R_keypoint02.push_back(kpts2[matches[i].trainIdx]);  //img2
	}

	std::vector<cv::Point2f>p01, p02;
	for (int i = 0; i < matches.size(); i++)
	{
		p01.push_back(R_keypoint01[i].pt);
		p02.push_back(R_keypoint02[i].pt);
	}

	//check if matches.size() too small
	if (matches.size() < 5)
		return 1;

	//find homography
	std::vector<uchar> RansacStatus;
	cv::Mat Fundamental = cv::estimateAffinePartial2D(p02, p01, RansacStatus, cv::RANSAC);

	warpAffine(img_target, img_reg, Fundamental, cv::Size(img_ref.cols, img_ref.rows));

	double tplt_x = img_ref.cols / 2;
	double tplt_y = img_ref.rows / 2;
	cv::Mat center = (cv::Mat_<double>(3, 1) << tplt_x, tplt_y, 1);

	cv::Mat inv_Fun;
	cv::invertAffineTransform(Fundamental, inv_Fun);

	cv::Mat cross = inv_Fun * center;

	translation.x = cross.at<double>(0, 0) - tplt_x;
	translation.y = cross.at<double>(1, 0) - tplt_y;

	return 0;
}

cv::Mat PaddingImg(cv::Mat img, cv::Size target_size)
{
	int w_diff = target_size.width - img.cols;
	int h_diff = target_size.height - img.rows;
	int w_padding = w_diff / 2;
	int h_padding = h_diff / 2;
	int w_pad_left = w_padding, h_pad_top = h_padding;
	cv::Mat img_padding;
	if (w_diff % 2 == 1)
	{
		w_pad_left++;
	}
	if (h_diff % 2 == 1)
	{
		h_pad_top++;
	}
	cv::copyMakeBorder(img.clone(), img_padding, h_pad_top, h_padding, w_pad_left, w_padding, cv::BORDER_CONSTANT);
	img = img_padding;
	return img;
}

int main()
{
	const char* filename = "C:\\Users\\pyzhu\\Desktop\\test\\registration\\zhen\\505546-8bit.png";
	cv::Mat img1 = imread(filename, cv::IMREAD_UNCHANGED);
	//imshow("img1", img1);


	filename = "C:\\Users\\pyzhu\\Desktop\\test\\registration\\zhen\\test.JPG";
	cv::Mat img2 = imread(filename, cv::IMREAD_UNCHANGED);
	//imshow("img2", img2);

	if (img1.size() != img2.size())
	{
		img1 = PaddingImg(img1, img2.size());
	}
	imshow("img1", img1);

	cv::Mat img_reg;
	cv::Point2d translation;

	int res = AKAZE(img1, img2, img_reg, translation);
	std::cout << res << std::endl;
	if (res == 0)
	{
		imshow("img_reg", img_reg);
		std::cout << translation << std::endl;
	}

	cv::waitKey(0);
}

