/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// Copyright (C) 2013, Alfonso Sanchez-Beato, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "pch.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp> // OpenCV window I/O
#include <opencv2/imgproc.hpp> // OpenCV image transformations


#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>
using namespace cv::xfeatures2d;

#include "opencv2/reg/mapaffine.hpp"
#include "opencv2/reg/mapshift.hpp"
#include "opencv2/reg/mapprojec.hpp"
#include "opencv2/reg/mappergradshift.hpp"
#include "opencv2/reg/mappergradeuclid.hpp"
#include "opencv2/reg/mappergradsimilar.hpp"
#include "opencv2/reg/mappergradaffine.hpp"
#include "opencv2/reg/mappergradproj.hpp"
#include "opencv2/reg/mapperpyramid.hpp"

static const char* DIFF_IM = "Image difference";
static const char* DIFF_REGPIX_IM = "Image difference: pixel registered";

using namespace cv;
using namespace cv::reg;
using namespace std;

static void showDifference(const Mat& image1, const Mat& image2, const char* title)
{
	Mat img1, img2;
	image1.convertTo(img1, CV_32FC3);
	image2.convertTo(img2, CV_32FC3);
	if (img1.channels() != 1)
		cvtColor(img1, img1, COLOR_BGR2GRAY);
	if (img2.channels() != 1)
		cvtColor(img2, img2, COLOR_BGR2GRAY);

	Mat imgDiff;
	img1.copyTo(imgDiff);
	imgDiff -= img2;
	imgDiff /= 2.f;
	imgDiff += 128.f;

	Mat imgSh;
	imgDiff.convertTo(imgSh, CV_8UC3);
	imshow(title, imgSh);
}


static void calcHomographyFeature(const Mat& img1, const Mat& img2)
{
	static const char* difffeat = "Difference feature registered";

	if (img1.empty() || img2.empty())
	{
		printf("Can't read one of the images\n");
		return;
	}

	// detecting keypoints
	cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> detector = cv::xfeatures2d::SurfFeatureDetector::create();
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);
	
	// computing descriptors
	cv::Ptr<cv::xfeatures2d::SurfDescriptorExtractor> extractor = cv::xfeatures2d::SurfDescriptorExtractor::create();
	cv::Mat descriptors1, descriptors2;
	extractor->compute(img1, keypoints1, descriptors1);
	extractor->compute(img2, keypoints2, descriptors2);

	// matching descriptors
	cv::BFMatcher matcher(cv::NORM_L2);
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);
 

	int N = matches.size() / 8;
	//int N = 20;
	nth_element(matches.begin(), matches.begin() + N - 1, matches.end());
	matches.erase(matches.begin() + N, matches.end());

	// drawing the results
	cv::namedWindow("matches", 1);
	cv::Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
	cv::imshow("matches", img_matches);

	std::vector<cv::KeyPoint> R_keypoint01, R_keypoint02;
	for (int i = 0; i < matches.size(); i++)
	{
		R_keypoint01.push_back(keypoints1[matches[i].queryIdx]);
		R_keypoint02.push_back(keypoints2[matches[i].trainIdx]);
	}

	std::vector<cv::Point2f>p01, p02;
	for (int i = 0; i < matches.size(); i++)
	{
		p01.push_back(R_keypoint01[i].pt);
		p02.push_back(R_keypoint02[i].pt);
	}


	std::vector<uchar> RansacStatus;
	cv::Mat Fundamental = findHomography(p02, p01, RansacStatus, cv::RANSAC);
	std::cout << Fundamental << std::endl;

	cv::Mat dst;
	warpPerspective(img2, dst, Fundamental, cv::Size(img2.cols, img2.rows));

	imshow("Feature based", dst);
	Mat imf1, resf;
	img1.convertTo(imf1, CV_64FC3);
	dst.convertTo(resf, CV_64FC3);
	showDifference(imf1, resf, difffeat);
}

static void calcHomographyPixel(const Mat& img1, const Mat& img2)
{
	static const char* diffpixel = "Difference pixel registered";

	// Register using pixel differences
	Ptr<MapperGradProj> mapper = makePtr<MapperGradProj>();
	MapperPyramid mappPyr(mapper);
	Ptr<Map> mapPtr = mappPyr.calculate(img1, img2);

	// Print result
	MapProjec* mapProj = dynamic_cast<MapProjec*>(mapPtr.get());
	mapProj->normalize();
	cout << "--- Pixel-based method\n" << Mat(mapProj->getProjTr()) << endl;

	// Display registration accuracy
	Mat dest, destShow;
	mapProj->inverseWarp(img2, dest);
	dest.convertTo(destShow, CV_8UC3);
	imshow("Pixel based", destShow);

	showDifference(img1, dest, diffpixel);
}

static void comparePixelVsFeature(const Mat& img1_8b, const Mat& img2_8b)
{
	static const char* difforig = "Difference non-registered";

	// Show difference of images
	Mat img1, img2;
	img1_8b.convertTo(img1, CV_64FC3);
	img2_8b.convertTo(img2, CV_64FC3);
	//showDifference(img1, img2, difforig);
	cout << endl << "--- Comparing feature-based with pixel difference based ---" << endl;

	// Register using SURF keypoints
	calcHomographyFeature(img1_8b, img2_8b);

	// Register using pixel differences
	calcHomographyPixel(img1, img2);

}

int main(void)
{
	const char* filename = "C:\\Users\\pyzhu\\Desktop\\test\\Image_c015.tif";
	Mat img1 = imread(filename, cv::IMREAD_UNCHANGED);
	if (!img1.data) {
		cout << "Could not open or find file" << endl;
		return -1;
	}

	Mat imgcmp1 = img1.clone();

	// Convert to double, 3 channels
	img1.convertTo(img1, CV_64FC3);

	filename = "C:\\Users\\pyzhu\\Desktop\\test\\img15Affine.tif";
	//filename = "C:\\Users\\pyzhu\\Desktop\\test\\testAffined.jpg";
	Mat imgcmp2 = imread(filename, IMREAD_UNCHANGED);
	if (!imgcmp2.data) {
		cout << "Could not open or find file" << endl;
		return -1;
	}
	comparePixelVsFeature(imgcmp1, imgcmp2);
	imshow("image1", imgcmp1);
	imshow("image2", imgcmp2);

	waitKey();
	return 0;
}
