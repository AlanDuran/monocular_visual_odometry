/*
 * distortion_utils.h
 *
 *  Created on: 31 ene 2020
 *      Author: alan
 */

#ifndef SRC_DISTORTION_UTILS_H_
#define SRC_DISTORTION_UTILS_H_

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

Mat create_mask(Size size, Mat camera_matrix, Mat dist_coeff, float proportion=0.9)
{
	Mat temp_mask, mask = Mat::zeros(size, CV_8UC1);
	undistort(Mat::ones(size, CV_8UC1)*255, temp_mask, camera_matrix, dist_coeff);
	resize(temp_mask, temp_mask, cv::Size(), proportion, proportion);
	float val = (1 - proportion) / 2;
	temp_mask.copyTo(mask(Rect(mask.cols*val, mask.rows*val, temp_mask.cols, temp_mask.rows)));
	temp_mask.deallocate(); //free memory
	return mask;
}

void undistort_n_grayscale(Mat src, Mat &gray, Mat &color, Mat camera_matrix, Mat dist_coeffs)
{
	undistort(src, gray, camera_matrix, dist_coeffs);
	Mat features, track_frame;
	gray.copyTo(color);
	cvtColor(src, gray, COLOR_BGR2GRAY);
}

void undistort_n_grayscale(Mat src, Mat gray, Mat camera_matrix, Mat dist_coeffs)
{
	undistort(src, gray, camera_matrix, dist_coeffs);
	Mat features, track_frame;
	cvtColor(src, src, COLOR_BGR2GRAY);
}


#endif /* SRC_DISTORTION_UTILS_H_ */
