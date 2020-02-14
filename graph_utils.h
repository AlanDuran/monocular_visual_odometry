/*
 * graph_utils.h
 *
 *  Created on: 31 ene 2020
 *      Author: alan
 */

#ifndef SRC_GRAPH_UTILS_H_
#define SRC_GRAPH_UTILS_H_

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void show_n_resize(Mat src, double scale, const String &title)
{
	Mat display;
	resize(src, display, Size(src.cols * scale, src.rows * scale));
	imshow(title, display);
	display.deallocate();
}

void draw_features(Mat src, vector<Point2f> &points, int points_size, vector<Scalar> &colors,
		int circle_size = 5)
{
	for(int i = 0; i < points_size; i++){
		circle(src, points[i], circle_size, colors[i], -1);
	}

	show_n_resize(src, 0.20, "Detected features in I(t)");
}

void draw_tracking(Mat src, vector<Point2f> &points, vector<Point2f> &tracked, int points_size,
		vector<Scalar> &colors, int circle_size = 5, int thickness = 1)
{
	for(int i = 0; i < points_size; i++){
		//circle(src, points[i], circle_size, colors[i], -1);
		line(src, tracked[i], points[i], colors[i], thickness, LINE_4);
		circle(src, tracked[i], circle_size, colors[i], -1);
	}

	show_n_resize(src, 0.20, "Tracked features in I(t+1)");
}

void print_rot_matrix(Mat R)
{
	printf("%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",
		R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
		R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
		R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2));
}

void print_rot_matrix(Mat R, Mat R2)
{
	printf("%lf %lf %lf\t\t%lf %lf %lf\n%lf %lf %lf\t\t%lf %lf %lf\n%lf %lf %lf\t\t%lf %lf %lf\n",
			R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), R2.at<double>(0,0), R2.at<double>(0,1), R2.at<double>(0,2),
			R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), R2.at<double>(1,0), R2.at<double>(1,1), R2.at<double>(1,2),
			R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), R2.at<double>(2,0), R2.at<double>(2,1), R2.at<double>(2,2));
}

void print_vector(Vec3f &vec)
{
	printf("%lf %lf %lf\n", vec.val[0], vec.val[1], vec.val[2]);
}

void print_vector(Vec3f &vec, Vec3f &vec2)
{
	printf("%lf %lf %lf\t\t%lf %lf %lf\n", vec.val[0], vec.val[1], vec.val[2],
			vec2.val[0], vec2.val[1], vec2.val[2]);
}
#endif /* SRC_GRAPH_UTILS_H_ */
