/*
 * rot2euler.h
 *
 *  Created on: 3 feb 2020
 *      Author: alan
 */

#ifndef ROT2EULER_H_
#define ROT2EULER_H_

#include <iostream>
#include <math.h>
#include <opencv2/imgproc.hpp>

#define PI 3.14159265359

using namespace std;
using namespace cv;

// Calculates rotation matrix given euler angles.
Mat eulerAnglesToRotationMatrix(Vec3f &theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );

    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );

    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);

    // Combined rotation matrix
    Mat R = R_z * R_y * R_x;

    return R;
}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());

    //printf("%lf\n", norm(I, shouldBeIdentity));
    return  norm(I, shouldBeIdentity) < 8e-6;
}

bool closeEnough(const float& a, const float& b) {
	float epsilon = std::numeric_limits<float>::epsilon();
    return (epsilon > std::abs(a - b));
}

vector<Vec3f> rotationMatrixToEulerAngles(Mat &R)
{
	vector<Vec3f> result;
	assert(isRotationMatrix(R));
	//Rz(φ)Ry(θ)Rx(ψ)

	//check for gimbal lock
	if (closeEnough(R.at<double>(2, 0), -1.0f))
	{
		float x = 0; //gimbal lock, value of x doesn't matter
		float y = PI / 2;
		float z = x + atan2(R.at<double>(0, 1), R.at<double>(0, 2));
		return result.push_back(Vec3f(x, y, z));
	}
	else if (closeEnough(R.at<double>(2, 0), 1.0f))
	{
		float x = 0;
		float y = -PI / 2;
		float z = -x + atan2(-R.at<double>(0, 1), -R.at<double>(0, 2));
		return result.push_back(Vec3f(x, y, z));
	}
	else
	{ //two solutions exist
		float y1 = -asin(R.at<double>(2, 0));
		float y2 = PI - y1;

		float x1 = atan2(R.at<double>(2, 1) / cos(y1), R.at<double>(2, 2) / cos(y1));
		float x2 = atan2(R.at<double>(2, 1) / cos(y2), R.at<double>(2, 2) / cos(y2));

		float z1 = atan2(R.at<double>(1, 0) / cos(y1), R.at<double>(0, 0) / cos(y1));
		float z2 = atan2(R.at<double>(1, 0) / cos(y2), R.at<double>(0, 0) / cos(y2));

		//choose one solution to return
		//for example the "shortest" rotation
		if ((std::abs(x1) + std::abs(y1) + std::abs(z1)) <= (std::abs(x2) + std::abs(y2) + std::abs(z2)))
		{
			return Vec3f(x1, y1, z1);
		}
		else
		{
			return Vec3f(x2, y2, z2);
		}
	}
}




#endif /* ROT2EULER_H_ */
