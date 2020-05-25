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

Vec3f rotationMatrixToEulerAngles(Mat &R)
{
    assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;

    y = atan2(-R.at<double>(2, 0), sy);

    if (!singular){
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        z = 0;
    }

    return Vec3f(x, y, z); //check_tangent(R, sy, x, y, z);
}


#endif /* ROT2EULER_H_ */
