/*
 * point_mgmt.h
 *
 *  Created on: 3 feb 2020
 *      Author: alan
 */

#ifndef SRC_POINT_MGMT_H_
#define SRC_POINT_MGMT_H_

#include <iostream>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void check_klt(vector<Point2f> &points, vector<Point2f> &new_points, int points_size,
		vector<uchar> &status)
{
	int removed = 0;

	for(int i = 0; i < points_size; i++)
	{
		int j = i - removed;

		Point2f pt = new_points.at(j);
		// Select good points
		if((pt.x < 0 || pt.y < 0) || status[i] == 0)
		{
			points.erase(points.begin() + j);
			new_points.erase(new_points.begin() + j);
			removed++;
		}
	}
}

void check_klt(vector<Point2f> &points, vector<Point2f> &new_points, int points_size,
		vector<uchar> &status, Mat feat_mask)
{
	int removed = 0;

	for(int i = 0; i < points_size; i++)
	{
		int j = i - removed;

		Point2f pt = new_points.at(j);
		// Select good points
		if((pt.x < 0 || pt.y < 0) || status[i] == 0 || !feat_mask.at<uint8_t>(pt))
		{
			points.erase(points.begin() + j);
			new_points.erase(new_points.begin() + j);
			removed++;
		}
	}
}


#endif /* SRC_POINT_MGMT_H_ */
