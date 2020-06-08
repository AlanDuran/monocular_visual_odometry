/*
 * point_mgmt.h
 *
 *  Created on: 3 feb 2020
 *      Author: alan
 */

#ifndef SRC_POINT_MGMT_H_
#define SRC_POINT_MGMT_H_

#include <iostream>
#include <numeric>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

double diamond_angle(double x, double y);
double shortest_distance(Vec3f line, Point2f point);

void check_klt(vector<Point2f> *pts, vector<Point2f> *new_pts, vector<uchar> &status,
	double max_angle = 20)
{
	vector<Point2f> &points = *pts; //Create a reference
	vector<Point2f> &new_points = *new_pts; //Create a reference

	int removed = 0;
	vector<double> dirs;
	vector<double> dirsf;
	uint points_size = new_points.size();
	max_angle /= 180; //conversion to half "diamond angle" (1 == 90°)

	for(uint i = 0; i < points_size; i++)
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

		else
		{
			Point2f pt2 = points.at(j);
			double dx = pt.x - pt2.x;
			double dy = pt.y - pt2.y;
			double dist = dx*dx + dy*dy;

			if (dist > 100e-3) {
				dirsf.push_back(diamond_angle(dx, dy));
				dirs.push_back(diamond_angle(dx, dy));
			}
			else
				dirs.push_back(std::numeric_limits<float>::infinity());
		}
	}

	points_size = new_points.size();
	double average = std::accumulate( dirsf.begin(), dirsf.end(), 0.0) / dirsf.size();
	removed = 0;

	for(uint i = 0; i < points_size; i++)
	{
		int j = i - removed;
		if(dirs[i] > average + max_angle || dirs[i] < average - max_angle)
		{
			points.erase(points.begin() + j);
			new_points.erase(new_points.begin() + j);
			removed++;
		}
	}
}

void check_klt(vector<Point2f> *pts, vector<Point2f> *new_pts,
		vector<uchar> &status, Mat feat_mask)
{
	vector<Point2f> &points = *pts; //Create a reference
	vector<Point2f> &new_points = *new_pts; //Create a reference
	int removed = 0;
	uint points_size = new_points.size();

	for(uint i = 0; i < points_size; i++)
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

void update_feature_positions(vector<Point2f> *last_pts, vector<Point2f> *new_pts, 
	uint max_features, uint max_distance = 10)
{
	vector<Point2f> &last_points = *last_pts; //Create a reference
	vector<Point2f> &new_points = *new_pts; //Create a reference
	vector<Point2f> points;

	uint curr_size = last_points.size();
	uint curr_removed = 0;

	if (curr_size > 0)
	{
		for (uint i = 0; i < curr_size; i++)
		{
			uint curr_index = i - curr_removed;
			uint flag = 1;
			uint j;

			for (j = 0; j < new_points.size(); j++) {
				double high_x = last_points[curr_index].x + max_distance;
				double low_x = last_points[curr_index].x - max_distance;
				double high_y = last_points[curr_index].y + max_distance;
				double low_y = last_points[curr_index].y - max_distance;
				double x = new_points[j].x;
				double y = new_points[j].y;

				if (((x <= high_x) && (x >= low_x) && (y <= high_y) && (y >= low_y))) {
					flag = 0;
					break;
				}
			}

			if (flag) {
				last_points.erase(last_points.begin() + curr_index);
				curr_removed++;
			}
			else {
				points.push_back(new_points[j]);
				new_points.erase(new_points.begin() + j);
			}
		}

		new_points.insert(new_points.begin(), points.begin(), points.end());
	}
}

// from https://stackoverflow.com/questions/1427422/cheap-algorithm-to-find-measure-of-angle-between-vectors
double diamond_angle(double x, double y)
{
	double angle;

	if (abs(x - y) < std::numeric_limits<float>::epsilon())
		angle = 0;
	else if (y >= 0)
        angle = (x >= 0 ? y/(x+y) : 1-x/(-x+y));
    else
        angle = (x < 0 ? 2-y/(-x-y) : 3+x/(x-y));

    if(angle > 2.0)
    	angle -= 4;

    return angle;
}

void check_epilines(vector<Point2f> *pts, vector<Point2f> *new_pts,
		vector<Vec3f> *epilines, double max_distance)
{
	vector<Point2f> &points = *pts; //Create a reference
	vector<Point2f> &new_points = *new_pts; //Create a reference
	vector<Vec3f> &epilines2 = *epilines; //Create a reference

	int removed = 0;
	uint points_size = new_points.size();

	for(uint i = 0; i < points_size; i++)
	{
		int j = i - removed;

		Point2f pt = new_points.at(j);
		double dist = shortest_distance(epilines2[j], pt);

		// Select good points
		if(dist > max_distance)
		{
			points.erase(points.begin() + j);
			new_points.erase(new_points.begin() + j);
			epilines2.erase(epilines2.begin() + j);
			removed++;
		}
	}
}

double shortest_distance(Vec3f line, Point2f point)
{
	//Distance = (| a*x1 + b*y1 + c |) / (sqrt( a*a + b*b))
	double num = line[0]*point.x + line[1]*point.y + line[2];
	double den = sqrt(line[0]*line[0] + line[1]*line[1]);
	return fabs(num) / den;
}

#endif /* SRC_POINT_MGMT_H_ */
