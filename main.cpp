/*
 * main.cpp
 *
 *  Created on: 31 ene 2020
 *      Author: alan
 */

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>

#include "distortion_utils.h"
#include "graph_utils.h"
#include "point_mgmt.h"
#include "rot2euler.h"

#define USE_VIDEO 0
#define INITIAL_FRAME 0
#define N_FEATURES 50

using namespace cv;
using namespace std;

string img_lst_path = "../data/photo/config/r180.xml";
string video_path = "../data/rot/static.mp4";
string camera_matrix_path = "../data/cal/ip7_photos_out.xml";
int framecount = 0;
vector<string> image_list;
VideoCapture capture;

static Mat get_next_img();
static bool readStringList( const string& filename, vector<string>& l );


int main(int argc, char **argv)
{
	/********************** open video *************************************************/
	#if USE_VIDEO
		string filename = argc > 1 ? argv[1] : video_path;
		capture.VideoCapture(filename);
		if (!capture.isOpened()) //error in opening the video input
		{
			cerr << "Unable to open file!" << endl;
			return 0;
		}
		capture.set(CAP_PROP_POS_FRAMES ,capture.get(CAP_PROP_POS_FRAMES) + INITIAL_FRAME);
	#else
		readStringList(img_lst_path, image_list);
	#endif

    /********************** Read camera matrix *****************************************/
	const string input_settings_file = argc > 2 ? argv[2] : camera_matrix_path;
	FileStorage fs(input_settings_file, FileStorage::READ); // Read the settings

	if (!fs.isOpened())
	{
		cout << "Could not open camera matrix file: \"" << input_settings_file << "\"" << endl;
		return -1;
	}

	Mat camera_matrix, dist_coeffs;
	fs["camera_matrix"] >> camera_matrix;
	fs["distortion_coefficients"] >> dist_coeffs;

	/************************ Initializations *****************************************/
	// Create some random colors
	vector<Scalar> colors;
	RNG rng;
	for(int i = 0; i < N_FEATURES; i++)
	{
		int r = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int b = rng.uniform(0, 256);
		colors.push_back(Scalar(r,g,b));
	}

	Mat frames[2], cap_frame;
	vector<Point2f> points[2];

	// Take first frame
	Mat features_img, tracking_img;
	cap_frame = get_next_img();

	if(cap_frame.empty())
	{
	  printf( "No image data \n" );
	  return -1;
	}

	cap_frame.copyTo(features_img);
	cvtColor(cap_frame, frames[0], COLOR_BGR2GRAY);
	// Create a mask image for drawing purposes
	Mat drawing_mask, feat_mask;
	Size frame_size(cap_frame.cols, cap_frame.rows);

	//other initializations
	int index = 0;
	int m_frames = 0;
	Mat R_f = Mat::eye(3, 3, CV_64F);
	double y_1 = 1000, acc_1 = 1000;

	while(true)
	{
    	/********************************** define features to track ***********************************/

		/* Parameters:
		- image	Input 8-bit or floating-point 32-bit, single-channel image.
		- corners	Output vector of detected corners.
		- maxCorners	Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.
		- qualityLevel	Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal ) or the Harris function response (see cornerHarris ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
		- minDistance	Minimum possible Euclidean distance between the returned corners.
		- mask	Optional region of interest. If the image is not empty (it needs to have the type CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
		- blockSize	Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See cornerEigenValsAndVecs .
		- useHarrisDetector	Parameter indicating whether to use a Harris detector (see cornerHarris) or cornerMinEigenVal.
		- k	Free parameter of the Harris detector. */
    	goodFeaturesToTrack(frames[index], points[index], N_FEATURES, 0.3, 7, feat_mask, 15, false, 0.04);
    	draw_features(features_img, points[index], points[index].size(), colors, 15);

		if(points[index].size() < 8)
		{
			cap_frame = get_next_img();
			if (cap_frame.empty())
				break;
			frames[index].copyTo(features_img);
			cvtColor(cap_frame, frames[index], COLOR_BGR2GRAY);

			m_frames++;
			printf("\nNot enough good features to track, missed frame %d\n", framecount);
			continue;
		}

		cap_frame = get_next_img();
		if(cap_frame.empty())
			break;

		/************** calculate optical flow with lucas-kanade *************************************/
		int new_index = (index + 1) % 2;
		cap_frame.copyTo(features_img);
		cap_frame.copyTo(tracking_img);
		cvtColor(cap_frame, frames[new_index], COLOR_BGR2GRAY);

		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 30, 0.01);
		//OPTFLOW_LK_GET_MIN_EIGENVALS
		calcOpticalFlowPyrLK(frames[index], frames[new_index], points[index], points[new_index],
				status, err, Size(10,10), 7, criteria, 0, 1e-2);

		//getting rid of points for which the KLT tracking failed or those who have gone outside the frame
		check_klt(points[index], points[new_index], points[new_index].size(), status);
		draw_tracking(tracking_img, points[index], points[new_index], points[index].size(), colors, 15, 5);

		printf("\n%d\n", (int)points[new_index].size());
		if(points[new_index].size() > 5)
		{
			Mat R, T, mask;
			Mat E = findEssentialMat(points[index], points[new_index], camera_matrix, RANSAC, 0.999, 0.1, noArray());
			recoverPose(E, points[index], points[new_index], camera_matrix, R, T, mask);

			//Angle conversion and selection based on sign change
			vector<Vec3f> angles, acc_angles;
			Vec3f my_angle, my_acc;

			angles = rotationMatrixToEulerAngles(R);
			int sel_1 = select_rotation(angles, y_1, 1);
			my_angle = angles.at(sel_1);
			my_angle  *= 180.0/PI;

			R_f = R*R_f;
			acc_angles = rotationMatrixToEulerAngles(R_f);
			int sel_2 = select_rotation(acc_angles, acc_1, 1);
			my_acc = acc_angles.at(sel_2);
			my_acc *= 180.0/PI;

			printf("\nEstimated current R:\t\t\tAccumulated estimation of R:\n");
			print_rot_matrix(R, R_f);
			printf("\nAngle:\n");
			print_vector(my_angle, my_acc);
			printf("\nAlternatitve angle:\n");
			angles.at((~sel_1)&0x1) *= 180.0/PI;
			acc_angles.at((~sel_2)&0x1) *= 180.0/PI;
			print_vector(angles.at((~sel_1)&0x1), acc_angles.at((~sel_2)&0x1));
		}
		else
		{
			m_frames++;
			printf("\nNot enough tracked features to recover essential matrix, missed frame %d\n", framecount);
		}

		index = new_index;
		waitKey(0);
	}

	waitKey(0);
}

static Mat get_next_img()
{
	Mat result;
	#if USE_VIDEO
		capture >> result;
	#else
		result = imread(image_list[framecount]);
	#endif

	framecount++;
	return result;
}

static bool readStringList( const string& filename, vector<string>& l )
    {
        l.clear();
        FileStorage fs(filename, FileStorage::READ);
        if( !fs.isOpened() )
            return false;
        FileNode n = fs.getFirstTopLevelNode();
        if( n.type() != FileNode::SEQ )
            return false;
        FileNodeIterator it = n.begin(), it_end = n.end();
        for( ; it != it_end; ++it )
            l.push_back((string)*it);
        return true;
    }

