/*
 * main.cpp
 *
 *  Created on: 31 ene 2020
 *      Author: alan
 */

#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>

#include "distortion_utils.h"
#include "graph_utils.h"
#include "point_mgmt.h"
#include "rot2euler.h"

// I/O parameters
#define USE_VIDEO 			1
#define INITIAL_FRAME 		20
#define STEP				1
#define SHOW_PLOTS			0
#define DRAW_SCALE			0.75
#define LINE_SIZE			1
#define CIRCLE_SIZE			5

// ALG parameters
#define MAX_DISTANCE 		5
#define MAX_DIR_ANGLE 		10
#define MAX_SKIPPED_FRAMES	5
#define N_FEATURES 			100

using namespace cv;
using namespace std;

string file_to_write = "../data/jetson/reports/pitch_wind.txt";
string error_report = "../data/jetson/reports/error.txt";
string img_lst_path = "../data/jetson/photo/roll2.xml";
string video_path = "../data/jetson/room/pitch.avi";
string camera_matrix_path = "../data/cal/jetson6_out.xml"; //jetson6_out, ip7_out, ip7_photos_out
int framecount = 0;
vector<string> image_list;
VideoCapture capture(video_path);

static Mat get_next_img();
static bool readStringList( const string& filename, vector<string>& l );


int main(int argc, char** argv)
{
	/********************** open video *************************************************/
#if USE_VIDEO
	//string filename = argc > 1 ? argv[1] : video_path;

	if (!capture.isOpened()) //error in opening the video input
	{
		cerr << "Unable to open file!" << endl;
		return 0;
	}
	capture.set(CAP_PROP_POS_FRAMES, capture.get(CAP_PROP_POS_FRAMES) + INITIAL_FRAME);
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
	for (int i = 0; i < N_FEATURES; i++)
	{
		int r = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int b = rng.uniform(0, 256);
		colors.push_back(Scalar(r, g, b));
	}

	Mat frames[2], cap_frame;
	vector<Point2f> points[2];

	// Take first frame
	Mat features_img, tracking_img, epilines_img, epilines_filtered_img;
	cap_frame = get_next_img();

	if (cap_frame.empty())
	{
		printf("No image data \n");
		return -1;
	}

	undistort_n_grayscale(cap_frame, frames[0], features_img, camera_matrix, dist_coeffs);
	// Create a mask image for drawing purposes
	Mat feat_mask;
	Size frame_size(cap_frame.cols, cap_frame.rows);
	feat_mask = create_mask(frame_size, camera_matrix, dist_coeffs);

	//other initializations
	int index = 0;
	int skp_frames_klt = 0, skp_frames_pose = 0, skp_frames_ep=0;
	int m_frames_klt = 0, m_frames_pose = 0, m_frames_gft = 0, m_frames_ep=0;
	double max = 0, min = 0;
	Mat R_f = Mat::eye(3, 3, CV_64F);

	while (true)
	{
		/********************************** define features to track *******************************************/

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
		int new_index = (index + 1) % 2;
		vector<Point2f> new_points;
		goodFeaturesToTrack(frames[index], new_points, N_FEATURES, 0.3, 7, feat_mask, 15, false, 0.04);
		update_feature_positions(&points[new_index], &new_points, N_FEATURES, 15);
		points[index] = points[new_index];
		printf("\n-------------------------------------------\ngft = %d, ", (int)points[index].size());

#if SHOW_PLOTS
		draw_features(features_img, points[index], points[index].size(), colors, CIRCLE_SIZE, DRAW_SCALE);
#endif

		if (points[index].size() < 8)
		{
			m_frames_gft++;
			points[new_index].clear();
			cap_frame = get_next_img();
			if (cap_frame.empty())
				break;

			undistort_n_grayscale(cap_frame, frames[index], features_img, camera_matrix, dist_coeffs);
			printf("\nNot enough good features to track, missed frame %d\n", framecount);
			continue;
		}
		//capture.set(CAP_PROP_POS_FRAMES , capture.get(CAP_PROP_POS_FRAMES) + STEP);
		cap_frame = get_next_img();
		if (cap_frame.empty())
			break;

		/************** calculate optical flow with lucas-kanade ****************************************************/

		undistort_n_grayscale(cap_frame, frames[new_index], features_img, camera_matrix, dist_coeffs);
		features_img.copyTo(tracking_img);
		features_img.copyTo(epilines_img);
		features_img.copyTo(epilines_filtered_img);
		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 30, 0.01);
		//OPTFLOW_LK_GET_MIN_EIGENVALS
		calcOpticalFlowPyrLK(frames[index], frames[new_index], points[index], points[new_index],
			status, err, Size(10, 10), 3, criteria, 0, 1e-3);

		//getting rid of points for which the KLT tracking failed or those who have gone outside the frame
		check_klt(&points[index], &points[new_index], status, MAX_DIR_ANGLE);

#if SHOW_PLOTS
		draw_tracking(tracking_img, points[index], points[new_index], points[index].size(), colors,
			CIRCLE_SIZE, LINE_SIZE, DRAW_SCALE);
#endif

		/************** filter some points with epipolar lines *******************************************************/

		printf("klt = %d, ", (int)points[new_index].size());
    	if(points[index].size() < 8)
		{
			skp_frames_klt++;
			if (skp_frames_klt + skp_frames_pose + skp_frames_ep > MAX_SKIPPED_FRAMES) {
				m_frames_klt += skp_frames_klt;
				m_frames_pose += skp_frames_pose;
				m_frames_ep += skp_frames_ep;
				skp_frames_klt = 0; skp_frames_ep = 0; skp_frames_pose = 0;
				index = new_index;
				printf("Exceded max number of skipped frames, missing last %d\n", MAX_SKIPPED_FRAMES);
			}
			printf("\nNot enough tracked features with LK, skipped klt %d, current frame %d\n", skp_frames_klt, framecount);
			continue;
		}

    	// Example. Estimation of fundamental matrix using the RANSAC algorithm
		vector<Vec3f> epilines2;
		Mat F = findFundamentalMat(points[index], points[new_index], FM_8POINT);
		computeCorrespondEpilines(points[new_index], 2, F, epilines2);

#if SHOW_PLOTS
		draw_epilines(epilines_img, &points[new_index], &epilines2, points[new_index].size(),
				colors, CIRCLE_SIZE, LINE_SIZE, DRAW_SCALE);
#endif

		check_epilines(&points[index], &points[new_index], &epilines2, MAX_DISTANCE);

#if SHOW_PLOTS
		draw_epilines(epilines_filtered_img, &points[new_index], &epilines2, points[new_index].size(),
				colors, CIRCLE_SIZE, LINE_SIZE, DRAW_SCALE, "Epilines filtered with distance");
#endif
    	/******************** recover pose ***************************************************************************/

		bool keep_frame = false;
		printf("epipolar = %d\n", (int)points[new_index].size());
		if(points[new_index].size() > 5)
		{
			Mat R, T, mask;
			Mat E = findEssentialMat(points[index], points[new_index], camera_matrix, RANSAC, 0.999, 0.1, noArray());
			recoverPose(E, points[index], points[new_index], camera_matrix, R, T, mask);

			//Angle conversion and selection based on sign change
			Vec3f angles, acc_angles;

			angles = rotationMatrixToEulerAngles(R) * (180/PI);

			if (fabs(angles[0]) < 10 && fabs(angles[1]) < 10 && fabs(angles[2]) < 10)
			{
				R_f = R * R_f;
				skp_frames_klt = 0; skp_frames_pose = 0;
			}
			else
			{
				skp_frames_pose++;
				keep_frame = true;
				printf("\nNoisy result. skipped pose = %d, current frame = %d\n", skp_frames_pose, framecount);
				fstream log;
				log.open(error_report, fstream::app);
				log << "Filtered frame " << framecount << "\n";
				log.close();
			}

			acc_angles = rotationMatrixToEulerAngles(R_f) * (180/PI);

			if(acc_angles.val[0] < min && framecount)
				min = acc_angles.val[0];
			if(acc_angles.val[0] > max && framecount < 375)
				max = acc_angles.val[0];

			printf("\nEstimated current R:\t\t\tAccumulated estimation of R:\n");
			print_rot_matrix(R, R_f);
			printf("\nAngle: (x = pitch, y = yaw, z = roll)\n");
			print_vector(angles, acc_angles);
		}
		else
		{
			skp_frames_ep++;
			keep_frame = true;
			printf("\nNot enough features to recover essential matrix, skipped ep %d, curr frame %d\n", skp_frames_ep, framecount);
		}

		if (skp_frames_klt + skp_frames_pose + skp_frames_ep>= MAX_SKIPPED_FRAMES) {
			m_frames_klt += skp_frames_klt; 
			m_frames_pose += skp_frames_pose;
			m_frames_ep += skp_frames_ep;
			skp_frames_klt = 0; skp_frames_pose = 0; skp_frames_ep = 0;
			keep_frame = false;
			printf("Exceded max number of skipped frames, missing last %d\n", MAX_SKIPPED_FRAMES);
		}

#if SHOW_PLOTS
		char key = char(waitKey(0));
#endif

		if (keep_frame)
			points[new_index].clear();
		else
			index = new_index;
	}

	int m_frames = m_frames_gft + m_frames_klt + m_frames_pose + m_frames_ep;
	double perc = 100 * m_frames / framecount;
	printf("m_frames = %d, klt_frames = %d, pose_frames = %d\npmissed = %f, n_frames = %d\n", m_frames, 
		m_frames_klt, m_frames_pose, perc, framecount);
	printf("min = %f, max = %f\n", min, max);

	fstream log;
	log.open(file_to_write, fstream::app);
	log << "\n----------------------------------\nMAX_DISTANCE\t\t" << MAX_DISTANCE << "\n";
	log << "MAX_DIR_ANGLE\t\t" << MAX_DIR_ANGLE << "\n";
	log << "N_FEATURES\t\t" << N_FEATURES << "\n";
	log << "STEP\t\t\t" << STEP << "\n\n";
	Vec3f angle = rotationMatrixToEulerAngles(R_f) * (180/PI);
	log << angle.val[0] << " " << angle.val[1] << " " << angle.val[2] << "\n";
	log << "gft_frames = " << m_frames_gft << "\tklt_frames = " << m_frames_klt;
	log << "\npose_frames = " << m_frames_pose << "\tep_frames = " << m_frames_ep;
	log << "\ntotal_missed = "<<  m_frames << "\tperc =" << perc << "%\tn_frames = " << framecount << "\n";
	log << "min = " << min << "\tmax = " << max << "\n";
	log.close();

#if SHOW_PLOTS
		waitKey(0);
#endif
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

