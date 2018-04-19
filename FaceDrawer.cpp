/* Chaz Acheronti */

// C/C++ libraries
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <chrono>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// The color we will use for all facial dots
cv::Scalar dot_color	{ 180.0, 244.0, 66.0 }; // cyan / green blue color
cv::Scalar line_color	{ 15.0, 100.0, 15.0}; // green

/** @brief Wrapper for drawing a point on an image.
	@param image Image where the point is drawn.
	@param point Point location given in CV point type (x, y)
*/
void DrawPoint(cv::Mat image, const cv::Point& point) {
	cv::circle(image, point, 5, dot_color, -1);
}

/** @brief Draw all facial points.
@param image Image where the line is drawn.
@param points vector<cv::Point>
*/
void DrawPoints(cv::Mat image, const std::vector<cv::Point>& points) {
	for (size_t i = 0; i < points.size(); i++)
		DrawPoint(image, points[i]);
}


/** @brief Wrapper for drawing a line from p1 to p2 on an image.
	@param image Image where the line is drawn.
	@param p1 Point location given in CV point type (x, y)
	@param p2 Point location given in CV point type (x, y)
*/
void DrawLine(cv::Mat image, const cv::Point& p1, const cv::Point& p2) {
	cv::line(image, p1, p2, line_color, 2, 4);
}



/** @brief given a vector of points, draw the delaunay triangulation
@param image cv::Mat of pixel colors to be drawn on
@param std::vector<cv::Point> Vector of facial cv::point
@param width Width of image
@param height Height of image
@param drawPoints Avoid O(2n) and draw points as they are read into the subdiv
*/
void DrawDelaunayTriangles(cv::Mat image, std::vector<cv::Point> points, int width, int height, bool drawPoints) {
	// initialze the rect space
	cv::Rect screen_space{ 0,0, width, height };

	// Create a sbudiv
	Subdiv2D subdiv(screen_space);

	// insert all facial points to the subdivision
	for (size_t i = 0; i < points.size(); i++) {
		if (drawPoints) DrawPoint(image, points[i]); // optimize to avoid iterating through points twice
		subdiv.insert(points[i]);
	}

	// now use opencv lib to get a list of all triangles
	vector<cv::Vec6f> triangles;
	subdiv.getTriangleList(triangles);

	// for every possible triangle
	for (size_t i = 0; i < triangles.size(); i++) {
		Vec6f triangle = triangles[i];
		Point pt1{ cvRound(triangle[0]), cvRound(triangle[1]) };
		Point pt2{ cvRound(triangle[2]), cvRound(triangle[3]) };
		Point pt3{ cvRound(triangle[4]), cvRound(triangle[5]) };

		// Draw rectangles completely inside the image.
		if (screen_space.contains(pt1) && screen_space.contains(pt2) && screen_space.contains(pt3))
		{
			DrawLine(image, pt1, pt2);
			DrawLine(image, pt2, pt3);
			DrawLine(image, pt3, pt1);
			//line(image, pt3, pt1, line_color, 1, CV_AA, 0);
		}
	}
}



/** @brief Parse string of facial points into a cv::point vector
	@param std::string String of facial points delimted by comma and space
	@return std::vector<cv::Point> Vector of cv::point
*/
std::vector<cv::Point> ParseFacialPoints(const std::string& facial_string) {
	// parse the string by leaving an index value at the start of a number
	// then move the end index until we hit a delimiter
	// once we find the delimiter, we take the substring from begin to end and convert to double

	size_t begin = 0, end = 0;
	int x, y;
	std::vector<cv::Point> facial_points;

	for (size_t i = 0; i < facial_string.size(); i++) {
		char curr_char = facial_string.at(i); // Parse the facial string by character
		if (!isdigit(curr_char) && curr_char != '.') {
			
			if (curr_char == ' ') { // save the value as x
				std::string value = facial_string.substr(begin, end - begin + 1);
				x = stoi(value);
				begin = end = i + 1; // reset begin and end index
			} 
			else if (curr_char == ',') { // save value as y and push as cv::point
				std::string value = facial_string.substr(begin, end - begin + 1);
				y = stoi(value);
				
				cv::Point new_point{ x, y }; // create and push the cv::point
				facial_points.push_back(new_point);

				begin = end = i + 1; // reset begin and end index
			} // no need to check for end of line since this is a string and not a c string (yay c++)

		}
		else end = i; // move the end index
	}
	return facial_points;
}

// Triangulate a single image file and overwrite it
void TriangulateImage(string directory_path, string image_file_path, int frame_id) {
	auto start_time = std::chrono::high_resolution_clock::now();

	cv::Mat source_image = imread(image_file_path, 1); // read the image from file path	

	// Get facial points string from database (for now fake data)
	std::string facial_str = "260.040343 888.611127,269.976639 986.237354,289.517197 1083.266163,318.881451 1173.145982,364.546544 1250.371343,418.218724 1309.539448,461.121964 1353.916568,504.180831 1384.225729,559.618409 1388.323818,621.603966 1370.890055,679.025618 1321.635214,733.523399 1257.153803,774.329893 1186.295180,799.438576 1104.623734,808.253363 1017.840100,812.229479 928.171773,811.221769 840.187872,284.726667 848.636778,314.568856 809.799045,363.772096 799.441036,415.291389 808.526863,460.919328 829.631910,583.462146 820.953046,636.755684 795.958497,691.394963 783.182439,743.891658 792.415951,775.381131 828.681755,529.630091 906.442344,529.574225 967.086822,530.012695 1026.679804,531.292755 1086.990969,467.638977 1106.164622,501.108780 1119.289102,536.497121 1129.913266,572.325308 1114.384652,604.607400 1099.128276,343.791704 920.670122,376.535971 906.083111,416.491883 907.205085,450.937132 925.887388,414.741360 935.063918,374.592095 935.758863,608.171787 919.124603,644.627460 896.407367,685.007219 893.673088,717.802710 903.722484,689.087399 921.081659,648.842417 925.134424,428.372733 1189.689410,468.256940 1173.033265,509.813908 1166.325831,544.873064 1173.577640,584.582054 1162.312021,632.018411 1162.765017,673.808561 1169.648399,637.463043 1224.757898,593.982270 1254.679665,550.961616 1263.246796,512.593977 1260.883046,468.722179 1238.307145,444.941818 1193.994500,511.391420 1191.338850,546.770070 1193.972674,587.503078 1186.534614,655.700551 1176.627786,590.680093 1215.187488,549.193046 1223.938076,512.354186 1220.627523";

	// Parse the string
	std::vector<cv::Point> facial_points = ParseFacialPoints(facial_str);

	if (!source_image.empty()) {
		// Draw the delaunay triangulation (and facial points)
		DrawDelaunayTriangles(source_image, facial_points, source_image.size().width, source_image.size().height, true);

		// Write the result 
		string result_image_path = directory_path + "/drawn_" + to_string(frame_id) + ".png";
		imwrite(result_image_path, source_image);

		std::cout << "Overwrote points for image " << frame_id << ".png at\n" << image_file_path << "  in ";
	}
	auto end_time = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed_time = end_time - start_time;

	std::cout << elapsed_time.count() << "s\n";
}


int main(int argc, char *argv[]) {
	string directory_path;
	cv::Mat source_image;

	int frame_id = 0, end_frame_id = -1;

	// processing command line arguments
	if (argc <= 2) {
		std::cout << "Usage: triangulation.exe \"path/to/video/dir\"  frame_id  [or]\ntriangulation.exe \"path/to/video/dir\"  start_frame_id  end_frame_id\n";
		return 0;
	}
	else if (argc == 3) {
		// get the file path
		directory_path = argv[1];
		// get the frame id
		frame_id = atoi(argv[2]);
	}
	else if (argc == 4) {
		// get the file path
		directory_path = argv[1];
		// get the start and end frame id
		frame_id = atoi(argv[2]);
		end_frame_id = atoi(argv[3]);
	}
	else std::cout << "Too many arguments!\n";
	
	if (end_frame_id > 0) {
		for (; frame_id < end_frame_id; frame_id++) {
			if (!directory_path.empty()) {
				string image_file_path = directory_path + "/" + to_string(frame_id) + ".png";
				TriangulateImage(directory_path, image_file_path, frame_id);
			}
		}
	}
	else {
		if (!directory_path.empty()) {
			string image_file_path = directory_path + "/" + to_string(frame_id) + ".png";
			TriangulateImage(directory_path, image_file_path, frame_id);
		}
	}
	
	return 0;
}