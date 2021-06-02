#include "stdafx.h"
#include "common.h"

typedef struct {
	Mat mat;
	int count;
} LabelInfo;

LabelInfo info;

bool isInside(Mat img, Point p) {
	return p.y >= 0 && p.y < img.rows && p.x >= 0 && p.x < img.cols;
}

float distance_between_points(Point p1, Point p2) {
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

float distance_between_point_and_line(Point p, Point lineP1, Point lineP2) {
	float numerator = fabs((lineP2.y - lineP1.y) * p.x - (lineP2.x - lineP1.x) * p.y + lineP2.x * lineP1.y - lineP2.y * lineP1.x);
	float denominator = sqrt(pow(lineP2.y - lineP1.y, 2) + pow(lineP2.x - lineP1.x, 2));

	return numerator / denominator;
}

int index_of_furthest_point_from_point(Point p, std::vector<Point> contour) {
	int index = 0;
	float max = 0.0f;

	for (int i = 0; i < contour.size(); i++) {
		float dist = distance_between_points(p, contour.at(i));

		if (dist > max) {
			max = dist;
			index = i;
		}
	}

	return index;
}

int index_of_furthest_point_from_segment(Point segmentStart, Point segmentEnd, std::vector<Point> contour) {
	int i = 0;
	
	while (contour.at(i) != segmentStart) {
		i++;
	}

	int index = i;
	float max = 0.0f;

	while (contour.at(i) != segmentEnd) {
		float dist = distance_between_point_and_line(contour.at(i), segmentStart, segmentEnd);
		if (dist > max) {
			max = dist;
			index = i;
			//std::cout << "New max: " << contour.at(i);
		}

		i++;
	}

	return index;
}

void labeling_twopass(Mat src) {
	Mat labels(src.rows, src.cols, CV_32SC1, Scalar(0));
	int label = 0;

	std::vector<std::vector<int>> edges;

	int di[4] = { -1, -1, -1, 0 };
	int dj[4] = { -1, 0, 1, -1 };
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				std::vector<int> L;
				for (int k = 0; k < 4; k++) {
					if (labels.at<int>(i + di[k], j + dj[k]) > 0) {
						L.push_back(labels.at<int>(i + di[k], j + dj[k]));
					}
				}
				if (L.size() == 0) {
					label++;
					labels.at<int>(i, j) = label;
					edges.resize(label + 1);
				}
				else {
					int min = INT_MAX;
					for (int x : L) {
						if (x < min) {
							min = x;
						}
					}
					labels.at<int>(i, j) = min;
					for (int y : L) {
						if (y != min) {
							edges[min].push_back(y);
							edges[y].push_back(min);
						}
					}
				}
			}
		}
	}

	int newLabel = 0;
	int* newLabels = new int[label + 1]{ 0 };

	for (int i = 1; i <= label; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			std::queue<int> Q;
			newLabels[i] = newLabel;
			Q.push(i);

			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();
				for (int y : edges[x]) {
					if (newLabels[y] == 0) {
						newLabels[y] = newLabel;
						Q.push(y);
					}
				}
			}
		}
	}

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
		}
	}

	delete[] newLabels;

	info.count = newLabel;
	info.mat = labels;
}

void border_tracing(Mat src, std::vector<Point>& contour, int label) {
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	int i = 0;
	int j = 0;
	bool found = false;
	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0 && info.mat.at<int>(i, j) == label) {
				found = true;
				break;
			}
		}
		if (found) {
			break;
		}
	}

	int n = 0;
	Point p0 = Point(j, i);
	contour.push_back(p0);
	Point p1;
	Point pn1;
	Point pn = p0;
	int dir = 7;

	do {
		n++;

		if (dir % 2 == 0) {
			dir = (dir + 7) % 8;
		}
		else {
			dir = (dir + 6) % 8;
		}

		i = pn.y;
		j = pn.x;

		while (src.at<uchar>(i + di[dir], j + dj[dir]) != 0 && info.mat.at<int>(i + di[dir], j + dj[dir]) != label) {
			dir = (dir + 1) % 8;
		}

		pn1 = pn;
		pn = Point(j + dj[dir], i + di[dir]);
		if (n == 1) {
			p1 = pn;
		}

		contour.push_back(pn);
	} while (!((n >= 2) && (pn == p1) && (pn1 == p0)));
}

std::vector<int> polygonal_approx(std::vector<Point> contour, float error) {
	std::vector<int> A;
	std::vector<int> B;

	int i = 0;
	int j = index_of_furthest_point_from_point(contour.at(0), contour);
	A.push_back(j);
	B.push_back(j);
	A.push_back(i);

	while (!A.empty()) {
		int k = A.at(A.size() - 1);
		int l = B.at(B.size() - 1);

		int m = index_of_furthest_point_from_segment(contour.at(k), contour.at(l), contour);

		if (distance_between_point_and_line(contour.at(m), contour.at(k), contour.at(l)) > error) {
			A.push_back(m);
		}
		else {
			for (std::vector<int>::iterator iter = A.begin(); iter != A.end(); ++iter) {
				if (*iter == k)
				{
					A.erase(iter);
					break;
				}
			}

			B.push_back(k);
		}
	}

	return B;
}

void contour_and_polygonal_approx() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, IMREAD_GRAYSCALE);

		int error;
		std::cout << "Enter error:";
		std::cin >> error;

		Mat contour_img(src.rows, src.cols, CV_8UC1, Scalar(255));
		Mat poly_approx_points_img(src.rows, src.cols, CV_8UC1, Scalar(255));
		Mat poly_approx_draw_img(src.rows, src.cols, CV_8UC1, Scalar(255));

		labeling_twopass(src);
		std::cout << "Number of objects found: " << info.count << "\n";
		std::cout << "Loading..." << "\n";

		for (int i = 1; i <= info.count; i++) {
			std::vector<Point> contour;
			border_tracing(src, contour, i);

			for (Point p : contour) {
				contour_img.at<uchar>(p.y, p.x) = 0;
			}

			std::vector<int> poly_approx = polygonal_approx(contour, error);

			for (int i = 0; i < poly_approx.size() - 1; i++) {
				circle(poly_approx_points_img, contour.at(poly_approx.at(i)), 3, Scalar(0), FILLED, LINE_8);
				line(poly_approx_draw_img, contour.at(poly_approx.at(i)), contour.at(poly_approx.at(i + 1)), Scalar(0), 2, LINE_8);
			}
			circle(poly_approx_points_img, contour.at(poly_approx.at(poly_approx.size() - 1)), 3, Scalar(0), FILLED, LINE_8);
		}

		std::cout << "Done!" << "\n";
		imshow("image", src);
		imshow("contour", contour_img);
		imshow("poly approx points", poly_approx_points_img);
		imshow("poly approx draw", poly_approx_draw_img);
		waitKey(0);
	}
}

int main()
{
	contour_and_polygonal_approx();
}