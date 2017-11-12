#pragma once

#include <vector>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::tuple;

typedef Matrix<double> Image;
typedef Matrix<tuple<uint,uint,uint>> Picture;

Image grayscale(BMP* src_image);
void Partition_color(Picture color_pict,Picture *pictures_color,const int parts);
void Partition(Image src,Image *cells,const int parts);
Image sobel_x(Image src_image);
Image sobel_y(Image src_image);
Image calc_gradient_direction(Image sobX, Image sobY);
Image calc_gradient_absolution(Image sobX, Image sobY);
vector<float> calc_histogram(Image dir, Image abs);
vector<float> calc_lbl(Image src);
vector<float> calc_color(Picture src);
Picture Transform(BMP* src);

struct Histogram
{
    int r[256];
    int g[256];
    int b[256];
};


template <typename T>
T sqr(T a) {
	return a*a;
}