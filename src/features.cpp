#include "features.hpp"
#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include <stdlib.h>

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

Picture Transform(BMP* src) {
	Picture res_image(src->TellWidth(),src->TellHeight());
	for (uint i = 0; i < res_image.n_rows; i++) 
		for (uint j = 0; j < res_image.n_cols; j++) {
			RGBApixel pixel = src->GetPixel(i,j);
			auto r = pixel.Red;
			auto g = pixel.Green;
			auto b = pixel.Blue; 
			res_image(i,j) = std::make_tuple(r,g,b);
		}
    return res_image;
}

Image grayscale(BMP* src) {    
	Image res_image(src->TellWidth(),src->TellHeight());
	double Y;
	for (uint i = 0; i < res_image.n_rows; i++) 
		for (uint j = 0; j < res_image.n_cols; j++) {
			RGBApixel pixel = src->GetPixel(i,j);
			Y = 0.299 * pixel.Red + 0.587 * pixel.Green + 0.114 * pixel.Blue; 
			res_image(i,j) = Y;
		}
    return res_image;
}

void Partition_color(Picture color_pict, Picture *pictures_color,const int parts){
	const uint sector = sqrt(parts);
    int fromX = 0, fromY = 0;
    
    for (uint i = 0; i < sector; i++) {
    	for (uint j = 0; j < sector; j++) {
    		*pictures_color = color_pict.submatrix(fromX,fromY,color_pict.n_rows/sector,color_pict.n_cols/sector);
    		pictures_color++;
    		fromY += color_pict.n_cols / sector;
    	}
    	fromY = 0;
    	fromX += color_pict.n_rows / sector;
    }
}


void Partition(Image src,Image *cells,const int parts) {
    const uint sector = sqrt(parts);
    int fromX = 0, fromY = 0;
    
    for (uint i = 0; i < sector; i++) {
    	for (uint j = 0; j < sector; j++) {
    		*cells = src.submatrix(fromX,fromY,src.n_rows/sector,src.n_cols/sector);
    		cells++;
    		fromY += src.n_cols / sector;
    	}
    	fromY = 0;
    	fromX += src.n_rows / sector;
    }
}

Image sobel_x(Image src_image) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    uint radius = (kernel.n_cols - 1) / 2;
    Image res_image(src_image.n_rows,src_image.n_cols);
    auto size = 2 * radius + 1;
    const auto start_i = radius;
    const auto end_i = src_image.n_rows + radius;
    const auto start_j = radius;
    const auto end_j = src_image.n_cols + radius;
    Image border(src_image.n_rows + 2*radius, src_image.n_cols + 2*radius);
    for(ssize_t i = 0; i < border.n_rows; ++i) //mirror
        for(ssize_t j = 0; j < border.n_cols; ++j){
            auto x = i - radius;
            auto y = j - radius;
            if(x >= src_image.n_rows)
                x = 2*(src_image.n_rows-1) - x;
            if(y >= src_image.n_cols)
                y = 2*(src_image.n_cols-1) - y;
            border(i,j) = src_image(abs(x),abs(y));
        }

    for (uint i = start_i; i < end_i; ++i) {
        for (uint j = start_j; j < end_j; ++j) {
            auto neighbourhood = border.submatrix(i-radius,j-radius,size,size);
            double color = 0;
            double sum = 0;
            for (uint hor = 0; hor < size; hor++)
                for (uint ver = 0; ver < size; ver++) {
                        color = neighbourhood(hor,ver);
                        sum += color * kernel(hor,ver);
                }
            res_image(i-radius, j-radius) = sum;
        }
    }
    return res_image;
}

Image sobel_y(Image src_image) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    uint radius = (kernel.n_cols - 1) / 2;
    Image res_image(src_image.n_rows,src_image.n_cols);
    auto size = 2 * radius + 1;
    const auto start_i = radius;
    const auto end_i = src_image.n_rows + radius;
    const auto start_j = radius;
    const auto end_j = src_image.n_cols + radius;
    Image border(src_image.n_rows + 2*radius, src_image.n_cols + 2*radius);
    for(ssize_t i = 0; i < border.n_rows; ++i) //mirror
        for(ssize_t j = 0; j < border.n_cols; ++j){
            auto x = i - radius;
            auto y = j - radius;
            if(x >= src_image.n_rows)
                x = 2*(src_image.n_rows-1) - x;
            if(y >= src_image.n_cols)
                y = 2*(src_image.n_cols-1) - y;
            border(i,j) = src_image(abs(x),abs(y));
        }

    for (uint i = start_i; i < end_i; ++i) {
        for (uint j = start_j; j < end_j; ++j) {
            auto neighbourhood = border.submatrix(i-radius,j-radius,size,size);
            double color = 0;
            double sum = 0;
            for (uint hor = 0; hor < size; hor++)
                for (uint ver = 0; ver < size; ver++) {
                        color = neighbourhood(hor,ver);
                        sum += color * kernel(hor,ver);
                }
            res_image(i-radius, j-radius) = sum;
        }
    }
    return res_image;
}

Image calc_gradient_direction(Image sobX, Image sobY) {
	auto width = sobX.n_cols;
	auto height = sobX.n_rows;
	Image res_image(height,width);

	for (uint i = 0; i < height; i++) 
		for (uint j = 0; j < width; j++)
			res_image(i,j) = std::atan2(sobY(i,j),sobX(i,j));

	return res_image;
}

Image calc_gradient_absolution(Image sobX, Image sobY) {
	auto width = sobX.n_cols;
	auto height = sobX.n_rows;
	Image res_image(height,width);

	for (uint i = 0; i < height; i++) 
		for (uint j = 0; j < width; j++)
			res_image(i,j) = sqrt( sqr(sobX(i,j)) + sqr(sobY(i,j)) );

	return res_image;
}

vector<float> calc_histogram(Image direct, Image absolut) {

	const int segments = 32;
	const double pi = 3.141592653;
	vector<float> res_histogram(segments, 0.0f);
	for (uint i = 0; i < direct.n_rows; ++i)
		for(uint j = 0; j < direct.n_cols; ++j){
			uint index = uint((direct(i,j) / (2*pi) * segments + pi)) % segments;
			res_histogram[index] += absolut(i,j);
		}

		//normalization
	float length2 = 0.0f;
	for (uint i = 0; i < segments; ++i)
		length2 += sqr(res_histogram[i]);
	if (length2 > 0.001){
		float length = sqrt(length2);
		for (uint i = 0; i < segments; ++i)
			res_histogram[i] /= length;
	}
	return res_histogram;
}

vector<float> calc_lbl(Image src_image) {
    uint radius = 1;
    Image res_image(src_image.n_rows,src_image.n_cols);
    auto size = 2 * radius + 1;
    const auto start_i = radius;
    const auto end_i = src_image.n_rows + radius;
    const auto start_j = radius;
    const auto end_j = src_image.n_cols + radius;
    Image border(src_image.n_rows + 2*radius, src_image.n_cols + 2*radius);
    for(ssize_t i = 0; i < border.n_rows; ++i) //mirror
        for(ssize_t j = 0; j < border.n_cols; ++j){
            auto x = i - radius;
            auto y = j - radius;
            if(x >= src_image.n_rows)
                x = 2*(src_image.n_rows-1) - x;
            if(y >= src_image.n_cols)
                y = 2*(src_image.n_cols-1) - y;
            border(i,j) = src_image(abs(x),abs(y));
        }
    vector<float> lbl_hist(256,0);
    for (uint i = start_i; i < end_i; ++i) {
        for (uint j = start_j; j < end_j; ++j) {
            auto neighbourhood = border.submatrix(i-radius,j-radius,size,size);
            vector<int> binary(8,0);
            double elem = border(i,j);
            double color = 0;
            for (uint hor = 0; hor < size; hor++)
                for (uint ver = 0; ver < size; ver++) {
                        color = neighbourhood(hor,ver);
                        if (color >= elem) {
                        	if (hor == 0) { binary[ver] = 1; }
                        	if (ver == 2) { binary[hor + ver] = 1; }
                        	if ( (hor == 1) && (ver == 0) ) { binary[7] = 1; }
                        	if ( (hor == 2) && (ver == 0) ) { binary[6] = 1; }
                        	if ( (hor == 2) && (ver == 1) ) { binary[5] = 1; }
                        }
                }
            int value = 0;
            for (int k = binary.size() - 1; k > 0; k--) {
            	value += binary[k] * std::pow(2,binary.size() -1 - k);
            }
            lbl_hist[value] += 1;
        }
    }
    return lbl_hist;
}

vector<float> calc_color(Picture src_image) {    
    Picture res_image(src_image.n_rows,src_image.n_cols);
    uint r = 0, g = 0, b = 0;
    Histogram Hist;
    vector<float> color_hist(3, 0);
    for(uint k = 0; k < 256; k++) 
    	Hist.r[k] = Hist.g[k] = Hist.b[k] = 0;
    for (uint hor = 0; hor < src_image.n_rows; hor++)
    	for (uint ver = 0; ver < src_image.n_cols; ver++){  
        	std::tie(r,g,b) = src_image(hor,ver);
            Hist.r[r]++;
            Hist.g[g]++;
            Hist.b[b]++;
        }
	int sb = 0, sg = 0, sr = 0;
    int k = 0, m = 255; 
    for(k = 0, m = 255; k <= m; )
    	sr = sr > 0 ? sr - Hist.r[k++] : sr + Hist.r[m--];
    color_hist[0] = k;
    for(k = 0, m = 255; k <= m; )
            sg = sg > 0 ? sg - Hist.g[k++] : sg + Hist.g[m--];
    color_hist[1] = k;
    for(k = 0, m = 255; k <= m; )
    	sb = sb > 0 ? sb - Hist.b[k++] : sb + Hist.b[m--];
    color_hist[2] = k;
    
    	//normalization
    float length2 = 0.0f;
	for (uint i = 0; i < 3; ++i)
		length2 += sqr(color_hist[i]);
	if (length2 > 0.001){
		float length = sqrt(length2);
		for (uint i = 0; i < 3; ++i)
			color_hist[i] /= length;
	}

    return color_hist;
}