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

/**
@file features.cpp
Документация в features.hpp
*/

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
	float Y;
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
    Matrix<float> kernel = {{-1, 0, 1},
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
            float color = 0;
            float sum = 0;
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

Image sobel_x_sse(Image src_image) {
    Matrix<float> kernel = {{-1, 0, 1},
                            {-2, 0, 2},
                            {-1, 0, 1}};
    uint radius = (kernel.n_cols - 1) / 2;
    Image res_image(src_image.n_rows,src_image.n_cols);
    auto size = 2 * radius + 1;
    const auto start_i = radius;
    const auto end_i = src_image.n_rows + radius;
    const auto start_j = radius;
    const auto end_j = src_image.n_cols + radius;

    const auto block_size = 4;
    const auto crop_cols = res_image.n_cols % block_size;
    const auto end_cols = res_image.n_cols - crop_cols + radius;

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
        //kernel matrix vectorization
    __m128 kernel1 = _mm_setr_ps(kernel(0,0), kernel(0,1), kernel(0,2), 0); //first row
    __m128 kernel2 = _mm_setr_ps(kernel(1,0), kernel(1,1), kernel(1,2), 0); //second row
    __m128 kernel3 = _mm_setr_ps(kernel(2,0), kernel(2,1), kernel(2,2), 0); //third row
    for (uint i = start_i; i < end_i; ++i) {
        for (uint j = start_j; j < end_cols; ++j) {
            auto neighbourhood = border.submatrix(i-radius,j-radius,size,size);
                //load pixels           
            __m128 row1 = _mm_setr_ps(neighbourhood(0,0), neighbourhood(0,1), neighbourhood(0,2), 0);
            __m128 row2 = _mm_setr_ps(neighbourhood(1,0), neighbourhood(1,1), neighbourhood(1,2), 0); 
            __m128 row3 = _mm_setr_ps(neighbourhood(2,0), neighbourhood(2,1), neighbourhood(2,2), 0); 
                //multiply 
            __m128 ker_row1 = _mm_mul_ps(kernel1, row1); //first row
            __m128 ker_row2 = _mm_mul_ps(kernel2, row2); //second row
            __m128 ker_row3 = _mm_mul_ps(kernel3, row3); //third row
                //sum
            __m128 ker_row12 = _mm_add_ps(ker_row1, ker_row2); // firts row + second row
            __m128 ker_row123 = _mm_add_ps(ker_row12, ker_row3); // all rows sum - a|b|c|0
            __m128 sum12 = _mm_hadd_ps(ker_row123, _mm_setzero_ps()); // a+b|c+0|0|0 
            __m128 sum123 = _mm_hadd_ps(sum12, _mm_setzero_ps()); // a+b+c+0|0|0|0
                //move sum value to res pixel
            res_image(i-radius,j-radius) = _mm_cvtss_f32(sum123);
        }
            //processing remaining pixels
        for(uint j = end_cols; j < end_j; ++j){
            auto neighbourhood = border.submatrix(i-radius,j-radius,size,size);
            float color = 0;
            float sum = 0;
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
    Matrix<float> kernel = {{ 1,  2,  1},
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
            float color = 0;
            float sum = 0;
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

Image sobel_y_sse(Image src_image) {
    Matrix<float> kernel = {{ 1,  2,  1},
                            { 0,  0,  0},
                            {-1, -2, -1}};
    uint radius = (kernel.n_cols - 1) / 2;
    Image res_image(src_image.n_rows,src_image.n_cols);
    auto size = 2 * radius + 1;
    const auto start_i = radius;
    const auto end_i = src_image.n_rows + radius;
    const auto start_j = radius;
    const auto end_j = src_image.n_cols + radius;

    const auto block_size = 4;
    const auto crop_cols = res_image.n_cols % block_size;
    const auto end_cols = res_image.n_cols - crop_cols + radius;

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
        //kernel matrix vectorization
    __m128 kernel1 = _mm_setr_ps(kernel(0,0), kernel(0,1), kernel(0,2), 0); //first row
    __m128 kernel2 = _mm_setr_ps(kernel(1,0), kernel(1,1), kernel(1,2), 0); //second row
    __m128 kernel3 = _mm_setr_ps(kernel(2,0), kernel(2,1), kernel(2,2), 0); //third row
    for (uint i = start_i; i < end_i; ++i) {
        for (uint j = start_j; j < end_cols; ++j) {
            auto neighbourhood = border.submatrix(i-radius,j-radius,size,size);
                //load pixels           
            __m128 row1 = _mm_setr_ps(neighbourhood(0,0), neighbourhood(0,1), neighbourhood(0,2), 0);
            __m128 row2 = _mm_setr_ps(neighbourhood(1,0), neighbourhood(1,1), neighbourhood(1,2), 0); 
            __m128 row3 = _mm_setr_ps(neighbourhood(2,0), neighbourhood(2,1), neighbourhood(2,2), 0); 
                //multiply 
            __m128 ker_row1 = _mm_mul_ps(kernel1, row1); //first row
            __m128 ker_row2 = _mm_mul_ps(kernel2, row2); //second row
            __m128 ker_row3 = _mm_mul_ps(kernel3, row3); //third row
                //sum
            __m128 ker_row12 = _mm_add_ps(ker_row1, ker_row2); // firts row + second row
            __m128 ker_row123 = _mm_add_ps(ker_row12, ker_row3); // all rows sum - a|b|c|0
            __m128 sum12 = _mm_hadd_ps(ker_row123, _mm_setzero_ps()); // a+b|c+0|0|0 
            __m128 sum123 = _mm_hadd_ps(sum12, _mm_setzero_ps()); // a+b+c+0|0|0|0
                //move sum value to res pixel
            res_image(i-radius,j-radius) = _mm_cvtss_f32(sum123);
        }
            //processing remaining pixels
        for(uint j = end_cols; j < end_j; ++j){
            auto neighbourhood = border.submatrix(i-radius,j-radius,size,size);
            float color = 0;
            float sum = 0;
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
Image calc_gradient_absolution_sse(Image sobX, Image sobY){
    auto width = sobX.n_cols;
	auto height = sobX.n_rows;
    Image res_image(height, width);
    const auto block_size = 4;
    const auto crop_cols = res_image.n_cols % block_size;
    const auto end_cols = res_image.n_cols - crop_cols;
        //SSE processing
    float *res_row_ptr = res_image._data.get();
    float *hor_row_ptr = sobX._data.get();
    float *ver_row_ptr = sobY._data.get();
    float *res_ptr, *hor_ptr, *ver_ptr;
	for(uint i = 0; i < res_image.n_rows; ++i){
        res_ptr = res_row_ptr;
        hor_ptr = hor_row_ptr;
        ver_ptr = ver_row_ptr;
        for(uint j = 0; j < end_cols; j+=block_size){
            __m128 hor_block = _mm_loadu_ps(hor_ptr);
            __m128 ver_block = _mm_loadu_ps(ver_ptr);
            __m128 hh_block = _mm_mul_ps(hor_block, hor_block);
            __m128 vv_block = _mm_mul_ps(ver_block, ver_block);            
            __m128 res_block = _mm_add_ps(hh_block, vv_block);
            res_block = _mm_sqrt_ps(res_block);
            _mm_storeu_ps(res_ptr, res_block);
            res_ptr += block_size;
            hor_ptr += block_size;
            ver_ptr += block_size;
        }
            //processing last pixels (no SSE)
        for (uint j = end_cols; j < width; j++)
			res_image(i,j) = sqrt( sqr(sobX(i,j)) + sqr(sobY(i,j)) );

        res_row_ptr += res_image.stride;
        hor_row_ptr += sobX.stride;
        ver_row_ptr += sobY.stride;
    }
        
    return res_image;
}

vector<float> calc_histogram(Image direct, Image absolut) {

	const int segments = 32;
	const float pi = 3.141592653;
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
            float elem = border(i,j);
            float color = 0;
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

float imgDif(Image image1, Image image2)
{
  assert((image1.n_rows == image2.n_rows) && (image1.n_cols == image2.n_cols));
  auto width = image1.n_cols;
  auto height = image1.n_rows;
  float res = 0;
  float pixel_res;

  float *row_ptr1 = image1._data.get();
  float *elem_ptr1;
  float *row_ptr2 = image2._data.get();
  float *elem_ptr2;
  for (size_t row_idx = 0; row_idx < height; ++row_idx)
  {
    elem_ptr1 = row_ptr1;
    elem_ptr2 = row_ptr2;
    for (size_t col_idx = 0; col_idx < width; ++col_idx)
    {
      pixel_res = (*elem_ptr1 > *elem_ptr2) ? *elem_ptr1 - *elem_ptr2 : *elem_ptr2 - *elem_ptr1;
      res = res > pixel_res ? res : pixel_res;
      ++elem_ptr1;
      ++elem_ptr2;
    }
    row_ptr1 ++;
    row_ptr2 ++;
  }
  return res;
}