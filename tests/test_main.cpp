/* Googletest main */
#include <gtest/gtest.h>
#include <iostream>

#include "matrix.h"
#include "classifier.h" 
#include "linear.h"
#include "argvparser.h"
#include "features.hpp"
#include "features.cpp"
#include "EasyBMP.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

#define PATH "../../data/Lenna.bmp"

TEST(TestGradientAbsolut, test) {

	BMP* src = new BMP();
    src->ReadFromFile(PATH);
	Image src_image = grayscale(src);
	Image sobX = sobel_x(src_image);
	Image sobY = sobel_y(src_image);
	Image image = calc_gradient_absolution(sobX, sobY);	
	Image image_SSE = calc_gradient_absolution_sse(sobX, sobY);
	ASSERT_EQ(image.n_rows, image_SSE.n_rows) << "different number of rows";
	ASSERT_EQ(image.n_cols, image_SSE.n_cols) << "different number of cols";

	float error = imgDif(image, image_SSE);
	ASSERT_FLOAT_EQ(0, error) << "max difference: " << error;
}

 
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}