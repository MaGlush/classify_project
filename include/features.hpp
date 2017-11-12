#pragma once
#include <vector>
#include <cmath>
#include <malloc.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <pmmintrin.h>

#include "Environment.h"
#include "Timer.h"

#include "matrix.h"
#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

#include <tuple>
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
using std::make_tuple;

typedef Matrix<float> Image;
typedef Matrix<tuple<uint,uint,uint>> Picture;

/**
@file features.hpp
*/

/**@function grayscale 
 * Конвертирует исходное BMP изображение в grayscale 
 * @param src_image - исходное изображение для обработки 
 * @return матрица <float>*/
Image grayscale(BMP* src_image);
/**@function Partition_color 
 * Разделяет исходное изображение на множетсво других
 * @param[in] color_pict - исходное изображение
 * @param[out] pictures_color - указатели на полученные изображения
 * @param[in] parts - количество изображений*/
void Partition_color(Picture color_pict,Picture *pictures_color,const int parts);
/**@function Partition
 * Разделяет исходное изображение на множетсво других
 * @param[in] src - исходное изображение
 * @param[out] cells - указатели на полученные изображения
 * @param[in] parts - количество изображений*/
void Partition(Image src, Image *cells, const int parts);
/**@function sobel_x 
 * Применяет горизонтальный фильтр Собеля к исходному изображению
 * @param src_image - исходное изображение
 * @return полученная матрица*/
Image sobel_x(Image src_image);
/**@function sobel_y
 * Применяет вертикальный фильтр Собеля к исходному изображению
 * @param src_image - исходное изображение
 * @return полученная матрица*/
Image sobel_y(Image src_image);
/**@function sobel_x_sse 
 * Применяет горизонтальный фильтр Собеля к исходному изображению c использованием технологии SSE
 * @param src_image - исходное изображение
 * @return полученная матрица*/
Image sobel_x_sse(Image src_image);
/**@function sobel_y_sse 
 * Применяет вертикальный фильтр Собеля к исходному изображению c использованием технологии SSE
 * @param src_image - исходное изображение
 * @return полученная матрица*/
Image sobel_y_sse(Image src_image);
/**@function  calc_gradient_direction
* Вычисляет направление градиентов
* @param sobX - исходное изображение с уже примененным горизонтальным фильтром Собеля sobel_x()
* @param sobY - исходное изображение с уже примененным вертикальным фильтром Собеля sobel_y()
* @return полученная матрица*/
Image calc_gradient_direction(Image sobX, Image sobY);
/**@function  calc_gradient_absolution
* Вычисляет модули градиентов
* @param sobX - исходное изображение с уже примененным горизонтальным фильтром Собеля sobel_x()
* @param sobY - исходное изображение с уже примененным вертикальным фильтром Собеля sobel_y()
* @return полученная матрица*/
Image calc_gradient_absolution(Image sobX, Image sobY);
/**@function  calc_gradient_absolution_sse
* Вычисляет модули градиентов с помощью технологии SSE
* @param sobX - исходное изображение с уже примененным горизонтальным фильтром Собеля sobel_x_sse()
* @param sobY - исходное изображение с уже примененным вертикальным фильтром Собеля sobel_y_sse()
* @return полученная матрица*/
Image calc_gradient_absolution_sse(Image sobX, Image sobY);
/**@function  calc_histogram
* Вычисляет гистограмму ориентированных градиентов
* @param dir - матрица направлений градиентов, вычисленная с помощью calc_gradient_direction()
* @param abs - матрица модулей градиентов, вычисленная с помощью calc_gradient_absolution()
* @return полученная гистограмма*/
vector<float> calc_histogram(Image dir, Image abs);
/**@function  calc_lbl
* Вычисляет гистограмму бинарных шаблонов
* @param src - исходная матрица 
* @return полученная гистограмма*/
vector<float> calc_lbl(Image src);
/**@function  calc_color
* Вычисляет гистограмму цветовых признаков
* @param src - исходная матрица 
* @return полученная гистограмма*/
vector<float> calc_color(Picture src);
/**@function  Transform
* Конвертирует BMP в Picture
* @param src - исходное изображение 
* @return полученная матрица*/
Picture Transform(BMP* src);

struct Histogram
{
    int r[256]; ///< red channel
    int g[256]; ///< green channel
    int b[256]; ///< blue channel
};

/**@function sqr
 * возводит переменную в квадрат
 * @param a - переменная 
 * @return a*a */
template <typename T>
T sqr(T a) {
	return a*a;
}