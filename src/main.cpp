#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <stdint.h>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "features.hpp"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

/**
@mainpage Классификация объектов с использованием SSE
@author Проект выполнила Глущенко Майя | 322 группа
*/

/**
@file main.cpp
*/

/**
@function LoadFileList
загружает список файлов и их меток классов из 'data_file' и записывает их в 'file_list'
@param[in] data_file хранит пути к файлам и метки классов
@param[out] file_list - вектор <путь к файлу, метка>*/
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

/**
@function LoadImages
загружает список файлов изображения из 'file_list' и сохраняет их в 'data_set' 
@param[in] file_list - вектор <путь к файлу, метка>
@param[out] data_set - вектор <указатель на изображение, метка> */
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

/**
@function SavePredictions 
сохраняет предсказанный результат в файл 
@param[in] file_list - вектор <путь к файлу, метка>
@param[in] labels - вектор <метка> 
@param[out] prediction_file - выходной текстовый файл, хранящий предсказания классификатора*/
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

/**
@function ExtractFeatures
извлекает признаки из 'data_set'. Здесь обрабатываются все изображения из 'data_set' для дальнейшего построения 
гистограммы 'features' для того, чтобы обучить модель и предсказать результат
@param[in] data_set - вектор <BMP* image, int label> 
@param[out] features - вектор <vector<float>, int labels> */
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    Timer t;
    t.start();
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        BMP* image = data_set[image_idx].first;
        int label = data_set[image_idx].second;
        Image res_image = grayscale(image);
        const int parts = 64;
        Image pictures[parts];
        Picture pictures_color[parts];
        Picture color_pict = Transform(image);
        Partition_color(color_pict,pictures_color,parts);
        Partition(res_image,pictures,parts);

        vector<float> HOG;
        for(int i = 0; i < parts; ++i) {
            Image sobX = sobel_x(pictures[i]);
            Image sobY = sobel_y(pictures[i]);

            Image direct = calc_gradient_direction(sobX,sobY);
            Image absolut = calc_gradient_absolution(sobX,sobY);

            vector<float> current_histogram = calc_histogram(direct, absolut);
            vector<float> current_lbl = calc_lbl(pictures[i]);
            vector<float> current_color = calc_color(pictures_color[i]);

            //конкатенация с ранее подсчитаннымм гистограммами
            HOG.insert(HOG.end(), current_histogram.begin(), current_histogram.end());
            //HOG.insert(HOG.end(), current_lbl.begin(), current_lbl.end());
            HOG.insert(HOG.end(), current_color.begin(), current_color.end());
        }
        features->push_back(make_pair(HOG, label));
    }
     t.check("Naive implementation");
     t.stop();
}

/**
@function ExtractFeaturesSSE
аналогично ExtractFeatures() извлекает признаки из 'data_set', но с использованием SSE
@param[in] data_set - вектор <BMP* image, int label> 
@param[out] features - вектор <vector<float>, int labels> */
void ExtractFeaturesSSE(const TDataSet& data_set, TFeatures* features) {
    Timer t;
    t.start();
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        BMP* image = data_set[image_idx].first;
        int label = data_set[image_idx].second;
        Image res_image = grayscale(image);
        const int parts = 64;
        Image pictures[parts];
        Picture pictures_color[parts];
        Picture color_pict = Transform(image);
        Partition_color(color_pict,pictures_color,parts);
        Partition(res_image,pictures,parts);

        vector<float> HOG;
        for(int i = 0; i < parts; ++i) {
            Image sobX = sobel_x_sse(pictures[i]);
            Image sobY = sobel_y_sse(pictures[i]);

            Image direct = calc_gradient_direction(sobX,sobY);
            Image absolut = calc_gradient_absolution_sse(sobX,sobY);

            vector<float> current_histogram = calc_histogram(direct, absolut);
            vector<float> current_lbl = calc_lbl(pictures[i]);
            vector<float> current_color = calc_color(pictures_color[i]);

            //конкатенация с ранее подсчитаннымм гистограммами
            HOG.insert(HOG.end(), current_histogram.begin(), current_histogram.end());
            //HOG.insert(HOG.end(), current_lbl.begin(), current_lbl.end());
            HOG.insert(HOG.end(), current_color.begin(), current_color.end());
        }
        features->push_back(make_pair(HOG, label));
    }
    t.check("SSE implementation");
    t.stop();
}

/**
@function ClearDataset
очищает стуктуру 'data_set'
@param data_set is vector of <BMP* image, int label> */
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

/**
@function TrainClassifier
обучает классификатор, используя данные из 'data_file' и сохраняет обученную модель в 'model_file'
@param[in] data_set - вектор <BMP* image, int label>
@param[out] model_file - выходной файл, хранящий информацию об обученной модели
@param[in] sse_flag - логическая переменная, показывающая тип последующей обработки */
void TrainClassifier(const string& data_file, const string& model_file, bool sse_flag) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    if(sse_flag)
        ExtractFeaturesSSE(data_set, &features);
    else
        ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.15;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

/**
@function PredictData
предсказывает классы данных из 'data_file', используя модель из 'model_file', сохраняет предсказания в 'prediction_file'
@param[in] data_set - вектор <BMP* image, int label>
@param[out] model_file - выходной файл, хранящий информацию об обученной модели
@param[in] sse_flag - логическая переменная, показывающая тип последующей обработки */
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file, bool sse_flag) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    if(sse_flag)
        ExtractFeaturesSSE(data_set, &features);
    else
        ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
    cmd.defineOption("SSE", "Using SSE");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");
    cmd.defineOptionAlternative("SSE", "s");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");
    bool sse_flag = cmd.foundOption("SSE");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file, sse_flag);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file, sse_flag);
    }
}