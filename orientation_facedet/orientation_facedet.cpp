#include "stdio.h"
#include "stdlib.h"
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "face_detection.h"
#include "scandir.h"
#include "gflags.h"
#include "assert.h"
#include <sstream>

DEFINE_string(model,"", "model path");
DEFINE_string(infile,"","input file with image paths");
DEFINE_string(outdir, "", "output dir");

class LocalImageData :public seeta::ImageData
{
public:
	LocalImageData(cv::Mat& img)
	{
		width = img.cols;
		height = img.rows;
		num_channels = 1;
		data = new unsigned char[width * height];
		assert(data != NULL);
		if (img.channels() != 1)
		{
			cv::Mat gray;
			cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
			for (int y = 0; y < height; y++)
			{
				memcpy(data + y * width, gray.data + y * gray.step[0], width);
			}

		}
		else
		{
			for (int y = 0; y < height; y++)
			{
				memcpy(data + y * width, img.data + y * img.step[0], width);
			}
		}
	}
	~LocalImageData()
	{
		if (data != NULL)
			delete[] data;
	}
public:
	seeta::ImageData* get_ImageData()
	{
		return (seeta::ImageData*)(this);
	}
};

//#define SHOW_DEBUG

bool rotate_images(cv::Mat& img, vector< cv::Mat >& imgs)
{
	int rotCodes[4] = { 0, cv::ROTATE_90_CLOCKWISE, cv::ROTATE_180, cv::ROTATE_90_COUNTERCLOCKWISE };
	for (int k = 0; k < 4; k++)
	{
		if (k == 0)
		{
			imgs.push_back(img);
			continue;
		}
		cv::Mat rot;
		cv::rotate(img, rot, rotCodes[k]);
		imgs.push_back(rot);
	}
	return true;
}

class LOG
{
private:
	std::string _filepath;
	std::vector< std::string> _logs;
public:
	LOG(const char* filepath)
	{
		_filepath = filepath;
		_logs.clear();
	}
	~LOG()
	{
		flush();
	}
private:
	bool flush()
	{
		if (!_logs.empty())
		{
			FILE* fd = fopen(_filepath.c_str(), "a+b");
			assert(fd != NULL);
			for (int k = 0; k < _logs.size(); k++)
			{
				fprintf(fd, "%s\r\n", _logs[k].c_str());
			}
			fclose(fd);
			_logs.clear();
		}
		return true;
	}
public:
	bool add(string log)
	{
		_logs.push_back(log);
		if (_logs.size() > 100)
			flush();
		return true;
	}

};

int main(int argc, char* argv[])
{
	google::ParseCommandLineFlags(&argc, &argv,false);
	seeta::FaceDetection* detection = new seeta::FaceDetection(FLAGS_model.c_str());
	assert(detection != NULL);

	detection->SetMinFaceSize(80);
	detection->SetScoreThresh(3.0f);
	detection->SetImagePyramidScaleFactor(0.8f);
	detection->SetWindowStep(4, 4);
	
	std::vector< std::string > filelist;
	scandir::loadlist(FLAGS_infile.c_str(), filelist);
	if (filelist.empty() == true)
	{
		std::cout << "no image found in list file" << std::endl;
		return 0;
	}


	unsigned long photoidx = 0;
	LOG log((FLAGS_outdir + "orientation_facedet.log").c_str());
#ifdef SHOW_DEBUG
	cv::namedWindow("auto_rotation", cv::WINDOW_NORMAL);
#endif
	for (int k = 0; k < filelist.size(); k++)
	{
		cv::Mat img = cv::imread(filelist[k], -1);
		if (img.data == NULL)
		{
			std::cout << "error loading " << filelist[k].c_str() << endl;
			continue;
		}

		std::vector< cv::Mat > rotImgs;
		std::vector< float > rotScores;
		std::vector< std::vector< seeta::FaceInfo > > rotFaces;
		rotate_images(img, rotImgs);
		for (int rot = 0; rot < rotImgs.size(); rot++)
		{
			LocalImageData localImg(rotImgs[rot]);
			std::vector< seeta::FaceInfo > faces = detection->Detect(*(localImg.get_ImageData()));
			float score = 0;
			std::vector< seeta::FaceInfo > fc;
			for (int j = 0; j < faces.size(); j++)
			{
				score += faces[j].score;
				fc.push_back(faces[j]);
			}
			rotScores.push_back(score);
			rotFaces.push_back(fc);

#ifdef SHOW_DEBUG
			for (int j = 0; j < faces.size(); j++)
			{
				seeta::Rect facebox = faces[j].bbox;
				cv::Rect rc(facebox.x, facebox.y, facebox.width, facebox.height);
				cv::rectangle(rotImgs[rot], rc, CV_RGB(255, 0, 0), 2);
			}
			cv::imshow("auto_rotation", rotImgs[rot]);
			cv::waitKey(-1);
#endif
		}

		int topScoreRot = 0;
		for (int rot = 1; rot < rotScores.size(); rot++)
		{
			if (rotScores[rot] > rotScores[topScoreRot])
				topScoreRot = rot;
		}
		char outpath[1024];
		sprintf(outpath, "%s%.9d.jpg", FLAGS_outdir.c_str(), photoidx++);
		cv::imwrite(outpath, rotImgs[topScoreRot]);


		ostringstream oss;
		oss << outpath << "|" <<fixed<<setprecision(2)<< topScoreRot << "," << rotScores[topScoreRot];
		if (rotFaces[topScoreRot].size() > 0)
		{
			oss << "|";
			for (int j = 0; j < rotFaces[topScoreRot].size(); j++)
			{
				oss << rotFaces[topScoreRot][j].bbox.x << ",";
				oss << rotFaces[topScoreRot][j].bbox.x + rotFaces[topScoreRot][j].bbox.width - 1 << ",";
				oss << rotFaces[topScoreRot][j].bbox.y << ",";
				oss << rotFaces[topScoreRot][j].bbox.y + rotFaces[topScoreRot][j].bbox.height - 1;
				if (j + 1 < rotFaces[topScoreRot].size())
					oss << ",";
			}
		}
		log.add(oss.str());

	}

#ifdef SHOW_DEBUG
	cv::destroyAllWindows();
#endif

	delete detection;
    return 0;
}