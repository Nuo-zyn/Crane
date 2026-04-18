#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <Windows.h>
#include <algorithm>

#define BEAN_YOLO "D:/opencv_test/bibilabu/best.onnx"

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace chrono;

struct yoloall
{
	Net net;                    // 神经网络
	int netWidth = 640;         
	int netHeight = 640;        
	float confThreshold = 0.5f; 
	float nmsThreshold = 0.45f; // 去重框用的
};




//数字YOLO
class DigitalModel
{
public:
	struct Result {
		int class_id;
		float confidence;
		Rect bbox;
		Point2f center;
	};

private:
	dnn::Net net;
public:
	DigitalModel(const string& onnx_path) {
		net = dnn::readNetFromONNX(onnx_path);
		net.setPreferableBackend(DNN_BACKEND_OPENCV);
		net.setPreferableTarget(DNN_TARGET_CPU);
		cout << "[数字YOLO] 加载完成: " << onnx_path << endl;
	}

	vector<Result> detect(const Mat& frame) {
		vector<Result> results;
		return results;
	}
};


//豆子YOLO
class BeanModel
{
private:
	yoloall bean_net;
public:
	/*struct Result {
		int class_id;
		float confidence;
		Rect bbox;
		Point2f center;
	};*/

	BeanModel(const string& onnx_path) {
		bean_net.net = dnn::readNetFromONNX(onnx_path);
		bean_net.net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
		bean_net.net.setPreferableTarget(dnn::DNN_TARGET_CPU);
		cout << "[豆子YOLO] 加载完成: " << onnx_path << endl;
	}

//豆子检测推理接口
	vector<DigitalModel::Result> detect(const Mat& frame) {
	vector<DigitalModel::Result> results;
	Mat blob;
	blobFromImage(frame, blob, 1.0 / 255.0, Size(bean_net.netWidth, bean_net.netHeight), Scalar(), true, false);
	bean_net.net.setInput(blob);
	Mat output = bean_net.net.forward();
	Mat outMat;

	if (output.dims == 3) {
		outMat = Mat(output.size[1], output.size[2], CV_32F, output.data);
	}
	else {
		outMat = output;
	}
	// 有时候数据是横着的，要转一下
	if (outMat.rows < outMat.cols) {
		transpose(outMat, outMat);
	}

	int numAnchors = outMat.rows;    // 多少个候选框
	int numAttributes = outMat.cols; // 每个框几个数
	int numClasses = numAttributes - 4;  // 类别数，减掉坐标那4个

	// 坐标缩放比例，模型输出是640x640下的，要转回原图的大小
	float scaleX = (float)frame.cols / bean_net.netWidth;
	float scaleY = (float)frame.rows / bean_net.netHeight;

	std::vector<Rect> tmpBoxes;
	std::vector<int> tmpClassIds;
	std::vector<float> tmpConfs;

	for (int i = 0; i < numAnchors; i++) {
		const float* data = outMat.ptr<float>(i);

		// 拿到 640x640 下的中心点坐标和宽高
		float x_center = data[0];
		float y_center = data[1];
		float width = data[2];
		float height = data[3];

		// 找出分数最高的类别（豆子类别）
		int classId = -1;
		float maxConf = 0.0f;
		for (int c = 0; c < numClasses; c++) {
			float conf = data[4 + c];
			if (conf > maxConf) {
				maxConf = conf;
				classId = c;
			}
		}
		if (maxConf >= bean_net.confThreshold) {
			//缩放回原来大小
			int left = (int)((x_center - width / 2) * scaleX);
			int top = (int)((y_center - height / 2) * scaleY);
			int w = (int)(width * scaleX);
			int h = (int)(height * scaleY);

			Rect box(left, top, w, h);
			// 防止框超出画面
			box = box & Rect(0, 0, frame.cols, frame.rows);

			// 过滤太小的框（排除噪点）
			if (box.width > 2 && box.height > 2) {
				tmpBoxes.push_back(box);
				tmpClassIds.push_back(classId);
				tmpConfs.push_back(maxConf);
			}
		}
	}
	// NMS是把重叠的框去掉，只留最好的那个
	vector<int> indices;
	NMSBoxes(tmpBoxes, tmpConfs, bean_net.confThreshold, bean_net.nmsThreshold, indices);

	// 封装结果
	for (int idx : indices) {
		DigitalModel::Result res;
		res.class_id = tmpClassIds[idx];
		res.confidence = tmpConfs[idx];
		res.bbox = tmpBoxes[idx];
		res.center.x = res.bbox.x + res.bbox.width / 2.0f;
		res.center.y = res.bbox.y + res.bbox.height / 2.0f;
		results.push_back(res);
	}

	return results;
	}
};


//--------检测结果结构体--------//
struct DetectionResult
{
	int class_id;			// 类别ID
	cv::Rect bbox;
    float confidence;       // 置信度          
    std::string label;
	cv::Point2f center;

    DetectionResult() 
		: bbox(), confidence(0.0f), class_id(-1), label("") {}
    DetectionResult(const cv::Rect& b, float conf, int cid, const std::string& lbl)
        : bbox(b), confidence(conf), class_id(cid), label(lbl) {
		center.x = b.x + b.width / 2.0f;
		center.y = b.y + b.height / 2.0f;
	}
};

static cv::VideoCapture cap;
static int detect_frame_count = 0;
const int DETECT_INTERVAL = 3; // 每3帧识别一次
static std::vector<DetectionResult> last_results; // 保存上一帧检测结果
static std::vector<DetectionResult> last_results;

static auto last_time = steady_clock::now();
static double fps = 0.0;

//FPS
double calculateFPS() {
	auto current_time = steady_clock::now();
	duration<double> delta = current_time - last_time;
	if (delta.count() > 0) {
		fps = 1.0 / delta.count();
	}
	last_time = current_time;
	return fps;
}

//显示FPS
void drawFPS(Mat& frame) {
	double current_fps = calculateFPS();
	putText(frame, "FPS: " + to_string((int)current_fps),
		Point(10, frame.rows - 30),
		FONT_HERSHEY_SIMPLEX, 0.8,
		Scalar(0, 255, 0), 2);
}


//串口
bool sendSerial(const string& data) {
	HANDLE h = CreateFileA("COM3", GENERIC_WRITE, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
	if (h == INVALID_HANDLE_VALUE) return false;

	DCB dcb;
	memset(&dcb, 0, sizeof(dcb));
	dcb.DCBlength = sizeof(dcb);
	GetCommState(h, &dcb);
	dcb.BaudRate = CBR_115200;
	dcb.ByteSize = 8;
	dcb.Parity = NOPARITY;
	dcb.StopBits = ONESTOPBIT;
	SetCommState(h, &dcb);

	DWORD w;
	WriteFile(h, data.c_str(), data.size(), &w, NULL);
	CloseHandle(h);
	return true;
}

//-------图像采集--------//
Mat capture_camera(DigitalModel& digit_net, BeanModel& bean_net, vector<DetectionResult>& results, bool need_detect){
	Mat frame;
	/*VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "无法打开摄像头" << std::endl;
		exit(1);
	}*/
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	cap >> frame;
	/*cap.release();*/
	if (frame.empty()) return frame;

	drawFPS(frame);
	// ----- 隔帧识别逻辑 ----- //
	detect_frame_count++;
	bool should_detect = need_detect &&
		(detect_frame_count % DETECT_INTERVAL == 0 || last_results.empty());

	if (should_detect) {
		results.clear();

		auto bean_res = bean_net.detect(frame);
		for (auto& r : bean_res) {
			results.emplace_back(r.bbox, r.confidence, r.class_id, "bean");
		}


		auto digit_res = digit_net.detect(frame);
		for (auto& r : digit_res) {
			results.emplace_back(r.bbox, r.confidence, r.class_id, "digit");
		}

		last_results = results;
	}
	else {
		results = last_results;
	}
	//串口发送第一个检测结果
	/*if (!results.empty()) {
		auto& r = results[0];
		string msg = to_string(r.class_id) + ","
			+ to_string((int)r.center.x) + ","
			+ to_string((int)r.center.y) + "\n";
		sendSerial(msg);
	}
	return frame;*/
}

int main() {
	//加载模型
	DigitalModel digit_net("digit_yolo.onnx");
	BeanModel bean_net(BEAN_YOLO); // 这里传入宏定义的字符串
	vector<DetectionResult> results;

	VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "无法打开摄像头" << std::endl;
		exit(1);
	}

	while (true) {
		Mat frame = capture_camera(digit_net, bean_net, results, true);
		if (frame.empty()) break;
		
		// 在图上画框和标签
		for (const auto& r : results) {
			rectangle(frame, r.bbox, Scalar(0, 255, 0), 2);
			putText(frame, r.label + ":" + to_string((int)(r.confidence * 100)) + "%",
				Point(r.bbox.x, r.bbox.y - 10),
				FONT_HERSHEY_SIMPLEX, 0.6,
				Scalar(0, 255, 0), 2);
		}
		imshow("检测结果", frame);
		if (waitKey(1) == 'q') break;
	}

	cap.release();

	destroyAllWindows();
	return 0;
}

