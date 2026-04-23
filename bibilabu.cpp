#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>// 包含OpenCV深度神经网络模块，用于加载ONNX模型实现YOLO推理
#include <iostream>
#include <vector>
#include <string>
#include <chrono>// 时间库，用于计算FPS、处理时间间隔
#include <Windows.h>// Windows系统头文件，提供系统级接口支持
#include <algorithm>// 算法库，提供排序、查找等通用算法
#include <thread>// C++多线程库，用于创建独立的推理线程
#include <mutex>// 互斥锁，保证多线程访问共享数据时安全不冲突
#include <atomic>// 原子变量，线程安全的布尔/数值标记，无需加锁
#include <condition_variable>// 条件变量，用于线程间等待/通知同步

#define BEAN_YOLO "D:/opencv_test/bibilabu/best.onnx"	//YOLO模型的ONNX文件路径

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace chrono;

struct yoloall
{
	Net net;							//神经网络
	int netWidth = 640;					
	int netHeight = 640;
	float confThreshold = 0.5f;			//置信度阈值
	float nmsThreshold = 0.45f;			// 去重多余检测框
	Mat blobCache;						// 图像预处理后的blob数据缓存，避免重复创建
	vector<Rect> boxCache;				// 检测框缓存容器
	vector<int> classIdCache;			// 类别ID缓存容器
	vector<float> confCache;			// 置信度缓存容器
};

//豆子YOLO
class YOLOModel
{
private:
	yoloall all_net;
public:

	struct Result {
		int class_id;			//目标类别编号
		float confidence;		//置信度
		Rect bbox;				//目标矩形框
		Point2f center;			//目标中心点坐标
	};

	YOLOModel(const string& onnx_path) {
		// 从ONNX文件读取并加载YOLO模型
		all_net.net = dnn::readNetFromONNX(onnx_path);
		// 设置推理后端
		all_net.net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
		// 设置推理目标：CPU推理
		all_net.net.setPreferableTarget(dnn::DNN_TARGET_CPU);
		//预分配缓存空间
		all_net.boxCache.reserve(1000);
		all_net.classIdCache.reserve(1000);
		all_net.confCache.reserve(1000);
		cout << "模型加载完成: " << onnx_path << endl;
	}

	// 检测函数
	vector<YOLOModel::Result> detect(const Mat& frame) {
		vector<YOLOModel::Result> results;
		// 图像预处理：转为模型需要的blob格式（归一化+尺寸调整+通道交换）
		blobFromImage(frame, all_net.blobCache, 1.0 / 255.0, Size(all_net.netWidth, all_net.netHeight), Scalar(), true, false);
		// 将预处理后的blob数据输入到神经网络
		all_net.net.setInput(all_net.blobCache);
		// 执行前向推理，得到推理模型输出结果
		Mat output = all_net.net.forward();
		Mat outMat;

		// 3维输出转2维矩阵方便遍历
		if (output.dims == 3) {
			outMat = Mat(output.size[1], output.size[2], CV_32F, output.data);
		}
		else {
			outMat = output;
		}
		if (outMat.rows < outMat.cols) {
			transpose(outMat, outMat);
		}

		int numAnchors = outMat.rows;
		int numAttributes = outMat.cols;
		int numClasses = numAttributes - 4;

		float scaleX = (float)frame.cols / all_net.netWidth;
		float scaleY = (float)frame.rows / all_net.netHeight;

		all_net.boxCache.clear();
		all_net.classIdCache.clear();
		all_net.confCache.clear();
		// 遍历每一个候选框
		for (int i = 0; i < numAnchors; i++) {
			const float* data = outMat.ptr<float>(i);
			float x_center = data[0];
			float y_center = data[1];
			float width = data[2];
			float height = data[3];

			int classId = -1;
			float maxConf = 0.0f;
			// 找到置信度高的
			for (int c = 0; c < numClasses; c++) {
				float conf = data[4 + c];
				if (conf > maxConf) {
					maxConf = conf;
					classId = c;
				}
			}
			// 保留需要的
			if (maxConf >= all_net.confThreshold) {
				int left = (int)((x_center - width / 2) * scaleX);
				int top = (int)((y_center - height / 2) * scaleY);
				int w = (int)(width * scaleX);
				int h = (int)(height * scaleY);

				Rect box(left, top, w, h);
				box = box & Rect(0, 0, frame.cols, frame.rows);

				if (box.width > 2 && box.height > 2) {
					all_net.boxCache.push_back(box);
					all_net.classIdCache.push_back(classId);
					all_net.confCache.push_back(maxConf);
				}
			}
		}
		vector<int> indices;
		// 去除重复的检测框
		NMSBoxes(all_net.boxCache, all_net.confCache, all_net.confThreshold, all_net.nmsThreshold, indices);
		// 遍历NMS筛选后的结果，封装成Result结构体
		for (int idx : indices) {
			YOLOModel::Result res;
			res.class_id = all_net.classIdCache[idx];
			res.confidence = all_net.confCache[idx];
			res.bbox = all_net.boxCache[idx];
			res.center.x = res.bbox.x + res.bbox.width / 2.0f;
			res.center.y = res.bbox.y + res.bbox.height / 2.0f;
			results.push_back(res);
		}
		return results;
	}
};

struct DetectionResult
{
	int class_id;
	cv::Rect bbox;
	float confidence;
	std::string label;
	cv::Point2f center;

	DetectionResult() : bbox(), confidence(0.0f), class_id(-1), label("") {}
	DetectionResult(const cv::Rect& b, float conf, int cid, const std::string& lbl)
		: bbox(b), confidence(conf), class_id(cid), label(lbl) {
		center.x = b.x + b.width / 2.0f;
		center.y = b.y + b.height / 2.0f;
	}
};

// ========== 多线程异步推理 ==========
static cv::VideoCapture cap;
static auto last_time = steady_clock::now();	//上一帧时间
static double fps = 0.0;	//实时帧率

// 线程同步变量
static mutex results_mutex;//保护结果
static mutex frame_mutex;//保护帧
static condition_variable cv_detect;//线程唤醒/休眠：没事干的时候休眠，不占用CPU
static atomic<bool> running{ true };//程序开始开关
static atomic<bool> frame_ready{ false };//新帧就绪标记
static atomic<bool> detect_done{ true };//推理完成标记

// 共享数据
static Mat shared_frame;
static vector<DetectionResult> shared_results;

double calculateFPS() {
	auto current_time = steady_clock::now();
	duration<double> delta = current_time - last_time;
	if (delta.count() > 0) {
		fps = 0.9 * fps + 0.1 * (1.0 / delta.count()); // 平滑FPS
	}
	last_time = current_time;
	return fps;
}

void drawFPS(Mat& frame) {
	putText(frame, "FPS: " + to_string((int)fps),
		Point(10, frame.rows - 30),
		FONT_HERSHEY_SIMPLEX, 0.8,
		Scalar(0, 255, 0), 2);
}

// 推理线程函数
void detect_thread(YOLOModel* all_net) {
	Mat local_frame;
	while (running) {
		// 等待新帧
		{
			unique_lock<mutex> lock(frame_mutex);
			cv_detect.wait(lock, [] { return frame_ready.load() || !running; });
			if (!running) break;
			if (!frame_ready) continue;
			local_frame = shared_frame.clone();
			frame_ready = false;
		}

		// 执行推理
		auto all_res = all_net->detect(local_frame);

		vector<DetectionResult> local_results;
		vector<string> class_names = {};///////////////////////////记得匹配类别名字！！！！！！！！！
		for (auto& r : all_res) {
			string label = class_names[r.class_id];  // 自动匹配名字
			local_results.emplace_back(r.bbox, r.confidence, r.class_id, label);
		}
		
		// 更新结果
		{
			lock_guard<mutex> lock(results_mutex);
			shared_results = move(local_results);
		}
		detect_done = true;
	}
}

int main() {
	YOLOModel all_net(BEAN_YOLO);

	cap.open(0);
	if (!cap.isOpened()) {
		cerr << "无法打开摄像头" << endl;
		return -1;
	}
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	cap.set(cv::CAP_PROP_FPS, 30); // 设置摄像头帧率
	cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // 减少缓冲区大小

	// 启动推理线程
	thread detect_thread_obj(detect_thread, &all_net);

	Mat frame;
	vector<DetectionResult> display_results;

	cout << "按 'q' 退出" << endl;

	while (true) {
		cap >> frame;
		if (frame.empty()) break;

		calculateFPS();

		// 获取最新检测结果
		{
			lock_guard<mutex> lock(results_mutex);
			display_results = shared_results;
		}

		// 画检测结果
		for (const auto& r : display_results) {
			rectangle(frame, r.bbox, Scalar(0, 255, 0), 2);
			putText(frame, r.label + ":" + to_string((int)(r.confidence * 100)) + "%",
				Point(r.bbox.x, r.bbox.y - 10),
				FONT_HERSHEY_SIMPLEX, 0.6,
				Scalar(0, 255, 0), 2);
		}

		drawFPS(frame);
		imshow("检测结果", frame);

		// 发送帧到推理线程
		if (detect_done.load()) {
			{
				lock_guard<mutex> lock(frame_mutex);
				shared_frame = frame.clone();
				frame_ready = true;
			}
			cv_detect.notify_one();
			detect_done = false;
		}

		if (waitKey(1) == 'q') break;
	}

	running = false;
	cv_detect.notify_all();
	detect_thread_obj.join();

	cap.release();
	destroyAllWindows();
	return 0;
}