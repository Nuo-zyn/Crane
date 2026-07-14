#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>// 包含OpenCV深度神经网络模块，用于加载ONNX模型实现YOLO推理
#include <iostream>
#include <vector>
#include <string>
#include <chrono>// 时间库，用于计算FPS、处理时间间隔
#include <algorithm>// 算法库，提供排序、查找等通用算法
#include <thread>// C++多线程库，用于创建独立的推理线程
#include <mutex>// 互斥锁，保证多线程访问共享数据时安全不冲突
#include <atomic>// 原子变量，线程安全的布尔/数值标记，无需加锁
#include <condition_variable>// 条件变量，用于线程间等待/通知同步
#include "Nuo.h"


#define ALL_YOLO "/home/nuo/vscode/crane.onnx"	//YOLO模型的ONNX文件路径

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace chrono;

struct yoloall
{
	Net net;							//神经网络
	int netWidth = 640;
	int netHeight = 640;
	float confThreshold = 0.3f;			//置信度阈值
	float nmsThreshold = 0.45f;			// 检测框去重
	// 注意：以下缓存已移除，改为detect()内局部变量，保证多线程安全
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
		cout << "模型加载完成: " << onnx_path << endl;
	}

	// 检测函数
// 检测函数
vector<YOLOModel::Result> detect(const Mat& frame) {
    vector<YOLOModel::Result> results;
    // 图像预处理：转为模型需要的blob格式（归一化+尺寸调整+通道交换）
    // 使用局部变量，保证多线程安全（每个线程各自独立的blob）
    Mat blob;
    blobFromImage(frame, blob, 1.0 / 255.0, Size(all_net.netWidth, all_net.netHeight), Scalar(), true, false);
    // 将预处理后的blob数据输入到神经网络
    all_net.net.setInput(blob);
    // 执行前向推理，得到推理模型输出结果
    Mat output = all_net.net.forward();
    Mat outMat;

    // ===== 诊断：打印输出形状 =====
    static int diag_count = 0;
    if (diag_count < 3) {
        cout << "[诊断] output.dims=" << output.dims << " sizes=";
        for (int d = 0; d < output.dims; d++) cout << output.size[d] << " ";
        cout << " total=" << output.total() << endl;
    }

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

    if (diag_count < 3) {
        cout << "[诊断] outMat: rows=" << outMat.rows << " cols=" << outMat.cols << endl;
        // 打印前3个候选框的原始数据
        for (int i = 0; i < min(3, outMat.rows); i++) {
            const float* d = outMat.ptr<float>(i);
            cout << "  anchor[" << i << "] x=" << d[0] << " y=" << d[1]
                 << " w=" << d[2] << " h=" << d[3];
            // 打印前3个类别置信度
            int nc = outMat.cols - 4;
            for (int c = 0; c < min(3, nc); c++) cout << " cls" << c << "=" << d[4+c];
            cout << endl;
        }
    }

    int numAnchors = outMat.rows;
    int numAttributes = outMat.cols;
    int numClasses = numAttributes - 4;

    float scaleX = (float)frame.cols / all_net.netWidth;
    float scaleY = (float)frame.rows / all_net.netHeight;

    // 使用临时容器，避免缓存污染
    vector<Rect> tempBoxCache;
    vector<int> tempClassIdCache;
    vector<float> tempConfCache;
    
    tempBoxCache.reserve(1000);
    tempClassIdCache.reserve(1000);
    tempConfCache.reserve(1000);
    
    // 遍历每一个候选框
    float globalMaxConf = 0.0f;
    for (int i = 0; i < numAnchors; i++) {
        const float* data = outMat.ptr<float>(i);
        float x_center = data[0];
        float y_center = data[1];
        float width = data[2];
        float height = data[3];

        int classId = -1;
        float maxConf = 0.0f;
        // 找到置信度最高的类别
        for (int c = 0; c < numClasses; c++) {
            float conf = data[4 + c];
            if (conf > maxConf) {
                maxConf = conf;
                classId = c;
            }
        }

        if (maxConf > globalMaxConf) globalMaxConf = maxConf;

        // 保留满足置信度阈值的
        if (maxConf >= all_net.confThreshold) {
            int left = (int)((x_center - width / 2) * scaleX);
            int top = (int)((y_center - height / 2) * scaleY);
            int w = (int)(width * scaleX);
            int h = (int)(height * scaleY);

            Rect box(left, top, w, h);
            box = box & Rect(0, 0, frame.cols, frame.rows);

            // 过滤：最小10x10，最大不超过画面80%（排除黑边误检）
            int maxW = (int)(frame.cols * 0.8f);
            int maxH = (int)(frame.rows * 0.8f);
            if (box.width > 10 && box.height > 10 && box.width < maxW && box.height < maxH) {
                tempBoxCache.push_back(box);
                tempClassIdCache.push_back(classId);
                tempConfCache.push_back(maxConf);
            }
        }
    }

    // ===== 诊断：通过阈值的检测数量 =====
    if (diag_count <= 5) {
        cout << "[诊断] 全局最高置信度=" << globalMaxConf
             << "  通过阈值(" << all_net.confThreshold << ")的检测数=" << tempBoxCache.size() << endl;
        // 打印前3个通过阈值的检测
        for (int i = 0; i < min(3, (int)tempBoxCache.size()); i++) {
            cout << "  det[" << i << "] box=" << tempBoxCache[i]
                 << " conf=" << tempConfCache[i]
                 << " class=" << tempClassIdCache[i] << endl;
        }
        diag_count++;
    }
    
    // 确保三个容器大小一致
    if (tempBoxCache.size() != tempConfCache.size()) {
        cerr << "警告：boxCache大小(" << tempBoxCache.size() 
             << ") 与 confCache大小(" << tempConfCache.size() << ") 不一致！" << endl;
        // 取最小的大小进行截断
        size_t minSize = min(tempBoxCache.size(), tempConfCache.size());
        tempBoxCache.resize(minSize);
        tempClassIdCache.resize(minSize);
        tempConfCache.resize(minSize);
    }
    
    vector<int> indices;
    // 只有当有检测结果时才执行NMS
    if (!tempBoxCache.empty() && tempBoxCache.size() == tempConfCache.size()) {
        NMSBoxes(tempBoxCache, tempConfCache, all_net.confThreshold, all_net.nmsThreshold, indices);
    }
    
    // 遍历NMS筛选后的结果，封装成Result结构体
    for (int idx : indices) {
        if (idx >= 0 && idx < (int)tempBoxCache.size()) {
            YOLOModel::Result res;
            res.class_id = tempClassIdCache[idx];
            res.confidence = tempConfCache[idx];
            res.bbox = tempBoxCache[idx];
            res.center.x = res.bbox.x + res.bbox.width / 2.0f;
            res.center.y = res.bbox.y + res.bbox.height / 2.0f;
            results.push_back(res);
        }
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

//=====【新增改动区域1：封装CameraDetector摄像头类，复用你原来整套单路摄像头逻辑】=====
// 将你原本main里的采集/推理/绘图/线程同步全部封装进此类，每个摄像头独立实例
class CameraDetector
{
private:
    // 摄像头采集对象（原全局cap改为类内私有成员，每个摄像头独立）
    VideoCapture cap;
    // 线程同步变量（原全局互斥锁/条件变量改为类内私有，两路完全隔离不冲突）
    mutex results_mutex;
    mutex frame_mutex;
    condition_variable cv_detect;
    atomic<bool> running{ true };
    atomic<bool> frame_ready{ false };
    atomic<bool> detect_done{ true };
    // 帧/推理结果缓存（原全局共享帧改为类内私有，两路互不覆盖）
    Mat shared_frame;
    vector<DetectionResult> shared_results;
    // 推理线程对象（每个摄像头独立推理线程）
    thread detect_thread_obj;
    // 每个摄像头独立持有YOLO模型实例（cv::dnn::Net非线程安全，不能共享）
    YOLOModel yolo_model;
    // 窗口名称，区分两路摄像头画面
    string win_name;
    // FPS计时（原全局last_time改为类内私有，两路独立计算FPS）
    steady_clock::time_point last_time;
    double fps = 0.0;
    // 类别名称列表（移至类内，避免全局重复定义）
    vector<string> class_names = { "box_1", "box_2", "box_3", "box_4", "box_5", "white_beans", "yellow_beans", "green_beans" };

    //=====【改动：原全局detect_thread函数移入类内，逻辑完全不变】=====
    void detect_thread() {
        Mat local_frame;
        while (running) {
            {
                unique_lock<mutex> lock(frame_mutex);
                cv_detect.wait(lock, [this] { return frame_ready.load() || !running; });
                if (!running) break;
                if (!frame_ready) continue;
                local_frame = shared_frame.clone();
                frame_ready = false;
            }

            auto all_res = yolo_model.detect(local_frame);
            vector<DetectionResult> local_results;
            for (auto& r : all_res) {
                // 防止类别ID越界
                string label = (r.class_id >= 0 && r.class_id < (int)class_names.size())
                    ? class_names[r.class_id]
                    : "unknown_" + to_string(r.class_id);
                local_results.emplace_back(r.bbox, r.confidence, r.class_id, label);
            }

            {
                lock_guard<mutex> lock(results_mutex);
                shared_results = move(local_results);
            }
            detect_done = true;
        }
    }
    //=====【改动：原全局calculateFPS移入类内，逻辑完全不变】=====
    double calculateFPS() {
        auto current_time = steady_clock::now();
        duration<double> delta = current_time - last_time;
        if (delta.count() > 0) {
            fps = 0.9 * fps + 0.1 * (1.0 / delta.count());
        }
        last_time = current_time;
        return fps;
    }

    //=====【改动：原全局drawFPS移入类内，逻辑完全不变】=====
    void drawFPS(Mat& frame) {
        putText(frame, "FPS: " + to_string((int)fps),
            Point(10, frame.rows - 30),
            FONT_HERSHEY_SIMPLEX, 0.8,
            Scalar(0, 255, 0), 2);
    }

public:
    //=====【新增：构造函数，统一初始化摄像头参数，和你原有配置完全一致】=====
    CameraDetector(int cam_id, string win_title, const string& onnx_path)
        : win_name(win_title), yolo_model(onnx_path)
    {
        // 打开摄像头，参数完全沿用你原来的配置
        cap.open(cam_id,CAP_V4L2);

        if (!cap.isOpened()) {
        cerr << "❌ 错误：无法打开摄像头 " << cam_id << " (" << win_title << ")" << endl;
        cerr << "   可能原因：" << endl;
        cerr << "   1. 摄像头索引 " << cam_id << " 不存在" << endl;
        cerr << "   2. 摄像头被其他程序占用" << endl;
        cerr << "   3. 没有访问权限(Linux可能需要: sudo chmod 666 /dev/video*)" << endl;
        return;  // 直接返回，不继续初始化
    } else {
        cout << "✅ 成功打开摄像头 " << cam_id << " (" << win_title << ")" << endl;
    }

        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(CAP_PROP_FPS, 30);
        cap.set(CAP_PROP_BUFFERSIZE, 1);
        last_time = steady_clock::now();
        // 启动该摄像头专属推理线程
        detect_thread_obj = thread(&CameraDetector::detect_thread, this);
    }

    //=====【新增：析构函数，自动释放线程、摄像头资源】=====
    ~CameraDetector() {
        running = false;
        cv_detect.notify_all();
        if (detect_thread_obj.joinable())
            detect_thread_obj.join();
        cap.release();
    }

    //=====【新增：判断摄像头是否正常打开】=====
    bool isOpened() {
        bool opened = cap.isOpened();

        if (!opened) {
        cerr << "⚠️ 警告：摄像头 " << win_name << " 未打开" << endl;
        }

        return opened;
    }

    //=====【核心改动：runOneFrame函数，完整复用你原来main循环里的摄像头业务代码】=====
    // 你提供的这段代码全部封装在此函数，无任何逻辑修改：
    // cap >> frame; if空判断、FPS计算、读取推理结果、绘制框、推送帧到推理线程全部保留
    bool runOneFrame(Mat& out_frame) {
        // 1. 采集帧（原代码逻辑）
        cap >> out_frame;
            if (out_frame.empty()) {
        // ===== 添加错误提示 =====
        cerr << "❌ " << win_name << " 获取到空帧，摄像头可能已断开" << endl;
        // ===== 错误提示结束 =====
        return false;
    }

        // 2. 更新FPS（原代码逻辑）
        calculateFPS();

        // 3. 获取最新检测结果（原代码锁逻辑完全保留）
        vector<DetectionResult> display_results;
        {
            lock_guard<mutex> lock(results_mutex);
            display_results = shared_results;
        }

        // 4. 绘制检测框、置信文字（原绘图逻辑完全保留）
        for (const auto& r : display_results) {
            rectangle(out_frame, r.bbox, Scalar(0, 255, 0), 2);
            putText(out_frame, r.label + ":" + to_string((int)(r.confidence * 100)) + "%",
                Point(r.bbox.x, r.bbox.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.6,
                Scalar(0, 255, 0), 2);
        }

        // 5. 绘制FPS（原逻辑）
        drawFPS(out_frame);
        // 新增：绘制摄像头窗口标题，区分两路画面
        putText(out_frame, win_name, Point(10, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);

        // 6. 推送帧到推理线程（你原有的线程通知逻辑完整保留）
        if (detect_done.load()) {
            {
                lock_guard<mutex> lock(frame_mutex);
                shared_frame = out_frame.clone();
                frame_ready = true;
            }
            cv_detect.notify_one();
            detect_done = false;
        }

        
        return true;
    }
};
int main() {
    // ===== 第一步：列出所有可用摄像头 =====
    cout << "========== 扫描可用摄像头 ==========" << endl;
    vector<int> available_cams;
    for (int i = 0; i < 10; i++) {
        VideoCapture testCap(i);
        if (testCap.isOpened()) {
            available_cams.push_back(i);
            cout << "✅ 摄像头索引 " << i << " 可用" << endl;
            testCap.release();
        } else {
            cout << "❌ 摄像头索引 " << i << " 不可用" << endl;
        }
    }
    cout << "====================================" << endl;
    
    if (available_cams.empty()) {
        cerr << "❌ 致命错误：没有找到任何摄像头！" << endl;
        cerr << "请检查：" << endl;
        cerr << "  1. 摄像头是否已连接" << endl;
        cerr << "  2. 驱动是否正常安装" << endl;
        cerr << "  3. 权限问题(运行: ls -la /dev/video*)" << endl;
        return -1;
    }
    
    cout << "找到 " << available_cams.size() << " 个摄像头" << endl;


    // ===== 根据实际情况选择摄像头（优先使用外置摄像头） =====
    int cam0_id, cam1_id;

    // 排序摄像头索引：外置摄像头通常索引号较大，优先使用
    sort(available_cams.begin(), available_cams.end(), greater<int>());
    cout << "摄像头优先级排序（外置优先）：";
    for (int cam : available_cams) cout << cam << " ";
    cout << endl;

    if (available_cams.size() >= 2) {
        // 索引最大的给 Camera 0（外置摄像头优先）
        cam0_id = available_cams[0];
        cam1_id = available_cams[1];
        cout << "Camera 0 (外置) -> /dev/video" << cam0_id << endl;
        cout << "Camera 1 (内置) -> /dev/video" << cam1_id << endl;
    } else if (available_cams.size() == 1) {
        cam0_id = available_cams[0];
        cam1_id = available_cams[0];  // 同一个摄像头，两个窗口
        cout << "⚠️ 只有一个摄像头，将显示两个相同画面" << endl;
    } else {
        cerr << "❌ 没有找到足够摄像头" << endl;
        return -1;
    }

    //////////////
    // Nuonuo serial;

    // if (!serial.open("/dev/ttyUSB0", 115200)) {
    //     std::cerr << "串口打开失败" << std::endl;
    //     return 1;
    // } else {
    //         std::cerr << "串口打开失败，仅终端输出" << std::endl;
    //         }
    //////////////

    // 实例化两路摄像头（各自独立加载模型，避免cv::dnn::Net线程不安全）
    cout << "正在加载模型..." << endl;
    CameraDetector cam0(cam0_id, "Camera 0", ALL_YOLO);
    CameraDetector cam1(cam1_id, "Camera 1", ALL_YOLO);

    // ===== 详细检查每个摄像头的状态 =====
    cout << "\n========== 检查摄像头状态 ==========" << endl;
    bool cam0_ok = cam0.isOpened();
    bool cam1_ok = cam1.isOpened();
    
    if (!cam0_ok) {
        cerr << "❌ Camera 0 初始化失败！" << endl;
    }
    if (!cam1_ok) {
        cerr << "❌ Camera 1 初始化失败！" << endl;
    }
    if (!cam0_ok || !cam1_ok) {
        cerr << "❌ 至少一个摄像头打开失败！" << endl;
        cerr << "建议：" << endl;
        cerr << "  1. 尝试修改摄像头索引" << endl;
        cerr << "  2. 检查物理连接" << endl;
        cerr << "  3. 运行: sudo chmod 666 /dev/video*" << endl;
        return -1;
    }
    cout << "✅ 所有摄像头就绪" << endl;
    cout << "===================================\n" << endl;
    // ===== 检查结束 =====
    
    Mat frame0, frame1;

    cout << "双摄像头独立窗口显示，按 q 退出" << endl;


    // 创建两个窗口
    namedWindow("Camera 0", WINDOW_NORMAL);
    namedWindow("Camera 1", WINDOW_NORMAL);


    // 设置窗口大小
    resizeWindow("Camera 0", 640, 480);
    resizeWindow("Camera 1", 640, 480);

    // 两个窗口位置分开
    moveWindow("Camera 0", 50, 50);
    moveWindow("Camera 1", 750, 50);

    while (true)
    {
        // 摄像头0
        bool ok0 = cam0.runOneFrame(frame0);

        // 摄像头1
        bool ok1 = cam1.runOneFrame(frame1);

        if (!ok0 || !ok1)
        {
            cerr << "摄像头读取失败" << endl;
            break;
        }

        // 分别显示
        imshow("Camera 0", frame0);

        imshow("Camera 1", frame1);

        // 任意窗口按q退出
        char key = waitKey(1);

        if(key=='q' || key=='Q')
        {
            break;
        }
    }
    
    destroyAllWindows();
    cout << "程序正常退出" << endl;
    return 0;
}