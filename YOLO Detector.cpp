/*
 * =============================================================================
 * YOLO检测器 - 完整代码
 *
 * 功能：
 *   - 摄像头实时目标检测
 *   - 支持数字识别和豆子分类
 *   - 串口发送识别结果
 *
 * 协议：
 *   - 帧格式：[AA] [类型] [数据...] [BB]
 *   - 类型：01=数字，02=豆子
 *   - 数据：ASCII字符（数字1~5，豆子Y/G/W）
 *
 * 使用：
 *   1. 编译：g++ -std=c++17 main.cpp -o detector `pkg-config --cflags --libs opencv4 openvino` -lpthread
 *   2. 配置config.txt
 *   3. 运行：./detector
 *
 * =============================================================================
 */

// =============================================================================
// 依赖头文件
// =============================================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <cstdint>
#include <cerrno>

// OpenCV
#include <opencv2/opencv.hpp>
// OpenVINO
#include <openvino/openvino.hpp>

// Linux串口
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace fs = std::filesystem;

// =============================================================================
// 全局变量
// =============================================================================
std::atomic<bool> g_exit(false);

void signalHandler(int) { g_exit = true; }


// =============================================================================
// 第一部分：配置管理
// =============================================================================

/*
 * Config - 配置管理类
 *
 * 从config.txt文件加载配置参数
 */
struct Config {
    // 模型路径
    std::string model_path = "models/best.onnx";
    std::string classes_file = "";

    // 摄像头索引（0=默认摄像头，1=第二个摄像头）
    int digit_camera = 0;
    int bean_camera = 1;

    // 检测参数
    float confidence_threshold = 0.5f;  // 置信度阈值
    float nms_threshold = 0.5f;         // NMS阈值
    int detection_interval = 10;         // 每隔几帧检测一次

    // 串口配置（留空则不使用串口）
    std::string serial_port = "/dev/ttyUSB0";
    int serial_baudrate = 115200;

    // OpenVINO性能
    int num_infer_requests = 2;

    // 从文件加载配置
    bool load(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cout << "配置文件未找到: " << path << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // 跳过注释和空行
            if (line.empty() || line[0] == '#') continue;

            // 解析 key=value
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));
            if (val.empty()) continue;

            // 赋值
            if (key == "model_path")           model_path = val;
            else if (key == "classes_file")   classes_file = val;
            else if (key == "digit_camera")   digit_camera = std::stoi(val);
            else if (key == "bean_camera")    bean_camera = std::stoi(val);
            else if (key == "confidence_threshold") confidence_threshold = std::stof(val);
            else if (key == "nms_threshold")   nms_threshold = std::stof(val);
            else if (key == "detection_interval") detection_interval = std::stoi(val);
            else if (key == "num_infer_requests") num_infer_requests = std::stoi(val);
            else if (key == "serial_port")    serial_port = val;
            else if (key == "serial_baudrate") serial_baudrate = std::stoi(val);
        }
        file.close();
        return true;
    }

    // 打印当前配置
    void print() const {
        std::cout << "=== 配置 ===" << std::endl;
        std::cout << "  模型: " << model_path << std::endl;
        std::cout << "  数字相机: " << digit_camera << std::endl;
        std::cout << "  豆子相机: " << bean_camera << std::endl;
        std::cout << "  串口: " << serial_port << " @ " << serial_baudrate << std::endl;
        std::cout << "===========" << std::endl;
    }

private:
    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }
};


// =============================================================================
// 第二部分：检测结果结构
// =============================================================================

/*
 * DetectionResult - 单个检测结果
 */
struct DetectionResult {
    int class_id;              // 类别ID
    std::string class_name;     // 类别名称
    float confidence;           // 置信度
    cv::Rect bbox;             // 边界框
};


// =============================================================================
// 第三部分：串口通讯
// =============================================================================

/*
 * SerialPort - 串口通讯类
 *
 * 功能：打开串口、发送数据
 * 协议：[AA] [类型] [数据...] [BB]
 */
class SerialPort {
public:
    SerialPort() : handle_(nullptr) {}
    ~SerialPort() { close(); }

    /*
     * 打开串口
     *
     * 参数：
     *   port     - 设备名，如 "/dev/ttyUSB0"
     *   baudrate - 波特率，默认115200
     */
    bool open(const std::string& port, int baudrate = 115200) {
        // 打开设备文件
        // O_RDWR = 读写模式
        // O_NOCTTY = 不作为控制终端
        // O_NONBLOCK = 非阻塞
        handle_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (handle_ < 0) {
            std::cerr << "串口打开失败: " << port << std::endl;
            return false;
        }

        // 获取串口配置
        struct termios tty;
        tcgetattr(handle_, &tty);

        // 设置波特率
        speed_t spd = B115200;
        if (baudrate == 9600) spd = B9600;
        else if (baudrate == 38400) spd = B38400;
        else if (baudrate == 57600) spd = B57600;
        cfsetospeed(&tty, spd);
        cfsetispeed(&tty, spd);

        // 8N1配置
        tty.c_cflag = CS8 | CLOCAL | CREAD;  // 8数据位、无校验、1停止位
        tty.c_iflag = 0;  // 原始输入模式
        tty.c_oflag = 0;  // 原始输出模式
        tty.c_lflag = 0;  // 不处理字符

        // 应用配置
        tcsetattr(handle_, TCSANOW, &tty);

        std::cout << "串口已打开: " << port << " @ " << baudrate << std::endl;
        return true;
    }

    // 关闭串口
    void close() {
        if (handle_ >= 0) {
            ::close(handle_);
            handle_ = -1;
        }
    }

    // 是否已打开
    bool isOpen() const { return handle_ >= 0; }

    /*
     * 发送数据
     *
     * 参数：
     *   data - 字节数据
     *   len  - 数据长度
     */
    bool write(const uint8_t* data, size_t len) {
        if (!isOpen()) return false;

        // 加锁保证线程安全
        std::lock_guard<std::mutex> lock(write_mutex_);

        // 循环发送直到全部发完
        size_t total = 0;
        while (total < len) {
            ssize_t n = ::write(handle_, data + total, len - total);
            if (n > 0) {
                total += n;
            } else if (n < 0) {
                // EAGAIN/EWOULDBLOCK是正常情况，继续重试
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    return false;
                }
            }
        }
        return true;
    }

    // 重载：发送vector数据
    bool write(const std::vector<uint8_t>& data) {
        return write(data.data(), data.size());
    }

private:
    int handle_;                  // 文件描述符
    std::mutex write_mutex_;     // 写入锁
};


// =============================================================================
// 第四部分：摄像头采集
// =============================================================================

/*
 * Camera - 摄像头采集类
 *
 * 功能：独立线程采集摄像头画面，提供最新帧
 */
class Camera {
public:
    Camera() {}
    ~Camera() { stop(); }

    /*
     * 打开摄像头
     *
     * 参数：
     *   source - 摄像头索引（0=第一个摄像头）
     */
    bool open(int source = 0) {
        // 设置采集参数
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);  // 缓冲区设为1减少延迟
        cap_.set(cv::CAP_PROP_FPS, 30);
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        // 打开摄像头
        if (!cap_.open(source, cv::CAP_V4L2)) {
            std::cerr << "无法打开摄像头 " << source << std::endl;
            return false;
        }

        // 获取实际参数
        width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps_ = cap_.get(cv::CAP_PROP_FPS);
        if (fps_ <= 0) fps_ = 30.0;

        std::cout << "摄像头已打开: " << width_ << "x" << height_ << " @ " << fps_ << "fps" << std::endl;
        return true;
    }

    // 启动采集线程
    void start() {
        if (running_) return;
        running_ = true;
        capture_thread_ = std::thread(&Camera::captureLoop, this);
    }

    // 停止采集
    void stop() {
        running_ = false;
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }
    }

    /*
     * 获取最新一帧
     *
     * 参数：
     *   frame - 输出图像
     *
     * 返回：
     *   true  成功获取帧
     *   false 未获取到帧
     */
    bool getLatestFrame(cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (frame_available_) {
            latest_frame_.copyTo(frame);
            frame_available_ = false;
            return true;
        }
        return false;
    }

private:
    // 采集循环（独立线程运行）
    void captureLoop() {
        cv::Mat frame;
        while (running_) {
            if (!cap_.read(frame) || frame.empty()) {
                break;
            }
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                frame.copyTo(latest_frame_);
                frame_available_ = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        running_ = false;
    }

private:
    cv::VideoCapture cap_;           // 视频捕获对象
    std::thread capture_thread_;     // 采集线程
    std::atomic<bool> running_{false};

    std::mutex frame_mutex_;        // 帧数据锁
    cv::Mat latest_frame_;           // 最新帧
    bool frame_available_{false};    // 是否有新帧

    int width_{0}, height_{0};       // 分辨率
    double fps_{0.0};                // 帧率
};


// =============================================================================
// 第五部分：YOLO检测器
// =============================================================================

/*
 * Detector - YOLO检测器
 *
 * 功能：使用OpenVINO进行目标检测
 */
class Detector {
public:
    Detector() : input_width_(0), input_height_(0), input_channels_(0), is_initialized_(false) {}

    /*
     * 初始化检测器
     *
     * 参数：
     *   model_xml         - 模型XML文件路径
     *   model_bin         - 模型BIN文件路径（IR格式）
     *   classes_file      - 类别名称文件
     *   device            - 运行设备，默认CPU
     *   num_infer_requests - 推理并发数
     */
    bool init(const std::string& model_xml, const std::string& model_bin,
              const std::string& classes_file, const std::string& device = "CPU",
              int num_infer_requests = 2) {
        try {
            // 加载类别名称
            if (!loadClassNames(classes_file)) return false;

            // 判断模型格式
            bool is_onnx = false;
            size_t dot_pos = model_xml.find_last_of('.');
            if (dot_pos != std::string::npos) {
                std::string ext = model_xml.substr(dot_pos);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".onnx") is_onnx = true;
            }

            // 加载模型
            std::cout << "加载模型: " << model_xml << std::endl;
            ov::Core core;
            model_ = core.read_model(model_xml);

            // 获取输入尺寸
            ov::Shape input_shape = model_->input().get_shape();
            input_width_  = static_cast<int>(input_shape[3]);
            input_height_ = static_cast<int>(input_shape[2]);
            input_channels_ = static_cast<int>(input_shape[1]);
            std::cout << "输入尺寸: " << input_width_ << "x" << input_height_ << "x" << input_channels_ << std::endl;

            // 配置推理性能
            ov::AnyMap properties;
            properties[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
            properties[ov::hint::num_requests.name()] = num_infer_requests;

            // 编译模型
            compiled_model_ = core.compile_model(model_, device, properties);

            // 创建双推理请求（流水线）
            infer_request_[0] = compiled_model_.create_infer_request();
            infer_request_[1] = compiled_model_.create_infer_request();

            is_initialized_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "检测器初始化失败: " << e.what() << std::endl;
            return false;
        }
    }

    /*
     * 异步推理：提交推理请求
     *
     * 参数：
     *   image - 输入图像
     */
    void submitAsync(const cv::Mat& image) {
        ov::Tensor input_tensor = infer_request_[active_request_].get_input_tensor();
        preprocess(image, input_tensor);
        infer_request_[active_request_].start_async();
    }

    /*
     * 异步推理：等待并获取结果
     *
     * 参数：
     *   confidence_threshold - 置信度阈值
     *   nms_threshold       - NMS阈值
     *   orig_width         - 原始图像宽度
     *   orig_height        - 原始图像高度
     */
    std::vector<DetectionResult> waitAsync(float confidence_threshold,
                                           float nms_threshold,
                                           int orig_width, int orig_height) {
        infer_request_[active_request_].wait();
        ov::Tensor output_tensor = infer_request_[active_request_].get_output_tensor();
        auto results = postprocess(output_tensor, confidence_threshold, nms_threshold,
                                  orig_width, orig_height);
        active_request_ = 1 - active_request_;  // 切换到另一个请求
        return results;
    }

private:
    // 加载类别名称文件
    bool loadClassNames(const std::string& classes_file) {
        std::ifstream file(classes_file);
        if (!file.is_open()) {
            std::cerr << "无法打开类别文件: " << classes_file << std::endl;
            return false;
        }

        class_names_.clear();
        std::string line;
        int max_id = -1;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int id; std::string name;
            if (iss >> id >> name) { if (id > max_id) max_id = id; }
        }
        if (max_id < 0) return false;

        class_names_.resize(max_id + 1);
        file.clear();
        file.seekg(0);
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int id; std::string name;
            if (iss >> id >> name) class_names_[id] = name;
        }
        file.close();

        std::cout << "已加载 " << class_names_.size() << " 个类别" << std::endl;
        return true;
    }

    // 图像预处理
    void preprocess(const cv::Mat& image, ov::Tensor& input_tensor) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_width_, input_height_));

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat float_rgb;
        rgb.convertTo(float_rgb, CV_32F, 1.0 / 255.0);

        // 分离RGB通道并填充到Tensor
        float* dst = input_tensor.data<float>();
        const int hw = input_width_ * input_height_;
        const float* src = reinterpret_cast<const float*>(float_rgb.data);

        for (int i = 0; i < hw; ++i) {
            dst[i] = src[i * 3];           // R
            dst[hw + i] = src[i * 3 + 1]; // G
            dst[2 * hw + i] = src[i * 3 + 2]; // B
        }
    }

    // 后处理：解析输出
    std::vector<DetectionResult> postprocess(const ov::Tensor& output_tensor,
                                            float confidence_threshold,
                                            float nms_threshold,
                                            int orig_width, int orig_height) {
        std::vector<DetectionResult> detections;

        ov::Shape shape = output_tensor.get_shape();
        if (shape.size() != 3) return detections;

        size_t features = shape[1];
        size_t anchors = shape[2];
        const float* data = output_tensor.data<const float>();

        float scale_x = static_cast<float>(orig_width) / input_width_;
        float scale_y = static_cast<float>(orig_height) / input_height_;

        // 输出格式：[cx, cy, w, h, class1_score, class2_score, ...]
        size_t num_classes = features - 4;
        for (size_t a = 0; a < anchors; ++a) {
            float cx = data[a];
            float cy = data[anchors + a];
            float w = data[2 * anchors + a];
            float h = data[3 * anchors + a];

            // 找最大置信度的类别
            float max_score = 0.0f;
            int max_class = -1;
            for (size_t c = 0; c < num_classes; ++c) {
                float score = data[(4 + c) * anchors + a];
                if (score > max_score) { max_score = score; max_class = static_cast<int>(c); }
            }

            // 检查置信度
            if (max_score < confidence_threshold) continue;
            if (max_class < 0 || max_class >= static_cast<int>(class_names_.size())) continue;

            // 计算边界框
            float x1 = (cx - w * 0.5f) * scale_x;
            float y1 = (cy - h * 0.5f) * scale_y;
            float x2 = (cx + w * 0.5f) * scale_x;
            float y2 = (cy + h * 0.5f) * scale_y;

            // 限制在图像范围内
            x1 = std::max(0.0f, x1); y1 = std::max(0.0f, y1);
            x2 = std::min(static_cast<float>(orig_width), x2);
            y2 = std::min(static_cast<float>(orig_height), y2);

            float bw = x2 - x1, bh = y2 - y1;
            if (bw <= 0 || bh <= 0) continue;

            DetectionResult det;
            det.class_id = max_class;
            det.class_name = class_names_[max_class];
            det.confidence = max_score;
            det.bbox = cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                               static_cast<int>(bw), static_cast<int>(bh));
            detections.push_back(det);
        }

        // NMS去重
        nms(detections, nms_threshold);
        return detections;
    }

    // NMS非极大值抑制
    void nms(std::vector<DetectionResult>& detections, float threshold) {
        if (detections.empty()) return;
        std::sort(detections.begin(), detections.end(),
                  [](const DetectionResult& a, const DetectionResult& b) { return a.confidence > b.confidence; });

        std::vector<bool> keep(detections.size(), true);
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!keep[i]) continue;
            for (size_t j = i + 1; j < detections.size(); ++j) {
                if (!keep[j]) continue;
                cv::Rect inter = detections[i].bbox & detections[j].bbox;
                float iou = static_cast<float>(inter.area()) /
                           static_cast<float>(detections[i].bbox.area() + detections[j].bbox.area() - inter.area());
                if (iou > threshold) keep[j] = false;
            }
        }

        size_t write = 0;
        for (size_t i = 0; i < detections.size(); ++i)
            if (keep[i]) detections[write++] = detections[i];
        detections.resize(write);
    }

private:
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_[2];  // 双请求流水线
    int active_request_{0};

    std::vector<std::string> class_names_;
    int input_width_, input_height_, input_channels_;
    bool is_initialized_;
};


// =============================================================================
// 第六部分：检测流水线（核心逻辑）
// =============================================================================

/*
 * Pipeline - 检测流水线
 *
 * 功能：整合摄像头、检测器、串口，实现实时检测和发送
 */
class Pipeline {
public:
    /*
     * 构造函数
     *
     * 参数：
     *   name                 - 流水线名称（用于显示）
     *   camera_index         - 摄像头索引
     *   model_xml            - 模型路径
     *   model_bin            - 模型bin路径
     *   classes_file         - 类别文件
     *   confidence_threshold - 置信度阈值
     *   nms_threshold        - NMS阈值
     *   detection_interval   - 检测间隔（帧数）
     */
    Pipeline(const std::string& name, int camera_index,
            const std::string& model_xml, const std::string& model_bin,
            const std::string& classes_file,
            float confidence_threshold = 0.5f,
            float nms_threshold = 0.5f,
            int detection_interval = 10)
        : name_(name),
          confidence_threshold_(confidence_threshold),
          nms_threshold_(nms_threshold),
          detection_interval_(detection_interval) {

        // 判断是数字相机还是豆子相机
        is_digit_ = (name.find("数字") != std::string::npos);

        // 打开摄像头
        if (!camera_.open(camera_index)) {
            std::cerr << "[" << name_ << "] 无法打开摄像头 " << camera_index << std::endl;
            return;
        }

        // 初始化检测器
        if (!detector_.init(model_xml, model_bin, classes_file, "CPU", 2)) {
            std::cerr << "[" << name_ << "] 检测器初始化失败" << std::endl;
            return;
        }

        initialized_ = true;
        std::cout << "[" << name_ << "] 流水线就绪" << std::endl;
    }

    ~Pipeline() { stop(); }

    // 设置串口
    void setSerial(SerialPort* serial) { serial_ = serial; }

    // 启动流水线
    void start() {
        if (running_ || !initialized_) return;
        camera_.start();
        running_ = true;
        fps_last_time_ = std::chrono::steady_clock::now();
        processing_thread_ = std::thread(&Pipeline::processingLoop, this);
        std::cout << "[" << name_ << "] 已启动" << std::endl;
    }

    // 停止流水线
    void stop() {
        running_ = false;
        if (processing_thread_.joinable()) processing_thread_.join();
        camera_.stop();
    }

    // 获取最新输出帧
    bool getLatestOutput(cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(output_mutex_);
        if (output_available_) {
            latest_output_.copyTo(frame);
            output_available_ = false;
            return true;
        }
        return false;
    }

    // 是否已初始化
    bool isInitialized() const { return initialized_; }

private:
    /*
     * 处理循环
     *
     * 主循环：获取帧 -> 检测 -> 发送 -> 显示
     */
    void processingLoop() {
        cv::Mat frame;
        while (running_) {
            // 第1步：获取摄像头帧
            if (!camera_.getLatestFrame(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            frame_counter_++;
            fps_frame_counter_++;

            // 计算FPS
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - fps_last_time_).count();
            if (elapsed >= 1.0) {
                current_fps_ = fps_frame_counter_ / elapsed;
                fps_frame_counter_ = 0;
                fps_last_time_ = now;
            }

            // 第2步：目标检测（每隔几帧检测一次）
            bool is_detect_frame = (frame_counter_ % detection_interval_ == 0);
            std::vector<DetectionResult> detections;

            if (is_detect_frame) {
                auto t_start = std::chrono::steady_clock::now();

                // 异步推理
                detector_.submitAsync(frame);
                detections = detector_.waitAsync(confidence_threshold_, nms_threshold_,
                                                  frame.cols, frame.rows);

                auto t_end = std::chrono::steady_clock::now();
                double infer_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    last_detections_ = detections;
                }

                // 打印检测结果
                std::cout << "[" << name_ << "] F" << frame_counter_
                         << " | " << std::fixed << std::setprecision(1) << infer_ms << "ms"
                         << " | FPS:" << std::setprecision(1) << current_fps_
                         << " | " << detections.size() << "个目标" << std::endl;

                for (const auto& det : detections) {
                    std::cout << "  -> " << det.class_name
                             << " " << static_cast<int>(det.confidence * 100) << "%"
                             << " [" << det.bbox.x << "," << det.bbox.y
                             << "," << det.bbox.width << "," << det.bbox.height << "]" << std::endl;
                }

                // 第3步：串口发送检测结果 
                sendResults(detections);
            } else {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                detections = last_detections_;
            }

            // 第4步：画框并显示
            drawDetections(frame, detections);

            // 显示FPS
            std::string tag = is_digit_ ? "Digit" : "Bean";
            std::string info = tag + " FPS:" + std::to_string(static_cast<int>(current_fps_));
            cv::putText(frame, info, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            // 推送输出帧
            {
                std::lock_guard<std::mutex> lock(output_mutex_);
                frame.copyTo(latest_output_);
                output_available_ = true;
            }
        }
    }

    /*
     * 串口发送检测结果 ★
     *
     * 帧格式：[AA] [类型] [数据...] [BB]
     *
     * 类型：01=数字，02=豆子
     * 数据：ASCII字符（数字1~5，豆子Y/G/W）
     *
     * 示例：
     *   数字 2,3,1 -> AA 01 32 33 31 BB
     *   豆子 Y,G   -> AA 02 59 47 BB
     */
    void sendResults(const std::vector<DetectionResult>& detections) {
        // 第1步：检查串口
        if (!serial_ || !serial_->isOpen()) return;

        // 第2步：检查检测结果
        if (detections.empty()) return;

        // 第3步：按x坐标从左到右排序
        std::vector<DetectionResult> sorted = detections;
        std::sort(sorted.begin(), sorted.end(),
                 [](const DetectionResult& a, const DetectionResult& b) {
                     return a.bbox.x < b.bbox.x;
                 });

        // 第4步：提取识别结果转换为ASCII字符
        std::vector<uint8_t> data;
        for (const auto& det : sorted) {
            std::string name = det.class_name;

            // 判断类别并转换
            if (name.find("Yellow") != std::string::npos) {
                data.push_back('Y');  // 黄豆
            } else if (name.find("Green") != std::string::npos) {
                data.push_back('G');  // 绿豆
            } else if (name.find("White") != std::string::npos) {
                data.push_back('W');  // 白豆
            } else if (name == "1" || name == "2" || name == "3" ||
                      name == "4" || name == "5") {
                data.push_back(name[0]);  // 数字直接用字符
            }
        }

        if (data.empty()) return;

        // 第5步：构建帧
        std::vector<uint8_t> frame;
        frame.reserve(2 + data.size() + 1);

        frame.push_back(0xAA);                           // 帧头
        frame.push_back(is_digit_ ? 0x01 : 0x02);       // 类型：01=数字，02=豆子
        frame.insert(frame.end(), data.begin(), data.end());  // 数据
        frame.push_back(0xBB);                           // 帧尾

        // 第6步：发送
        serial_->write(frame);
    }

    // 画检测框
    void drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections) {
        for (const auto& det : detections) {
            // 不同类别不同颜色
            cv::Scalar color;
            switch (det.class_id % 6) {
                case 0: color = cv::Scalar(255, 0, 0); break;
                case 1: color = cv::Scalar(0, 255, 0); break;
                case 2: color = cv::Scalar(0, 0, 255); break;
                case 3: color = cv::Scalar(255, 255, 0); break;
                case 4: color = cv::Scalar(255, 0, 255); break;
                default: color = cv::Scalar(0, 255, 255); break;
            }

            // 画框
            cv::rectangle(frame, det.bbox, color, 2);

            // 画标签
            std::string label = det.class_name + " " +
                               std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            int baseline;
            cv::Size sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(frame,
                         cv::Point(det.bbox.x, det.bbox.y - sz.height - baseline),
                         cv::Point(det.bbox.x + sz.width, det.bbox.y),
                         color, cv::FILLED);
            cv::putText(frame, label,
                       cv::Point(det.bbox.x, det.bbox.y - baseline),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }

private:
    std::string name_;
    Camera camera_;
    Detector detector_;
    bool initialized_{false};
    bool is_digit_;  // true=数字相机，false=豆子相机

    std::thread processing_thread_;
    std::atomic<bool> running_{false};

    std::mutex output_mutex_;
    cv::Mat latest_output_;
    bool output_available_{false};

    int frame_counter_{0};
    int fps_frame_counter_{0};
    int detection_interval_;
    float confidence_threshold_;
    float nms_threshold_;

    double current_fps_{0.0};
    std::chrono::steady_clock::time_point fps_last_time_;

    std::mutex stats_mutex_;
    std::vector<DetectionResult> last_detections_;

    // 串口
    SerialPort* serial_{nullptr};
};


// =============================================================================
// 第七部分：主程序
// =============================================================================

int main(int argc, char* argv[]) {
    // 设置信号处理
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "===== YOLO检测器 =====" << std::endl;

    // 加载配置
    Config cfg;
    cfg.load("config.txt");
    cfg.print();

    // 检查模型文件
    if (!fs::exists(cfg.model_path)) {
        std::cerr << "模型未找到: " << cfg.model_path << std::endl;
        return 1;
    }

    // 自动查找类别文件
    if (cfg.classes_file.empty()) {
        fs::path md = fs::path(cfg.model_path).parent_path();
        if (fs::exists(md / "classes.txt")) cfg.classes_file = (md / "classes.txt").string();
        else if (fs::exists("models/classes.txt")) cfg.classes_file = "models/classes.txt";
    }

    // 解析命令行参数
    std::string camera_mode = "all";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--camera" || arg == "-c") && i + 1 < argc) {
            camera_mode = argv[++i];
        }
    }

    // 判断启用哪些相机
    bool use_digit = (camera_mode == "all" || camera_mode == "digit" || camera_mode == "0");
    bool use_bean  = (camera_mode == "all" || camera_mode == "bean"  || camera_mode == "1");

    if (!use_digit && !use_bean) {
        std::cerr << "未知相机模式: " << camera_mode << " (可选: all/digit/bean)" << std::endl;
        return 1;
    }

    std::cout << "相机模式: " << camera_mode << std::endl;

    // 打开串口
    SerialPort serial;
    if (!cfg.serial_port.empty()) {
        if (serial.open(cfg.serial_port, cfg.serial_baudrate)) {
            std::cout << "串口通讯已启用" << std::endl;
        } else {
            std::cerr << "串口打开失败，仅终端输出" << std::endl;
        }
    } else {
        std::cout << "未配置串口" << std::endl;
    }

    // 创建流水线
    Pipeline* digit = nullptr;
    Pipeline* bean  = nullptr;

    if (use_digit) {
        digit = new Pipeline("数字相机", cfg.digit_camera, cfg.model_path, "",
                           cfg.classes_file,
                           cfg.confidence_threshold, cfg.nms_threshold, cfg.detection_interval);
        if (serial.isOpen()) digit->setSerial(&serial);
    }

    if (use_bean) {
        bean = new Pipeline("豆子相机", cfg.bean_camera, cfg.model_path, "",
                           cfg.classes_file,
                           cfg.confidence_threshold, cfg.nms_threshold, cfg.detection_interval);
        if (serial.isOpen()) bean->setSerial(&serial);
    }

    // 启动流水线
    bool any = false;
    if (digit && digit->isInitialized()) { digit->start(); any = true; }
    else if (digit) std::cerr << "数字相机初始化失败" << std::endl;

    if (bean && bean->isInitialized()) { bean->start(); any = true; }
    else if (bean) std::cerr << "豆子相机初始化失败" << std::endl;

    if (!any) {
        std::cerr << "无可用相机" << std::endl;
        return 1;
    }

    std::cout << "按 q/ESC 退出" << std::endl;

    // 主循环：显示图像
    cv::Mat df, bf;
    while (!g_exit) {
        if (digit && digit->getLatestOutput(df)) cv::imshow("Digit Camera", df);
        if (bean && bean->getLatestOutput(bf)) cv::imshow("Bean Camera", bf);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // 清理
    if (digit) { digit->stop(); delete digit; }
    if (bean)  { bean->stop();  delete bean; }
    cv::destroyAllWindows();

    std::cout << "已退出" << std::endl;
    return 0;
}
