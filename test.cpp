#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    // 1. 打开摄像头（0 表示默认摄像头）
    VideoCapture cap(0);

    // 检查摄像头是否成功打开
    if (!cap.isOpened())
    {
        cout << "无法打开摄像头！" << endl;
        return -1;
    }

    // 设置窗口名称
    string windowName = "OpenCV Camera";
    namedWindow(windowName, WINDOW_NORMAL);

    Mat frame;
    while (true)
    {
        // 2. 读取摄像头帧
        cap >> frame;

        // 检查帧是否为空（摄像头断开等情况）
        if (frame.empty())
        {
            cout << "摄像头帧为空！" << endl;
            break;
        }

        // 3. 在画面上显示文字
        putText(frame, "Press 'q' to quit", Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // 4. 显示画面
        imshow(windowName, frame);

        // 按 'q' 键退出（1ms 延迟，防止卡死）
        if (waitKey(1) == 'q')
        {
            break;
        }
    }

    // 释放摄像头资源
    cap.release();
    destroyAllWindows();

    return 0;
}
