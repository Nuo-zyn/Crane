cmake_minimum_required(VERSION 4.0)
project(VSCODE)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 手动指定 OpenCV 安装路径（确保能找到 4.13.0）
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 生成可执行文件，源文件是你的 bibilabu.cpp
add_executable(bibilabu bibilabu.cpp)

# 链接 OpenCV 库
target_link_libraries(bibilabu ${OpenCV_LIBS})
