cmake_minimum_required(VERSION 3.10)
project(Crane)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 OpenCV 路径
set(OpenCV_DIR "D:/OpenCV/opencv/build")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(Crane bibilabu.cpp)

# 链接 OpenCV 库
target_link_libraries(Crane ${OpenCV_LIBS})

# 复制 DLL 到输出目录 (Windows)
if(WIN32)
    add_custom_command(TARGET Crane POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/OpenCV/opencv/build/x64/vc16/bin/opencv_world4100.dll"
        $<TARGET_FILE_DIR:Crane>
    )
    add_custom_command(TARGET Crane POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/OpenCV/opencv/build/x64/vc16/bin/opencv_world4100d.dll"
        $<TARGET_FILE_DIR:Crane>
    )
endif()
