cmake_minimum_required(VERSION 2.8.3)
project(sgbm_ros)

set(CMAKE_VERBOSE_MAKEFILE "true")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  cv_bridge
  image_transport
  code_utils
)

find_package( CUDA REQUIRED )
include(/home/gao/lib/opencv-3.3.1/build/OpenCVConfig.cmake)
#find_package( OpenCV REQUIRED)
find_package( Eigen3 REQUIRED)

catkin_package(
   CATKIN_DEPENDS roscpp rospy
)

include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Eigen library path: ${EIGEN3_INCLUDE_DIR}")

message(STATUS "CUDA include path: ${CUDA_INCLUDE_DIRS}")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    -gencode=arch=compute_30,code=sm_30     # Quadro K2000
#    -gencode=arch=compute_35,code=sm_35
#    -gencode=arch=compute_50,code=sm_50
#    -gencode=arch=compute_52,code=sm_52
#    -gencode=arch=compute_53,code=sm_53    # TX1
#    -gencode=arch=compute_62,code=sm_62    # TX2
    )

#cuda_add_executable( sgbm_ros_node
#    src/sgbm_ros.cu
#    src/median_filter.cu
#    src/hamming_cost.cu
#    src/disparity_method.cu
#    src/debug.cu
#    src/costs.cu)

#target_link_libraries( sgbm_ros_node
#  ${catkin_LIBRARIES}
#  ${OpenCV_LIBS}
#)

cuda_add_executable( sgbm_dual_node
    src/sgbm_dual.cu
    src/median_filter.cu
    src/hamming_cost.cu
    src/disparity_method.cu
    src/debug.cu
    src/costs.cu)

target_link_libraries( sgbm_dual_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
