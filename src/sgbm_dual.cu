#include "../include/sgbm_ros/backward.hpp"
#include "../include/sgbm_ros/disparity_method.h"
#include "../include/sgbm_ros/transport_util.h"
#include <Eigen/Eigen>
#include <assert.h>
#include <cmath> // std::string
#include <code_utils/sys_utils/tic_toc.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cv_bridge/cv_bridge.h>
#include <dirent.h>
#include <fstream>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <stdexcept>
#include <stdint.h>
#include <stdlib.h>
#include <string> // std::string
#include <vector>

using namespace std;
using namespace Eigen;
using namespace cv;

namespace backward
{
backward::SignalHandling sh;
}

ros::Publisher depthImg_pub;

cv::Mat image_left, image_right;
cv::Mat depthImage, disparityImage;
sensor_msgs::Image image_ros;
ros::Time image_time;
bool is_image_ready = false;

const int limit_left   = 30;
const int limit_right  = 230;
const double max_depth = 15.0;
const double min_depth = 0.4;

int _p1, _p2;
int dis_min = 100, dis_max;
float fx_l, fx_r;
float fy_l, fy_r;
float scale_ = 1.0;
float baseLine;
float cx_l, cx_r;
float cy_l, cy_r;

int image_row = 100, image_col = 100;

void
stereoIimageProcessCallback( const sensor_msgs::ImageConstPtr& image_msg )
{
    image_left  = cv_bridge::toCvCopy( image_msg, "mono8" )->image.rowRange( 0, image_row );
    image_right = cv_bridge::toCvCopy( image_msg, "mono8" )->image.rowRange( image_row, 2 * image_row );
    image_time  = image_msg->header.stamp;

    is_image_ready  = true;
}

void
getDepthImage( cv::Mat dispImg, cv::Mat& depthImg )
{
    uchar* p_dis;
    float* p_dep;

    for ( int row_index = 0; row_index < dispImg.rows; row_index++ )
    {
        p_dis = dispImg.ptr< uchar >( row_index );
        p_dep = depthImg.ptr< float >( row_index );

        for ( int col_index = 0; col_index < dispImg.cols; col_index++ )
        {
            if ( col_index < limit_left || col_index > limit_right )
                p_dep[col_index] = -1.0;
            else
            {
                if ( p_dis[col_index] <= dis_min )
                    p_dep[col_index] = -1.0;
                else
                    p_dep[col_index] = ( fx_l * baseLine ) / p_dis[col_index];
            }
        }
    }
}

void
stereo_sgbm( )
{
    if ( !is_image_ready )
        return;
    // Compute
    sys_utils::tic::TicTocPart sgm_tic;

    float elapsed_time_ms;
    compute_disparity_method2( image_left, image_right, disparityImage, &elapsed_time_ms );

    getDepthImage( disparityImage, depthImage );

    toImageMsg( image_ros, depthImage, image_time );

    depthImg_pub.publish( image_ros );
    ROS_DEBUG( "Time in total %fms, SGBM cuda cost %fms", sgm_tic.tocEnd( ), elapsed_time_ms );

    is_image_ready = false;
}

int
main( int argc, char** argv )
{
    ros::init( argc, argv, "sgbm_dual" );
    ros::NodeHandle nh( "~" );
    ros::console::set_logger_level( ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug );

    double _baseline;
    int run_rate;
    double fx, fy, cx, cy;
    std::string image_sub_topic;

    nh.param( "p1", _p1, 600 );
    nh.param( "p2", _p2, 2400 );
    nh.param( "baseline", _baseline, 0.24 );
    nh.param( "rate", run_rate, 10 );
    nh.param( "fx", fx, 100.0 );
    nh.param( "fy", fy, 100.0 );
    nh.param( "cx", cx, 100.0 );
    nh.param( "cy", cy, 100.0 );
    nh.param( "row", image_row, 100 );
    nh.param( "col", image_col, 100 );
    nh.getParam( "image_topic", image_sub_topic );

    fx_l     = fx * scale_;
    fy_l     = fy * scale_;
    fx_r     = fx * scale_;
    fy_r     = fy * scale_;
    cx_l     = cx * scale_;
    cy_l     = cy * scale_;
    cx_r     = cx * scale_;
    cy_r     = cy * scale_;
    baseLine = _baseline;
    dis_min  = fx_l * baseLine / max_depth;
    dis_max  = fx_l * baseLine / min_depth;

    std::cout << "[INFO]image sub topic: " << image_sub_topic << std::endl;
    std::cout << "[INFO]fx fy: " << fx_l << " " << fy_l << std::endl;
    std::cout << "[INFO]cx cy: " << cx_l << " " << cy_l << std::endl;
    std::cout << "[INFO]baseline: " << baseLine << "m" << std::endl;
    std::cout << "[INFO]run rate: " << run_rate << "Hz" << std::endl;
    std::cout << "[INFO]min disparaty: " << dis_min << " pixels" << std::endl;
    std::cout << "[INFO]max disparaty: " << dis_max << " pixels" << std::endl;

    disparityImage            = cv::Mat( image_row, image_col, CV_8UC1 );
    depthImage                = cv::Mat( image_row, image_col, CV_32FC1 );
    image_ros.encoding        = sensor_msgs::image_encodings::TYPE_32FC1;
    image_ros.header.frame_id = "/ref_frame";

    ros::Subscriber subImgs
    = nh.subscribe< sensor_msgs::Image >( image_sub_topic,
                                          100, //
                                          &stereoIimageProcessCallback,
                                          ros::VoidConstPtr( ),
                                          ros::TransportHints( ).tcpNoDelay( true ) );

    depthImg_pub = nh.advertise< sensor_msgs::Image >( "depth_image", 1 );

    ros::Rate rate( run_rate );
    bool status = ros::ok( );

    init_disparity_method( _p1, _p2 );

    while ( status )
    {
        ros::spinOnce( );

        stereo_sgbm( );

        status = ros::ok( );
        rate.sleep( );
    }

    finish_disparity_method( );
    ROS_WARN( "SGBM End!!!" );
}
