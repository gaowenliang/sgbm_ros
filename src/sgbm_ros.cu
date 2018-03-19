#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

#include "geometry_msgs/PoseStamped.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <cmath> // std::string
#include <stdint.h>
#include <string> // std::string

#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp> // cv::Mat & cv::Size
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Eigen>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "../include/sgbm_ros/disparity_method.h"

#include "../include/sgbm_ros/backward.hpp"
#include "../include/sgbm_ros/transport_util.h"


using namespace std;
using namespace Eigen;
using namespace cv;

namespace backward
{
backward::SignalHandling sh;
}

ros::Publisher depthImg_pub;

cv::Mat image_left_, image_right_;
ros::Time last_sync_image_stamp_;
bool is_sync_image_ready_ = false;

const int limit_left   = 30;
const int limit_right  = 230;
const double max_depth = 20.0;

int _p1, _p2;
float fx_l, fx_r;
float fy_l, fy_r;
float scale_ = 1.0;
float baseLine;
float cx_l, cx_r;
float cy_l, cy_r;

void
imageProcessCallback( const sensor_msgs::ImageConstPtr& left_image_msg, const sensor_msgs::ImageConstPtr& right_image_msg )
{
  //    image_left_ = matFromImage(left_image_msg).clone();
  //    image_right_ = matFromImage(right_image_msg).clone();

  image_left_            = cv_bridge::toCvCopy( left_image_msg, "mono8" )->image;
  image_right_           = cv_bridge::toCvCopy( right_image_msg, "mono8" )->image;
  last_sync_image_stamp_ = left_image_msg->header.stamp;
  is_sync_image_ready_   = true;
}


cv::Mat
getDepthImage( cv::Mat dispImg )
{
    cv::Mat depImg( dispImg.rows, dispImg.cols, CV_32FC1 );

    double max_d = -1.0;
    for ( int i = 0; i < dispImg.rows; i++ )
        for ( int j = 0; j < dispImg.cols; j++ )
        {
            unsigned char disparity = dispImg.at< unsigned char >( i, j );

            float z_;
            //	cout<<"disparity: "<<disparity<<endl;
            //	if(disparity > 1000.0)
            //		z_ = -1.0;
            //	else
            z_ = ( float )( fx_l * baseLine ) / disparity;
            //         cout<<"depth: "<<z_ <<endl;

            if ( j < limit_left || j > limit_right ) // crop..
                z_ = -1.0;
            depImg.at< float >( i, j ) = z_;

            if ( z_ > max_depth )
                z_ = -1.0;
            else if ( z_ > max_d )
                max_d = z_;
        }

    cv::Mat depth_image = depImg.clone( );

    return depth_image;
}

void
stereo_sgbm( )
{
    if ( !is_sync_image_ready_ )
      return;
    // Compute
    ros::Time time_start = ros::Time::now( );
    float elapsed_time_ms;
    cv::Mat disparity_im  = compute_disparity_method( image_left_, image_right_, &elapsed_time_ms );
    ros::Time time_middle = ros::Time::now( );
    cv::Mat depthImg      = getDepthImage( disparity_im );
    ros::Time time_end    = ros::Time::now( );
    ROS_WARN( "Time in SGBM %f", ( time_end - time_start ).toSec( ) );
    // ROS_INFO("Time in coreSGBM %f", (time_middle - time_start).toSec() );

    sensor_msgs::Image image_ros;
    // cout<<"depth image: \n"<<depthImg<<endl;
    toImageMsg( image_ros, depthImg, last_sync_image_stamp_);
    image_ros.encoding        = sensor_msgs::image_encodings::TYPE_32FC1;
    image_ros.header.frame_id = "/ref_frame";
    depthImg_pub.publish( image_ros );
    is_sync_image_ready_ = false;
}

int
main( int argc, char** argv )
{
    ros::init( argc, argv, "sgbm_ros_node" );
    ros::NodeHandle nh( "~" );

    double _baseline;
    int run_rate;
    double fx,fy,cx,cy;
    nh.param( "p1", _p1, 600 );
    nh.param( "p2", _p2, 2400 );
    nh.param( "baseline", _baseline, 0.24 );
    nh.param( "rate", run_rate, 10 );
    nh.param( "fx", fx, 100.0 );
    nh.param( "fy", fy, 100.0 );
    nh.param( "cx", cx, 100.0 );
    nh.param( "cy", cy, 100.0 );

    fx_l = fx * scale_;
    fy_l = fy * scale_;
    fx_r = fx * scale_;
    fy_r = fy * scale_;
    cx_l = cx * scale_;
    cy_l = cy * scale_;
    cx_r = cx * scale_;
    cy_r = cy * scale_;
    baseLine = _baseline;

    std::cout << "[INFO]fx fy: " << fx_l << " " << fy_l << std::endl;
    std::cout << "[INFO]cx cy: " << cx_l << " " << cy_l << std::endl;
    std::cout << "[INFO]baseline: " << baseLine << "m" << std::endl;
    std::cout << "[INFO]run rate: " << run_rate << "Hz" << std::endl;

    message_filters::Subscriber< sensor_msgs::Image > sub_imgL( nh, "/image_left", 2 );
    message_filters::Subscriber< sensor_msgs::Image > sub_imgR( nh, "/image_right", 2 );
    typedef message_filters::sync_policies::ApproximateTime< sensor_msgs::Image, sensor_msgs::Image > SyncPolicy;
    message_filters::Synchronizer< SyncPolicy > sync( SyncPolicy( 3 ), sub_imgL, sub_imgR );
    sync.registerCallback( boost::bind( &imageProcessCallback, _1, _2 ) );

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
