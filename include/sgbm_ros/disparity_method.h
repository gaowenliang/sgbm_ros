/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef DISPARITY_METHOD_H_
#define DISPARITY_METHOD_H_

#include "configuration.h"
#include "cost_aggregation/cost_aggregation.h"
#include "costs.h"
#include "debug.h"
#include "hamming_cost.h"
#include "median_filter.h"
#include "util.h"
#include <opencv2/opencv.hpp>
#include <stdint.h>

void
init_disparity_method( const uint8_t _p1, const uint8_t _p2 );

void
compute_disparity_method2( cv::Mat left, cv::Mat right, cv::Mat& disparity, float* elapsed_time_ms );

void
compute_depth( cv::Mat left, cv::Mat right, cv::Mat& depth, float* elapsed_time_ms, const float FbaseLine );

void
finish_disparity_method( );

static void
free_memory( );

static void
createStreams( );

static void
destroyStreams( );

#endif /* DISPARITY_METHOD_H_ */
