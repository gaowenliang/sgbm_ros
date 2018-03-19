#ifndef COSTAGGREGATIONGENERIC_H
#define COSTAGGREGATIONGENERIC_H

#include "../configuration.h"
#include "../util.h"
#include "CostAggregationGenericIndexesIncrement.h"
#include "costAggregationGenericIteration.h"

template< class T, int add_col, int dir_type, bool recompute, bool join_dispcomputation >
__device__ __forceinline__ void
CostAggregationGeneric( uint8_t* d_cost,
                        uint8_t* d_L,
                        const int P1,
                        const int P2,
                        const int initial_row,
                        const int initial_col,
                        const int max_iter,
                        const int cols,
                        int add_index,
                        const T* _d_transform0,
                        const T* _d_transform1,
                        const int add_imindex,
                        uint8_t* __restrict__ d_disparity,
                        const uint8_t* d_L0,
                        const uint8_t* d_L1,
                        const uint8_t* d_L2,
                        const uint8_t* d_L3,
                        const uint8_t* d_L4,
                        const uint8_t* d_L5,
                        const uint8_t* d_L6 )
{
    const int lane = threadIdx.x % WARP_SIZE;
    const int dis  = 4 * lane;
    int index      = initial_row * cols * MAX_DISPARITY + initial_col * MAX_DISPARITY + dis;
    int col, index_im;
    if ( recompute || join_dispcomputation )
    {
        if ( recompute )
        {
            col = initial_col;
        }
        index_im = initial_row * cols + initial_col;
    }

    const int MAX_PAD        = UCHAR_MAX - P1;
    const uint32_t p1_vector = uchars_to_uint32( P1, P1, P1, P1 );
    const uint32_t p2_vector = uchars_to_uint32( P2, P2, P2, P2 );
    int old_value1;
    int old_value2;
    int old_value3;
    int old_value4;
    uint32_t min_cost, min_cost_p2, old_values;
    T rp0, rp1, rp2, rp3;

    if ( recompute )
    {
        if ( dir_type == DIR_LEFTRIGHT )
        {
            // clang-format off
            CostAggregationGenericIteration< T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation >(
            index, index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp0, &rp1, &rp2, &rp3,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
            // clang-format on

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index, index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp3, &rp0, &rp1, &rp2,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index, index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp2, &rp3, &rp0, &rp1,
            d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index, index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp1, &rp2, &rp3, &rp0,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            for ( int i = 4; i < max_iter - 3; i += 4 )
            {
                CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
                &index, &index_im, &col, add_index, add_imindex );

                // clang-format off
                CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
                index, index_im,
                col,
                &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
                &min_cost, &min_cost_p2,
                d_cost,
                d_L,
                p1_vector, p2_vector,
                _d_transform0, _d_transform1,
                lane,
                MAX_PAD,
                dis,
                &rp0, &rp1, &rp2, &rp3,
                d_disparity,
                d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

                CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
                &index, &index_im, &col, add_index, add_imindex );

                // clang-format off
                CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
                index, index_im,
                col,
                &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
                &min_cost, &min_cost_p2,
                d_cost,
                d_L,
                p1_vector, p2_vector,
                _d_transform0, _d_transform1,
                lane,
                MAX_PAD,
                dis,
                &rp3, &rp0, &rp1, &rp2,
                d_disparity,
                d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

                CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
                &index, &index_im, &col, add_index, add_imindex );

                // clang-format off
                CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
                index, index_im,
                col,
                &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
                &min_cost, &min_cost_p2,
                d_cost,
                d_L,
                p1_vector, p2_vector,
                _d_transform0, _d_transform1,
                lane,
                MAX_PAD,
                dis,
                &rp2, &rp3, &rp0, &rp1,
                d_disparity,
                d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

                CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
                &index, &index_im, &col, add_index, add_imindex );

                // clang-format off
                CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
                index, index_im,
                col,
                &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
                &min_cost, &min_cost_p2,
                d_cost,
                d_L,
                p1_vector, p2_vector,
                _d_transform0, _d_transform1,
                lane,
                MAX_PAD,
                dis,
                &rp1, &rp2, &rp3, &rp0,
                d_disparity,
                d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
                // clang-format on
            }

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index, index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp0, &rp1, &rp2, &rp3,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index, index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp3, &rp0, &rp1, &rp2,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index, index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp2, &rp3, &rp0, &rp1,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index, index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp1, &rp2, &rp3, &rp0,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
        }
        else if ( dir_type == DIR_RIGHTLEFT )
        {
            // clang-format off
            CostAggregationGenericIteration< T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation >(
            index, index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp0, &rp1, &rp2, &rp3,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp1, &rp2, &rp3, &rp0,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp2, &rp3, &rp0, &rp1,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp3, &rp0, &rp1, &rp2,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            for ( int i = 4; i < max_iter - 3; i += 4 )
            {
                CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
                &index, &index_im, &col, add_index, add_imindex );

                // clang-format off
                CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
                index,
                index_im,
                col,
                &old_values,
                &old_value1, &old_value2, &old_value3, &old_value4,
                &min_cost, &min_cost_p2,
                d_cost,
                d_L,
                p1_vector, p2_vector,
                _d_transform0, _d_transform1,
                lane,
                MAX_PAD,
                dis,
                &rp0, &rp1, &rp2, &rp3,
                d_disparity,
                d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

                CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
                &index, &index_im, &col, add_index, add_imindex );

                // clang-format off
                CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
                index,
                index_im,
                col,
                &old_values,
                &old_value1, &old_value2, &old_value3, &old_value4,
                &min_cost, &min_cost_p2,
                d_cost,
                d_L,
                p1_vector, p2_vector,
                _d_transform0, _d_transform1,
                lane,
                MAX_PAD,
                dis,
                &rp1, &rp2, &rp3, &rp0,
                d_disparity,
                d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

                CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
                &index, &index_im, &col, add_index, add_imindex );

                // clang-format off
                CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
                index,
                index_im,
                col,
                &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
                &min_cost, &min_cost_p2,
                d_cost,
                d_L,
                p1_vector, p2_vector,
                _d_transform0, _d_transform1,
                lane,
                MAX_PAD,
                dis,
                &rp2, &rp3, &rp0, &rp1,
                d_disparity,
                d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

                CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
                &index, &index_im, &col, add_index, add_imindex );

                // clang-format off
                CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
                index,
                index_im,
                col,
                &old_values,
                &old_value1, &old_value2, &old_value3, &old_value4,
                &min_cost, &min_cost_p2,
                d_cost,
                d_L,
                p1_vector, p2_vector,
                _d_transform0, _d_transform1,
                lane,
                MAX_PAD,
                dis,
                &rp3, &rp0, &rp1, &rp2,
                d_disparity,
                d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
            }

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp0, &rp1, &rp2, &rp3,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp1, &rp2, &rp3, &rp0,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp2, &rp3, &rp0, &rp1,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            // clang-format off
            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp3, &rp0, &rp1, &rp2,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
        }
        else
        {
            // clang-format off
            CostAggregationGenericIteration< T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp0, &rp1, &rp2, &rp3,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

            for ( int i = 1; i < max_iter; i++ )
            {
                CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
                &index, &index_im, &col, add_index, add_imindex );

                // clang-format off
                CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
                index,
                index_im,
                col,
                &old_values,
                &old_value1, &old_value2, &old_value3, &old_value4,
                &min_cost, &min_cost_p2,
                d_cost,
                d_L,
                p1_vector, p2_vector,
                _d_transform0, _d_transform1,
                lane,
                MAX_PAD,
                dis,
                &rp0, &rp1, &rp2, &rp3,
                d_disparity,
                d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
            }

            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp0, &rp1, &rp2, &rp3,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
        }
    }
    else
    {
        // clang-format off
        CostAggregationGenericIteration< T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation >(
        index,
        index_im,
        col,
        &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
        &min_cost, &min_cost_p2,
        d_cost,
        d_L,
        p1_vector, p2_vector,
        _d_transform0, _d_transform1,
        lane,
        MAX_PAD,
        dis,
        &rp0, &rp1, &rp2, &rp3,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );

        for ( int i = 1; i < max_iter; i++ )
        {
            CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
            &index, &index_im, &col, add_index, add_imindex );

            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values, &old_value1, &old_value2, &old_value3, &old_value4,
            &min_cost, &min_cost_p2,
            d_cost,
            d_L,
            p1_vector, p2_vector,
            _d_transform0, _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp0, &rp1, &rp2, &rp3,
            d_disparity,
            d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
        }

        CostAggregationGenericIndexesIncrement< add_col, recompute, join_dispcomputation >(
        &index, &index_im, &col, add_index, add_imindex );

        // clang-format off
        CostAggregationGenericIteration< T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation >(
        index, index_im,
        col,
        &old_values,
        &old_value1, &old_value2, &old_value3, &old_value4,
        &min_cost, &min_cost_p2,
        d_cost,
        d_L,
        p1_vector, p2_vector,
        _d_transform0, _d_transform1,
        lane,
        MAX_PAD,
        dis,
        &rp0, &rp1, &rp2, &rp3,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
    }
}

#endif // COSTAGGREGATIONGENERIC_H
