#ifndef COSTAGGREGATIONDIAGONALGENERIC_H
#define COSTAGGREGATIONDIAGONALGENERIC_H

#include "../configuration.h"
#include "../util.h"
#include "CostAggregationGenericIndexesIncrement.h"
#include "costAggregationGenericIteration.h"

template< int add_index, class T, int dir_type, bool recompute, bool join_dispcomputation >
__device__ __forceinline__ void
CostAggregationDiagonalGeneric( uint8_t* d_cost,
                                uint8_t* d_L,
                                const int P1,
                                const int P2,
                                const int initial_row,
                                const int initial_col,
                                const int max_iter,
                                const int col_nomin,
                                const int col_copycost,
                                const int cols,
                                const T* _d_transform0,
                                const T* _d_transform1,
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
    int col        = initial_col;
    int index      = initial_row * cols * MAX_DISPARITY + initial_col * MAX_DISPARITY + dis;
    int index_im;
    if ( recompute || join_dispcomputation )
    {
        index_im = initial_row * cols + col;
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

    // clang-format off
    CostAggregationGenericIteration< T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation >(
    index,
    index_im,
    col,
    &old_values,
    &old_value1,
    &old_value2,
    &old_value3,
    &old_value4,
    &min_cost,
    &min_cost_p2,
    d_cost,
    d_L,
    p1_vector,
    p2_vector,
    _d_transform0,
    _d_transform1,
    lane,
    MAX_PAD,
    dis,
    &rp0,
    &rp1,
    &rp2,
    &rp3,
    d_disparity,
    d_L0,
    d_L1,
    d_L2,
    d_L3,
    d_L4,
    d_L5,
    d_L6 );

    for ( int i = 1; i < max_iter; i++ )
    {
        CostAggregationDiagonalGenericIndexesIncrement< add_index, recompute, join_dispcomputation >(
        &index, &index_im, &col, cols, initial_row, i, dis );

        if ( col == col_copycost )
        {
            // clang-format off
            CostAggregationGenericIteration< T, ITER_COPY, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1,
            &old_value2,
            &old_value3,
            &old_value4,
            &min_cost,
            &min_cost_p2,
            d_cost,
            d_L,
            p1_vector,
            p2_vector,
            _d_transform0,
            _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp0,
            &rp1,
            &rp2,
            &rp3,
            d_disparity,
            d_L0,
            d_L1,
            d_L2,
            d_L3,
            d_L4,
            d_L5,
            d_L6 );
        }
        else
        {
            // clang-format off
            CostAggregationGenericIteration< T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation >(
            index,
            index_im,
            col,
            &old_values,
            &old_value1,
            &old_value2,
            &old_value3,
            &old_value4,
            &min_cost,
            &min_cost_p2,
            d_cost,
            d_L,
            p1_vector,
            p2_vector,
            _d_transform0,
            _d_transform1,
            lane,
            MAX_PAD,
            dis,
            &rp0,
            &rp1,
            &rp2,
            &rp3,
            d_disparity,
            d_L0,
            d_L1,
            d_L2,
            d_L3,
            d_L4,
            d_L5,
            d_L6 );
        }
    }

    CostAggregationDiagonalGenericIndexesIncrement< add_index, recompute, join_dispcomputation >(
    &index, &index_im, &col, cols, max_iter, initial_row, dis );

    if ( col == col_copycost )
    {
        // clang-format off
        CostAggregationGenericIteration< T, ITER_COPY, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation >(
        index,
        index_im,
        col,
        &old_values,
        &old_value1,
        &old_value2,
        &old_value3,
        &old_value4,
        &min_cost,
        &min_cost_p2,
        d_cost,
        d_L,
        p1_vector,
        p2_vector,
        _d_transform0,
        _d_transform1,
        lane,
        MAX_PAD,
        dis,
        &rp0,
        &rp1,
        &rp2,
        &rp3,
        d_disparity,
        d_L0,
        d_L1,
        d_L2,
        d_L3,
        d_L4,
        d_L5,
        d_L6 );
    }
    else
    {
        // clang-format off
        CostAggregationGenericIteration< T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation >(
        index,
        index_im,
        col,
        &old_values,
        &old_value1,
        &old_value2,
        &old_value3,
        &old_value4,
        &min_cost,
        &min_cost_p2,
        d_cost,
        d_L,
        p1_vector,
        p2_vector,
        _d_transform0,
        _d_transform1,
        lane,
        MAX_PAD,
        dis,
        &rp0,
        &rp1,
        &rp2,
        &rp3,
        d_disparity,
        d_L0,
        d_L1,
        d_L2,
        d_L3,
        d_L4,
        d_L5,
        d_L6 );
    }
}




#endif // COSTAGGREGATIONDIAGONALGENERIC_H
