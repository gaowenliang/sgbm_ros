#ifndef COSTAGGREGATIONKERNEL_H
#define COSTAGGREGATIONKERNEL_H

#include "../configuration.h"
#include "../util.h"
#include "costAggregationDiagonalGeneric.h"
#include "costAggregationGeneric.h"

// clang-format off

template< class T >
__global__ void
CostAggregationKernel_DiagonalDownUpRightLeft( uint8_t* d_cost,
                                               uint8_t* d_L,
                                               const int P1,
                                               const int P2,
                                               const int rows,
                                               const int cols,
                                               const T* d_transform0,
                                               const T* d_transform1,
                                               uint8_t* __restrict__ d_disparity,
                                               const uint8_t* d_L0,
                                               const uint8_t* d_L1,
                                               const uint8_t* d_L2,
                                               const uint8_t* d_L3,
                                               const uint8_t* d_L4,
                                               const uint8_t* d_L5,
                                               const uint8_t* d_L6 )
{
    const int initial_col
    = cols - ( blockIdx.x * ( blockDim.x / WARP_SIZE ) + ( threadIdx.x / WARP_SIZE ) ) - 1;
    if ( initial_col < cols )
    {
        const int initial_row           = rows - 1;
        const int add_index             = -1;
        const int col_nomin             = 0;
        const int col_copycost          = cols - 1;
        const int max_iter              = rows - 1;
        const bool recompute            = false;
        const bool join_dispcomputation = false;

        CostAggregationDiagonalGeneric< add_index, T, DIR_DOWNUP, recompute, join_dispcomputation >(
        d_cost,
        d_L,
        P1, P2,
        initial_row, initial_col,
        max_iter,
        col_nomin,
        col_copycost,
        cols,
        d_transform0, d_transform1,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
    }
}

template< class T >
__global__ void
CostAggregationKernel_DiagonalDownUpLeftRight( uint8_t* d_cost,
                                               uint8_t* d_L,
                                               const int P1,
                                               const int P2,
                                               const int rows,
                                               const int cols,
                                               const T* d_transform0,
                                               const T* d_transform1,
                                               uint8_t* __restrict__ d_disparity,
                                               const uint8_t* d_L0,
                                               const uint8_t* d_L1,
                                               const uint8_t* d_L2,
                                               const uint8_t* d_L3,
                                               const uint8_t* d_L4,
                                               const uint8_t* d_L5,
                                               const uint8_t* d_L6 )
{
    const int initial_col
    = cols - ( blockIdx.x * ( blockDim.x / WARP_SIZE ) + ( threadIdx.x / WARP_SIZE ) ) - 1;
    if ( initial_col >= 0 )
    {
        const int initial_row           = rows - 1;
        const int add_index             = 1;
        const int col_nomin             = cols - 1;
        const int col_copycost          = 0;
        const int max_iter              = rows - 1;
        const bool recompute            = false;
        const bool join_dispcomputation = false;

        CostAggregationDiagonalGeneric< add_index, T, DIR_DOWNUP, recompute, join_dispcomputation >(
        d_cost,
        d_L,
        P1, P2,
        initial_row, initial_col,
        max_iter,
        col_nomin,
        col_copycost,
        cols,
        d_transform0, d_transform1,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
    }
}

template< class T >

__global__ void
CostAggregationKernel_DiagonalUpDownRightLeft( uint8_t* d_cost,
                                               uint8_t* d_L,
                                               const int P1,
                                               const int P2,
                                               const int rows,
                                               const int cols,
                                               const T* d_transform0,
                                               const T* d_transform1,
                                               uint8_t* __restrict__ d_disparity,
                                               const uint8_t* d_L0,
                                               const uint8_t* d_L1,
                                               const uint8_t* d_L2,
                                               const uint8_t* d_L3,
                                               const uint8_t* d_L4,
                                               const uint8_t* d_L5,
                                               const uint8_t* d_L6 )
{
    const int initial_col = blockIdx.x * ( blockDim.x / WARP_SIZE ) + ( threadIdx.x / WARP_SIZE );
    if ( initial_col < cols )
    {
        const int initial_row           = 0;
        const int add_index             = -1;
        const int col_nomin             = 0;
        const int col_copycost          = cols - 1;
        const int max_iter              = rows - 1;
        const bool recompute            = false;
        const bool join_dispcomputation = PATH_AGGREGATION == 8;

        CostAggregationDiagonalGeneric< add_index, T, DIR_UPDOWN, recompute, join_dispcomputation >(
        d_cost,
        d_L,
        P1, P2,
        initial_row, initial_col,
        max_iter,
        col_nomin,
        col_copycost,
        cols,
        d_transform0, d_transform1,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
    }
}

template< class T >

__global__ void
CostAggregationKernel_DiagonalUpDownLeftRight( uint8_t* d_cost,
                                               uint8_t* d_L,
                                               const int P1,
                                               const int P2,
                                               const int rows,
                                               const int cols,
                                               const T* d_transform0,
                                               const T* d_transform1,
                                               uint8_t* __restrict__ d_disparity,
                                               const uint8_t* d_L0,
                                               const uint8_t* d_L1,
                                               const uint8_t* d_L2,
                                               const uint8_t* d_L3,
                                               const uint8_t* d_L4,
                                               const uint8_t* d_L5,
                                               const uint8_t* d_L6 )
{
    const int initial_col = blockIdx.x * ( blockDim.x / WARP_SIZE ) + ( threadIdx.x / WARP_SIZE );
    if ( initial_col < cols )
    {
        const int initial_row           = 0;
        const int add_index             = 1;
        const int col_nomin             = cols - 1;
        const int col_copycost          = 0;
        const int max_iter              = rows - 1;
        const bool recompute            = false;
        const bool join_dispcomputation = false;

        CostAggregationDiagonalGeneric< add_index, T, DIR_UPDOWN, recompute, join_dispcomputation >(
        d_cost,
        d_L,
        P1, P2,
        initial_row, initial_col,
        max_iter,
        col_nomin,
        col_copycost,
        cols,
        d_transform0, d_transform1,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
    }
}

template< class T >

__global__ void
CostAggregationKernel_LeftToRight( uint8_t* d_cost,
                                   uint8_t* d_L,
                                   const int P1,
                                   const int P2,
                                   const int rows,
                                   const int cols,
                                   const T* d_transform0,
                                   const T* d_transform1,
                                   uint8_t* __restrict__ d_disparity,
                                   const uint8_t* d_L0,
                                   const uint8_t* d_L1,
                                   const uint8_t* d_L2,
                                   const uint8_t* d_L3,
                                   const uint8_t* d_L4,
                                   const uint8_t* d_L5,
                                   const uint8_t* d_L6 )
{
    const int initial_row = blockIdx.x * ( blockDim.x / WARP_SIZE ) + ( threadIdx.x / WARP_SIZE );
    if ( initial_row < rows )
    {
        const int initial_col           = 0;
        const int add_index             = MAX_DISPARITY;
        const int add_imindex           = 1;
        const int max_iter              = cols - 1;
        const int add_col               = 1;
        const bool recompute            = true;
        const bool join_dispcomputation = false;

        CostAggregationGeneric< T, add_col, DIR_LEFTRIGHT, recompute, join_dispcomputation >(
        d_cost,
        d_L,
        P1, P2,
        initial_row, initial_col,
        max_iter,
        cols,
        add_index,
        d_transform0, d_transform1,
        add_imindex,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
    }
}

template< class T >

__global__ void
CostAggregationKernel_RightToLeft( uint8_t* d_cost,
                                   uint8_t* d_L,
                                   const int P1,
                                   const int P2,
                                   const int rows,
                                   const int cols,
                                   const T* d_transform0,
                                   const T* d_transform1,
                                   uint8_t* __restrict__ d_disparity,
                                   const uint8_t* d_L0,
                                   const uint8_t* d_L1,
                                   const uint8_t* d_L2,
                                   const uint8_t* d_L3,
                                   const uint8_t* d_L4,
                                   const uint8_t* d_L5,
                                   const uint8_t* d_L6 )
{
    const int initial_row = blockIdx.x * ( blockDim.x / WARP_SIZE ) + ( threadIdx.x / WARP_SIZE );
    if ( initial_row < rows )
    {
        const int initial_col           = cols - 1;
        const int add_index             = -MAX_DISPARITY;
        const int add_imindex           = -1;
        const int max_iter              = cols - 1;
        const int add_col               = -1;
        const bool recompute            = true;
        const bool join_dispcomputation = false;

        CostAggregationGeneric< T, add_col, DIR_RIGHTLEFT, recompute, join_dispcomputation >(
        d_cost,
        d_L,
        P1, P2,
        initial_row, initial_col,
        max_iter,
        cols,
        add_index,
        d_transform0, d_transform1,
        add_imindex,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
    }
}

template< class T >
__global__ void
CostAggregationKernel_DownToUp( uint8_t* d_cost,
                                uint8_t* d_L,
                                const int P1,
                                const int P2,
                                const int rows,
                                const int cols,
                                const T* d_transform0,
                                const T* d_transform1,
                                uint8_t* __restrict__ d_disparity,
                                const uint8_t* d_L0,
                                const uint8_t* d_L1,
                                const uint8_t* d_L2,
                                const uint8_t* d_L3,
                                const uint8_t* d_L4,
                                const uint8_t* d_L5,
                                const uint8_t* d_L6 )
{
    const int initial_col = blockIdx.x * ( blockDim.x / WARP_SIZE ) + ( threadIdx.x / WARP_SIZE );
    if ( initial_col < cols )
    {
        const int initial_row           = rows - 1;
        const int add_index             = -cols * MAX_DISPARITY;
        const int add_imindex           = -cols;
        const int max_iter              = rows - 1;
        const int add_col               = 0;
        const bool recompute            = false;
        const bool join_dispcomputation = PATH_AGGREGATION == 4;

        CostAggregationGeneric< T, add_col, DIR_DOWNUP, recompute, join_dispcomputation >(
        d_cost,
        d_L,
        P1, P2,
        initial_row, initial_col,
        max_iter,
        cols,
        add_index,
        d_transform0, d_transform1,
        add_imindex,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
    }
}

template< class T >
//__launch_bounds__(64, 16)
__global__ void
CostAggregationKernel_UpToDown( uint8_t* d_cost,
                                uint8_t* d_L,
                                const int P1,
                                const int P2,
                                const int rows,
                                const int cols,
                                const T* d_transform0,
                                const T* d_transform1,
                                uint8_t* __restrict__ d_disparity,
                                const uint8_t* d_L0,
                                const uint8_t* d_L1,
                                const uint8_t* d_L2,
                                const uint8_t* d_L3,
                                const uint8_t* d_L4,
                                const uint8_t* d_L5,
                                const uint8_t* d_L6 )
{
    const int initial_col = blockIdx.x * ( blockDim.x / WARP_SIZE ) + ( threadIdx.x / WARP_SIZE );

    if ( initial_col < cols )
    {
        const int initial_row           = 0;
        const int add_index             = cols * MAX_DISPARITY;
        const int add_imindex           = cols;
        const int max_iter              = rows - 1;
        const int add_col               = 0;
        const bool recompute            = false;
        const bool join_dispcomputation = false;

        CostAggregationGeneric< T, add_col, DIR_UPDOWN, recompute, join_dispcomputation >(
        d_cost,
        d_L,
        P1, P2,
        initial_row, initial_col,
        max_iter,
        cols,
        add_index,
        d_transform0, d_transform1,
        add_imindex,
        d_disparity,
        d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6 );
    }
}

#endif // COSTAGGREGATIONKERNEL_H
