#ifndef COSTAGGREGATIONGENERICINDEXESINCREMENT_H
#define COSTAGGREGATIONGENERICINDEXESINCREMENT_H

#include "../configuration.h"
#include "../util.h"

template< int add_col, bool recompute, bool join_dispcomputation >
__device__ __forceinline__ void
CostAggregationGenericIndexesIncrement( int* index, int* index_im, int* col, const int add_index, const int add_imindex )
{
    *index += add_index;

    if ( recompute || join_dispcomputation )
    {
        *index_im += add_imindex;
        if ( recompute )
        {
            *col += add_col;
        }
    }
}

template< int add_index, bool recompute, bool join_dispcomputation >
__device__ __forceinline__ void
CostAggregationDiagonalGenericIndexesIncrement(
int* index, int* index_im, int* col, const int cols, const int initial_row, const int i, const int dis )
{
    *col += add_index;
    if ( add_index > 0 && *col > cols )
    {
        *col = 0;
    }
    else if ( *col < 0 )
    {
        *col = cols - 1;
    }
    *index = abs( initial_row - i ) * cols * MAX_DISPARITY + *col * MAX_DISPARITY + dis;

    if ( recompute || join_dispcomputation )
    {
        *index_im = abs( initial_row - i ) * cols + *col;
    }
}

#endif // COSTAGGREGATIONGENERICINDEXESINCREMENT_H
