/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/

#include "Matrix.h"

Matrix::Matrix()
: matrix{1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0}
{

}

Matrix operator*(const Matrix &a, const Matrix &b)
{
    Matrix result;

    result[ 0] = a[ 0] * b[ 0] + a[ 4] * b[ 1] + a[ 8] * b[ 2] + a[12] * b[ 3];
    result[ 1] = a[ 1] * b[ 0] + a[ 5] * b[ 1] + a[ 9] * b[ 2] + a[13] * b[ 3];
    result[ 2] = a[ 2] * b[ 0] + a[ 6] * b[ 1] + a[10] * b[ 2] + a[14] * b[ 3];
    result[ 3] = a[ 3] * b[ 0] + a[ 7] * b[ 1] + a[11] * b[ 2] + a[15] * b[ 3];

    result[ 4] = a[ 0] * b[ 4] + a[ 4] * b[ 5] + a[ 8] * b[ 6] + a[12] * b[ 7];
    result[ 5] = a[ 1] * b[ 4] + a[ 5] * b[ 5] + a[ 9] * b[ 6] + a[13] * b[ 7];
    result[ 6] = a[ 2] * b[ 4] + a[ 6] * b[ 5] + a[10] * b[ 6] + a[14] * b[ 7];
    result[ 7] = a[ 3] * b[ 4] + a[ 7] * b[ 5] + a[11] * b[ 6] + a[15] * b[ 7];

    result[ 8] = a[ 0] * b[ 8] + a[ 4] * b[ 9] + a[ 8] * b[10] + a[12] * b[11];
    result[ 9] = a[ 1] * b[ 8] + a[ 5] * b[ 9] + a[ 9] * b[10] + a[13] * b[11];
    result[10] = a[ 2] * b[ 8] + a[ 6] * b[ 9] + a[10] * b[10] + a[14] * b[11];
    result[11] = a[ 3] * b[ 8] + a[ 7] * b[ 9] + a[11] * b[10] + a[15] * b[11];

    result[12] = a[ 0] * b[12] + a[ 4] * b[13] + a[ 8] * b[14] + a[12] * b[15];
    result[13] = a[ 1] * b[12] + a[ 5] * b[13] + a[ 9] * b[14] + a[13] * b[15];
    result[14] = a[ 2] * b[12] + a[ 6] * b[13] + a[10] * b[14] + a[14] * b[15];
    result[15] = a[ 3] * b[12] + a[ 7] * b[13] + a[11] * b[14] + a[15] * b[15];

    return result;
}
