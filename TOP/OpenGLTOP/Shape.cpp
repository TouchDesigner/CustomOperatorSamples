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

#include "Shape.h"
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
// Enable M_PI
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <math.h>
#include "Matrix.h"

Shape::Shape()
: myVAO(0), myVBO(0), myElementCount(0), myTranslateX(0.0), myTranslateY(0.0),
    myScaleX(1.0), myScaleY(1.0), myRotation(0.0)
{
}

Shape::~Shape()
{
    glDeleteVertexArrays(1, &myVAO);
    glDeleteBuffers(1, &myVBO);
}

void
Shape::setVertices(GLfloat *vertices, int elements)
{
    if (myVBO == 0)
    {
        glGenBuffers(1, &myVBO);
    }
    myElementCount = elements;
    glBindBuffer(GL_ARRAY_BUFFER, myVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * elements, vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
Shape::setRotation(GLfloat degrees)
{
    myRotation = static_cast<GLfloat>(degrees * M_PI / 180.0);
}

void
Shape::setTranslate(GLfloat x, GLfloat y)
{
    myTranslateX= x;
    myTranslateY = y;
}

void
Shape::setScale(GLfloat x, GLfloat y)
{
    myScaleX = x;
    myScaleY = y;
}

void
Shape::bindVAO() const
{
    glBindVertexArray(myVAO);
}

void
Shape::setup(GLuint attrib)
{
    if (myVAO == 0)
    {
        glGenVertexArrays(1, &myVAO);
    }
    if (myVBO == 0)
    {
        glGenBuffers(1, &myVBO);
    }
    bindVAO();
    glBindBuffer(GL_ARRAY_BUFFER, myVBO);
    glEnableVertexAttribArray(attrib);
    glVertexAttribPointer(attrib, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);
}

Matrix
Shape::getMatrix() const
{
    Matrix scale;
    scale[0] = myScaleX;
    scale[5] = myScaleY;

    Matrix translate;
    translate[3] = myTranslateX;
    translate[7] = myTranslateY;

    Matrix rotate;
    rotate[0] = rotate[5] = std::cos(myRotation);
    rotate[1] = std::sin(myRotation);
    rotate[4] = -rotate[1];
    rotate[10] = rotate[15] = 1.0;

    return rotate * translate * scale;
}

GLint
Shape::getElementCount() const
{
    return myElementCount;
}
