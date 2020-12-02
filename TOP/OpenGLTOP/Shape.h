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

#ifndef Shape_h
#define Shape_h

#include "TOP_CPlusPlusBase.h"
#include "Matrix.h"

class Shape
{
    /*
     A very naive 2D shape
     */
public:
    Shape();
    ~Shape();
    Shape(const Shape&) = delete;
    Shape& operator=(const Shape&) = delete;
    void setVertices(GLfloat *vertices, int elements);
    void setRotation(GLfloat degrees);
    void setTranslate(GLfloat x, GLfloat y);
    void setScale(GLfloat x, GLfloat y);
    void setup(GLuint attrib);
    void bindVAO() const;
    Matrix getMatrix() const;
    GLint getElementCount() const;
private:
    GLuint myVAO;
    GLuint myVBO;
    GLuint myElementCount;
    GLfloat myTranslateX;
    GLfloat myTranslateY;
    GLfloat myScaleX;
    GLfloat myScaleY;
    GLfloat myRotation;
};

#endif /* Shape_h */
