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

#include "Program.h"
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#endif

static const char *compileError = "A shader could not be compiled.";
static const char *linkError = "A shader could not be linked.";

Program::Program()
: myProgram(0)
{
}

Program::~Program()
{
    if (myProgram)
    {
        glDeleteProgram(myProgram);
        myProgram = 0;
    }
}

const char * Program::build(const char *vertex, const char *fragment)
{
	const char *error = nullptr;
	GLuint vertexShader = 0, fragmentShader = 0;

    if (myProgram)
    {
        glDeleteProgram(myProgram);
        myProgram = 0;
    }

	vertexShader = compileShader(vertex, GL_VERTEX_SHADER, &error);
	if (error == nullptr)
	{
		fragmentShader = compileShader(fragment, GL_FRAGMENT_SHADER, &error);
	}

	if (error == nullptr)
	{
		myProgram = glCreateProgram();
		glAttachShader(myProgram, vertexShader);
		glAttachShader(myProgram, fragmentShader);
	}

	if (vertexShader)
	{
		glDeleteShader(vertexShader);
	}
	if (fragmentShader)
	{
		glDeleteShader(fragmentShader);
	}

	if (error == nullptr)
	{
		glLinkProgram(myProgram);

		GLint status;
		glGetProgramiv(myProgram, GL_LINK_STATUS, &status);
		if (status == GL_FALSE)
		{
			error = linkError;
			glDeleteProgram(myProgram);
			myProgram = 0;
		}
	}

	return error;
}

GLuint
Program::getName() const
{
    return myProgram;
}

GLuint
Program::compileShader(const char *source, GLenum type, const char **error)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        *error = compileError;

        glDeleteShader(shader);
        shader = 0;
    }
    else
    {
        *error = nullptr;
    }

    return shader;
}
