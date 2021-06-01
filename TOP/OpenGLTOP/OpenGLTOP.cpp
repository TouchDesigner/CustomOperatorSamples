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

#include "OpenGLTOP.h"
#include "Parameters.h"

#include <assert.h>
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <string.h>
#endif
#include <cstdio>

static const char *vertexShader = "#version 330\n\
uniform mat4 uModelView; \
in vec3 P; \
void main() { \
    gl_Position = vec4(P, 1) * uModelView; \
}";

static const char *fragmentShader = "#version 330\n\
uniform vec4 uColor; \
out vec4 finalColor; \
void main() { \
    finalColor = uColor; \
}";

static const char *uniformError = "A uniform location could not be found.";

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{
DLLEXPORT
void
FillTOPPluginInfo(TOP_PluginInfo *info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TOP_ExecuteMode::OpenGL_FBO;

	// The opType is the unique name for this TOP. It must start with a 
	// capital A-Z character, and all the following characters must lower case
	// or numbers (a-z, 0-9)
	info->customOPInfo.opType->setString("Openglsample");

	// The opLabel is the text that will show up in the OP Create Dialog
	info->customOPInfo.opLabel->setString("OpenGL Sample");

	// Will be turned into a 3 letter icon on the nodes
	info->customOPInfo.opIcon->setString("OGL");

	// Information about the author of this OP
	info->customOPInfo.authorName->setString("Author Name");
	info->customOPInfo.authorEmail->setString("email@email.com");

	// This TOP works with 0 inputs
	info->customOPInfo.minInputs = 0;
	info->customOPInfo.maxInputs = 0;

}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context *context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll

    // Note we can't do any OpenGL work during instantiation

	return new OpenGLTOP(info, context);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL

    // We do some OpenGL teardown on destruction, so ask the TOP_Context
    // to set up our OpenGL context
    context->beginGLCommands();

	delete (OpenGLTOP*)instance;

    context->endGLCommands();
}

};


OpenGLTOP::OpenGLTOP(const OP_NodeInfo* info, TOP_Context *context)
: myNodeInfo(info), myExecuteCount(0), myRotation(0.0), myError(nullptr),
    myProgram(), myDidSetup(false), myModelViewUniform(-1), myColorUniform(-1)
{

#ifdef _WIN32
	// GLEW is global static function pointers, only needs to be inited once,
	// and only on Windows.
	static bool needGLEWInit = true;
	if (needGLEWInit)
	{
		needGLEWInit = false;
		context->beginGLCommands();
		// Setup all our GL extensions using GLEW
		glewInit();
		context->endGLCommands();
	}
#endif


	// If you wanted to do other GL initialization inside this constructor, you could
	// uncomment these lines and do the work between the begin/end
	//
	//context->beginGLCommands();
	// Custom GL initialization here
	//context->endGLCommands();
}

OpenGLTOP::~OpenGLTOP()
{

}

void
OpenGLTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs *inputs, void* reserved1) 
{
	// Setting cookEveryFrame to true causes the TOP to cook every frame even
	// if none of its inputs/parameters are changing. Set it to false if it
    // only needs to cook when inputs/parameters change.
	ginfo->cookEveryFrame = true;
}

bool
OpenGLTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs *inputs, void* reserved1)
{
	// In this function we could assign variable values to 'format' to specify
	// the pixel format/resolution etc that we want to output to.
	// If we did that, we'd want to return true to tell the TOP to use the settings we've
	// specified.
	// In this example we'll return false and use the TOP's settings
	return false;
}


void
OpenGLTOP::execute(TOP_OutputFormatSpecs* outputFormat ,
							const OP_Inputs* inputs,
							TOP_Context* context,
							void* reserved1)
{
	myError = nullptr;
	myExecuteCount++;

	// These functions must be called before
	// beginGLCommands()/endGLCommands() block
	double speed = myParms.evalSpeed(inputs);

	Color color1 = myParms.evalColor1(inputs);
	Color color2 = myParms.evalColor1(inputs);

    myRotation += speed;

    int width = outputFormat->width;
    int height = outputFormat->height;

    float ratio = static_cast<float>(height) / static_cast<float>(width);

    Matrix view;
    view[0] = ratio;

    context->beginGLCommands();
    
    setupGL();

    if (!myError)
    {
        glViewport(0, 0, width, height);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(myProgram.getName());

        // Draw the square

        glUniform4f(myColorUniform, static_cast<GLfloat>(color1.r), static_cast<GLfloat>(color1.g), static_cast<GLfloat>(color1.b), 1.0f);

        mySquare.setTranslate(0.5f, 0.5f);
        mySquare.setRotation(static_cast<GLfloat>(myRotation));

        Matrix model = mySquare.getMatrix();
        glUniformMatrix4fv(myModelViewUniform, 1, GL_FALSE, (model * view).matrix);

        mySquare.bindVAO();

        glDrawArrays(GL_TRIANGLES, 0, mySquare.getElementCount() / 3);

        // Draw the chevron

        glUniform4f(myColorUniform, static_cast<GLfloat>(color2.r), static_cast<GLfloat>(color2.g), static_cast<GLfloat>(color2.b), 1.0f);

        myChevron.setScale(0.8f, 0.8f);
        myChevron.setTranslate(-0.5, -0.5);
        myChevron.setRotation(static_cast<GLfloat>(myRotation));

        model = myChevron.getMatrix();
        glUniformMatrix4fv(myModelViewUniform, 1, GL_FALSE, (model * view).matrix);

        myChevron.bindVAO();

        glDrawArrays(GL_TRIANGLES, 0, myChevron.getElementCount() / 3);

        // Tidy up

        glBindVertexArray(0);
        glUseProgram(0);
    }

    context->endGLCommands();
}

int32_t
OpenGLTOP::getNumInfoCHOPChans(void * reserved1)
{
	// We return the number of channel we want to output to any Info CHOP
	// connected to the TOP. In this example we are just going to send one channel.
	return 2;
}

void
OpenGLTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void * reserved1)
{
	// This function will be called once for each channel we said we'd want to return
	// In this example it'll only be called once.

	if (index == 0)
	{
		chan->name->setString("executeCount");
		chan->value = (float)myExecuteCount;
	}

	if (index == 1)
	{
		chan->name->setString("rotation");
		chan->value = (float)myRotation;
	}
}

bool		
OpenGLTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved1)
{
	infoSize->rows = 2;
	infoSize->cols = 2;
	// Setting this to false means we'll be assigning values to the table
	// one row at a time. True means we'll do it one column at a time.
	infoSize->byColumn = false;
	return true;
}

void
OpenGLTOP::getInfoDATEntries(int32_t index,
										int32_t nEntries,
										OP_InfoDATEntries* entries,
										void* reserved1)
{
	char tempBuffer[4096];

	if (index == 0)
	{
		// Set the value for the first column
#ifdef _WIN32
		strcpy_s(tempBuffer, "executeCount");
#else // macOS
        strlcpy(tempBuffer, "executeCount", sizeof(tempBuffer));
#endif
		entries->values[0]->setString(tempBuffer);

		// Set the value for the second column
#ifdef _WIN32
		sprintf_s(tempBuffer, "%d", myExecuteCount);
#else // macOS
        snprintf(tempBuffer, sizeof(tempBuffer), "%d", myExecuteCount);
#endif
		entries->values[1]->setString(tempBuffer);
	}

	if (index == 1)
	{
		// Set the value for the first column
#ifdef _WIN32
		strcpy_s(tempBuffer, "rotation");
#else // macOS
		strlcpy(tempBuffer, "rotation", sizeof(tempBuffer));
#endif
		entries->values[0]->setString(tempBuffer);

		// Set the value for the second column
#ifdef _WIN32
		sprintf_s(tempBuffer, "%g", myRotation);
#else // macOS
		snprintf(tempBuffer, sizeof(tempBuffer), "%g", myRotation);
#endif
		entries->values[1]->setString(tempBuffer);
	}
}

void
OpenGLTOP::getErrorString(OP_String *error, void* reserved1)
{
	error->setString(myError);
}

void
OpenGLTOP::setupParameters(OP_ParameterManager* manager, void* reserved1)
{
	myParms.setup(manager);
}

void
OpenGLTOP::pulsePressed(const char* name, void* reserved1)
{
	if (!strcmp(name, "Reset"))
	{
		myRotation = 0.0;
	}
}

void OpenGLTOP::setupGL()
{
    if (myDidSetup == false)
    {
        myError = myProgram.build(vertexShader, fragmentShader);

        // If an error occurred creating myProgram, we can't proceed
        if (myError == nullptr)
        {
            GLint vertAttribLocation = glGetAttribLocation(myProgram.getName(), "P");
            myModelViewUniform = glGetUniformLocation(myProgram.getName(), "uModelView");
            myColorUniform = glGetUniformLocation(myProgram.getName(), "uColor");

            if (vertAttribLocation == -1 || myModelViewUniform == -1 || myColorUniform == -1)
            {
                myError = uniformError;
            }

            // Set up our two shapes
            GLfloat square[] = {
                -0.5, -0.5, 1.0,
                0.5, -0.5, 1.0,
                -0.5,  0.5, 1.0,

                0.5, -0.5, 1.0,
                0.5,  0.5, 1.0,
                -0.5,  0.5, 1.0
            };

            mySquare.setVertices(square, 2 * 9);
            mySquare.setup(vertAttribLocation);

            GLfloat chevron[] = {
                -1.0, -1.0,  1.0,
                -0.5,  0.0,  1.0,
                0.0, -1.0,  1.0,

                -0.5,  0.0,  1.0,
                0.5,  0.0,  1.0,
                0.0, -1.0,  1.0,

                0.0,  1.0,  1.0,
                0.5,  0.0,  1.0,
                -0.5,  0.0,  1.0,

                -1.0,  1.0,  1.0,
                0.0,  1.0,  1.0,
                -0.5,  0.0,  1.0
            };

            myChevron.setVertices(chevron, 4 * 9);
            myChevron.setup(vertAttribLocation);
        }

        myDidSetup = true;
    }
}
