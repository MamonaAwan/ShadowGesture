//opencv//
#include <opencv\cv.h>
#include <copencv\xcore.h>
#include <opencv\highgui.h>
//opengl//
#include <gl/glut.h>
#include <gl/GL.H>
#include <gl/GLU.H>
#include <gl/GLAUX.H>
//basic//
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "BlobLabeling.h"

////sound//
//#include <Mmsystem.h> 
//#pragma comment(lib, "Winmm.lib" )
////선언//
//MCI_OPEN_PARMS mciOpen;
//MCI_PLAY_PARMS mciPlay;

using namespace std;

#define INIT_TIME 20
#define ZETA 10

//함수들//
void gldraw();
void initGL(void); // opengl 초기화 함수 및 texture 불러들이는 함수
double round(double number, int nPos);
void PlayMusic(int tone);

//변수들//
bool BlobFlag = false;
bool colorFlag = false;
int counter = 0;
bool startflag = false;
GLint countX = 0;
GLint countY = 0;
bool flag_x = false;
bool flag_y = false;
bool moveStart = false;
GLint _width = 640;
GLint _height = 480;
GLfloat pix_x = 0;
GLfloat pix_y = 0;
GLint pointCnt = 0;
GLint pixCnt = 0;
GLint bufferCnt = 0;
GLint Rotate_x = 0;
GLint Rotate_y = 0;
GLint Rotate_z = 0;
GLint rad;
GLint mode = 0;

GLint Tmpcnt1 = 0;
GLint Tmpcnt2 = 0;
int color=0;
bool colorflag=0;


GLfloat Trans_x = 0;
GLfloat Trans_y = 0;
GLfloat pointX[307200];
GLfloat pointY[307200];
GLfloat gl_x1;
GLfloat gl_y1;
GLfloat gl_x2;
GLfloat gl_y2;
GLfloat diffDis;

CvPoint recUp,recDown;
CvPoint pt1,pt2;
CvPoint pt_line;

CvScalar pixel;
CvScalar lineColor = cvScalar(0,0,0);;
CBlobLabeling blob;
CvRect hand;
CvRect tmpHand;

//// convex hull
CvMemStorage* storage = cvCreateMemStorage(0);
CvSeq* contours;
CvPoint pt0;
CvPoint end_pt;
CvPoint MaxDist_pt;
CvPoint tmpDist_pt;
CvPoint hull_pt;
CvSeq* hull;
int Threshold = 100;
double tmpDist;
double MaxDist =0.0;
int rectsize = 150;
CvConvexityDefect *Item;

//optical flow 변수
CvSize winSize;
char status;
CvPoint2D32f featureLeft[2], featureRight[2];
float point_distance[2];
CvPoint ptSelected[2];

//Moment 변수 선언
CvMoments moments;
CvHuMoments huMoments;
double M;
int x_order, y_order;
double centerX;
double centerY;
double m00;
float x_plus=0.2;
float y_plus=0.3;

//// texture
GLuint texture[1];

// 초기 카메라 입력 변수
IplImage* imgCam;
CvCapture* capture = cvCaptureFromCAM(1);

//배경 연산 시 사용되는 변수
IplImage *imgAv = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_32F, 3);
IplImage *imgSgm = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_32F, 3);
IplImage *imgTmp = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_32F, 3);
IplImage *imgLower = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_32F, 3);
IplImage *imgUpper = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_32F, 3);
IplImage *imgMsk   = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *drawMsk   = cvCreateImage (cvSize(_width*2, _height*2), IPL_DEPTH_8U, 1);
IplImage *drawCol   = cvCreateImage (cvSize(_width*2, _height*2), IPL_DEPTH_8U, 3);
IplImage *imgInput = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 3);

//이미지 처리 과정에 사용되는 변수
IplImage *imgGray = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *imgCanny = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *imgBackgnd = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *imgAnd = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *imgBinary = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *subHand = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *subtmpHand = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *imgCircle = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *imgPre = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);
IplImage *imgDiff = cvCreateImage (cvSize(_width, _height), IPL_DEPTH_8U, 1);

//opengl 연산 시 사용 되는 변수
IplImage *img_gl = cvCreateImage (cvSize(_width*2, _height*2), IPL_DEPTH_8U, 3);
IplImage *mask   = cvCreateImage (cvSize(_width*2, _height*2), IPL_DEPTH_8U, 1);
IplImage *eraser   = cvCreateImage (cvSize(_width*2, _height*2), IPL_DEPTH_8U, 3);

//최종 Display 변수
IplImage *final = cvCreateImage (cvSize(_width*2, _height*2), IPL_DEPTH_8U, 3);

//구조체
AUX_RGBImageRec *LoadBMP(char *Filename)				// Loads A Bitmap Image
{
	FILE *File=NULL;									// File Handle
	if (!Filename)										// Make Sure A Filename Was Given
	{
		return NULL;									// If Not Return NULL
	}

	File=fopen(Filename,"r");							// Check To See If The File Exists

	if (File)											// Does The File Exist?
	{
		fclose(File);									// Close The Handle
		return auxDIBImageLoad(Filename);				// Load The Bitmap And Return A Pointer
	}

	return NULL;										// If Load Failed Return NULL
}