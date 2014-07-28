
#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

#define FRAME_WIDTH		240
#define FRAME_HEIGHT	180
#define TPL_WIDTH 		16
#define TPL_HEIGHT 		12
#define WIN_WIDTH		TPL_WIDTH * 2
#define WIN_HEIGHT		TPL_HEIGHT * 2
#define TM_THRESHOLD	0.4
#define STAGE_INIT		1
#define STAGE_TRACKING	2

#define POINT_TL(r)		cvPoint(r.x, r.y)
#define POINT_BR(r)		cvPoint(r.x + r.width, r.y + r.height)
#define POINTS(r)		POINT_TL(r), POINT_BR(r)

void DRAW_RECTS(IplImage *f, IplImage *d, CvRect rw, CvRect ro)
{

	do {
		cvRectangle(f, cvPoint(rw.x, rw.y), cvPoint(rw.x + rw.width, rw.y + rw.height), CV_RGB(255, 0, 0), 1, 8, 0);
		cvRectangle(f, POINTS(ro), CV_RGB(0, 255, 0), 1, 8, 0);
		cvRectangle(d, POINTS(rw), cvScalarAll(255), 1, 8, 0);
		cvRectangle(d, POINTS(ro), cvScalarAll(255), 1, 8, 0);
	} while (0);
}

#define	DRAW_TEXT(f, t, d, use_bg)								\
if (d)															\
{																\
	CvSize _size;												\
	cvGetTextSize(t, &font, &_size, NULL);						\
if (use_bg)													\
{															\
	cvRectangle(f, cvPoint(0, f->height), \
	cvPoint(_size.width + 5, \
	f->height - _size.height * 2), \
	CV_RGB(255, 0, 0), CV_FILLED, 8, 0);		\
}															\
	cvPutText(f, t, cvPoint(2, f->height - _size.height / 2), \
	&font, CV_RGB(255, 255, 0));						\
	d--;														\
}


void exit_nicely(char* msg);