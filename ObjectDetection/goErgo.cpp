#include "stdafx.h"
#include "windows.h"


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


void detectAndDisplay(Mat mat_frame);

String face_cascade_name = "C:/opencv/OpenCV/sources/data/haarcascades/haarcascade_frontalface_alt2.xml";
String eyes_cascade_name = "C:/opencv/OpenCV/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

VideoCapture capture;
unsigned int stime, etime;
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;
int rSig, ypos;
int trigDist = 0;
int trigHeight = 0;
bool alm, pause;
int		text_delay, stage = STAGE_INIT;
CvSeq*	comp = 0;
CvRect	window, eye;
int key, found;
ULONG64 blink_count;
FILETIME st, et;

using namespace cv;
using namespace std;

Mat mat_frame;
IplImage*		frame, *gray, *diff, *tpl;
IplImage* previous = NULL;
CvMemStorage*	storage;
IplConvKernel*	kernel;
CvFont			font;
char*			wnd_name = "video";
char*			wnd_debug = "diff";

int  get_connected_components(IplImage* img, IplImage* prev_img, CvRect window, CvSeq** comp);
int	 is_eye_pair(CvSeq* comp, int num, CvRect* eye);
int  locate_eye(IplImage* img, IplImage* tpl, CvRect* window, CvRect* eye);
int	 is_blink(CvSeq* comp, int num, CvRect window, CvRect eye);
void delay_frames(int nframes);
int init();
void exit_nicely(char* msg);

void
delay_frames(int nframes)
{
	int i;

	for (i = 0; i < nframes; i++)
	{
		capture.read(mat_frame);
		if (frame) delete frame;
		frame = new IplImage(mat_frame);
		if (!frame)
			exit_nicely("cannot query frame");
		cvShowImage(wnd_name, frame);
		if (diff)
			cvShowImage(wnd_debug, diff);
		cvWaitKey(30);
	}
}

int
is_eye_pair(CvSeq* comp, int num, CvRect* eye)
{
	if (comp == 0 || num != 2)
		return 0;

	CvRect r1 = cvBoundingRect(comp, 1);
	comp = comp->h_next;

	if (comp == 0)
		return 0;

	CvRect r2 = cvBoundingRect(comp, 1);

	/* the width of the components are about the same */
	if (abs(r1.width - r2.width) >= 5)
		return 0;

	/* the height f the components are about the same */
	if (abs(r1.height - r2.height) >= 5)
		return 0;

	/* vertical distance is small */
	if (abs(r1.y - r2.y) >= 5)
		return 0;

	/* reasonable horizontal distance, based on the components' width */
	int dist_ratio = abs(r1.x - r2.x) / r1.width;
	if (dist_ratio < 2 || dist_ratio > 5)
		return 0;

	/* get the centroid of the 1st component */
	CvPoint point = cvPoint(
		r1.x + (r1.width / 2),
		r1.y + (r1.height / 2)
		);

	/* return eye boundaries */
	*eye = cvRect(
		point.x - (TPL_WIDTH / 2),
		point.y - (TPL_HEIGHT / 2),
		TPL_WIDTH,
		TPL_HEIGHT
		);

	return 1;
}

/**
* Locate the user's eye with template matching
*
* @param	IplImage* img     the source image
* @param	IplImage* tpl     the eye template
* @param	CvRect*   window  search within this window,
*                            will be updated with the recent search window
* @param	CvRect*   eye     output parameter, will contain the current
*                            location of user's eye
* @return	int               '1' if found, '0' otherwise
*/
int
locate_eye(IplImage* img, IplImage* tpl, CvRect* window, CvRect* eye)
{
	IplImage*	tm;
	CvRect		win;
	CvPoint		minloc, maxloc, point;
	double		minval, maxval;
	int			w, h;

	/* get the centroid of eye */
	point = cvPoint(
		(*eye).x + (*eye).width / 2,
		(*eye).y + (*eye).height / 2
		);

	/* setup search window
	replace the predefined WIN_WIDTH and WIN_HEIGHT above
	for your convenient */
	win = cvRect(
		point.x - WIN_WIDTH / 2,
		point.y - WIN_HEIGHT / 2,
		WIN_WIDTH,
		WIN_HEIGHT
		);

	/* make sure that the search window is still within the frame */
	if (win.x < 0)
		win.x = 0;
	if (win.y < 0)
		win.y = 0;
	if (win.x + win.width > img->width)
		win.x = img->width - win.width;
	if (win.y + win.height > img->height)
		win.y = img->height - win.height;

	/* create new image for template matching result where:
	width  = W - w + 1, and
	height = H - h + 1 */
	w = win.width - tpl->width + 1;
	h = win.height - tpl->height + 1;
	tm = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 1);

	/* apply the search window */
	cvSetImageROI(img, win);

	/* template matching */
	cvMatchTemplate(img, tpl, tm, CV_TM_SQDIFF_NORMED);
	cvMinMaxLoc(tm, &minval, &maxval, &minloc, &maxloc, 0);

	/* release things */
	cvResetImageROI(img);
	cvReleaseImage(&tm);

	/* only good matches */
	if (minval > TM_THRESHOLD)
		return 0;

	/* return the search window */
	*window = win;

	/* return eye location */
	*eye = cvRect(
		win.x + minloc.x,
		win.y + minloc.y,
		TPL_WIDTH,
		TPL_HEIGHT
		);

	return 1;
}

int
is_blink(CvSeq* comp, int num, CvRect window, CvRect eye)
{
	if (comp == 0 || num != 1)
		return 0;

	CvRect r1 = cvBoundingRect(comp, 1);

	/* component is within the search window */
	if (r1.x < window.x)
		return 0;
	if (r1.y < window.y)
		return 0;
	if (r1.x + r1.width > window.x + window.width)
		return 0;
	if (r1.y + r1.height > window.y + window.height)
		return 0;

	/* get the centroid of eye */
	CvPoint pt = cvPoint(
		eye.x + eye.width / 2,
		eye.y + eye.height / 2
		);

	/* component is located at the eye's centroid */
	if (pt.x <= r1.x || pt.x >= r1.x + r1.width)
		return 0;
	if (pt.y <= r1.y || pt.y >= r1.y + r1.height)
		return 0;

	return 1;
}


/**
* Initialize images, memory, and windows
*/
int
init()
{
	//GetSystemTimeAsFileTime(&st);
	stime = GetTickCount();
	face_cascade = cv::CascadeClassifier::CascadeClassifier(face_cascade_name);
	eyes_cascade = cv::CascadeClassifier::CascadeClassifier(eyes_cascade_name);

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		//printf("--(!)Error loading face cascade\n");
		return -1; 
	}
	if (!eyes_cascade.load(eyes_cascade_name)) {
		//printf("--(!)Error loading eyes cascade\n");
		return -1; 
	}

	//-- 2. Read the video stream

	capture = VideoCapture::VideoCapture();
	capture.open(0);
	if (!capture.isOpened()) {
		//printf("--(!)Error opening video capture\n"); return -1; 
		exit_nicely("Cannot initialize camera!");
	}

	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

	capture.read(mat_frame);
	if (frame) delete frame;
	frame = new IplImage(mat_frame);
	if (!frame)
		exit_nicely("cannot query frame!");

	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.4, 0.4, 0, 1, 8);
	cvNamedWindow(wnd_name, 1);

	capture.read(mat_frame);
	if (frame) delete frame;
	frame = new IplImage(mat_frame);
	if (!frame)
		exit_nicely("cannot query frame!");
	//DRAW_TEXT(frame, msg[i], delay, 0);
	cvShowImage(wnd_name, frame);
	cvWaitKey(30);

	storage = cvCreateMemStorage(0);
	if (!storage)
		exit_nicely("cannot allocate memory storage!");

	kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);
	gray = cvCreateImage(cvGetSize(frame), 8, 1);
	previous = cvCreateImage(cvGetSize(frame), 8, 1);
	diff = cvCreateImage(cvGetSize(frame), 8, 1);
	tpl = cvCreateImage(cvSize(TPL_WIDTH, TPL_HEIGHT), 8, 1);

	if (!kernel || !gray || !previous || !diff || !tpl)
		exit_nicely("system error.");

	gray->origin = frame->origin;
	previous->origin = frame->origin;
	diff->origin = frame->origin;

	cvNamedWindow(wnd_debug, 1);
	return 1;
}

/**
* This function provides a way to exit nicely
* from the system
*
* @param char* msg error message to display
*/
void
exit_nicely(char* msg)
{
	cvDestroyAllWindows();
	capture.release();

	if (gray)
		cvReleaseImage(&gray);
	if (previous)
		cvReleaseImage(&previous);
	if (diff)
		cvReleaseImage(&diff);
	if (tpl)
		cvReleaseImage(&tpl);
	if (storage)
		cvReleaseMemStorage(&storage);

	if (msg != NULL)
	{
		//fprintf(stderr, msg);
		//fprintf(stderr, "\n");
		exit(1);
	}

	exit(0);
}

void draw_rects(IplImage *f, IplImage *d, CvRect rw, CvRect ro)
{

	do {
		if (alm)
			cvRectangle(f, cvPoint(rw.x, rw.y), cvPoint(rw.x + rw.width, rw.y + rw.height), CV_RGB(255, 0, 0), 2, 8, 0);
		else
			cvRectangle(f, cvPoint(rw.x, rw.y), cvPoint(rw.x + rw.width, rw.y + rw.height), CV_RGB(0, 255, 0), 1, 8, 0);
		cvRectangle(f, cvPoint(ro.x,ro.y), cvPoint(ro.x + ro.width, ro.y + ro.height), CV_RGB(0, 0, 255), 1, 8, 0);
		cvRectangle(d, cvPoint(rw.x, rw.y), cvPoint(rw.x + rw.width, rw.y + rw.height), cvScalarAll(255), 1, 8, 0);
		cvRectangle(d, cvPoint(ro.x,ro.y), cvPoint(ro.x + ro.width, ro.y + ro.height), cvScalarAll(255), 1, 8, 0);
	} while (0);
}
void getDistance(Mat mat_frame, int interval) 
{ //OpenCV functions

	//pushmatrix and popmatrix prevents the buttons from beeing scaled with the video
	//if (alm) stroke(255, 0, 0); //draw all lines red if alarm is active
	//else stroke(0, 255, 0);
	//strokeWeight(2);
	//Rectangle[] faces = opencv.detect();
	std::vector<Rect> faces;
	Mat mat_frame_gray;

	cvtColor(mat_frame, mat_frame_gray, COLOR_BGR2GRAY);
	equalizeHist(mat_frame_gray, mat_frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(mat_frame_gray, faces, 1.1, 3, 0, Size(30, 30));


	int dist = 0;
	for (int i = 0; i < faces.size(); i++) {
		//printf(faces[i].x + "," + faces[i].y);
		cv::rectangle(mat_frame, faces[i], CV_RGB(0, 255, 0));
		rSig = faces[i].height;
		ypos = faces[i].y;
		int delta = trigDist - faces[i].height;
		//the following line draws a second box with the limit distance
		if (trigDist != 0)
		{
			cv::Rect  deltaRect = cv::Rect(faces[i].x - delta / 2, faces[i].y - delta / 2, faces[i].width + delta, trigDist);
			/*
			if (alm)
				cv::rectangle(mat_frame, deltaRect, CV_RGB(255, 0, 0), 2, 8,0);
			else
				cv::rectangle(mat_frame, deltaRect, CV_RGB(0, 255, 0), 2, 8, 0);
				*/
		}
			//rect(faces[i].x - delta / 2, faces[i].y - delta / 2, faces[i].width + delta, trigDist);
	}
	//This draws a line at the limit height:
	//if (trigHeight != 0) line(0, trigHeight, width, trigHeight);
	//popMatrix();
}

void getProximity(Mat mat_frame)
{
	getDistance(mat_frame,0);  // run the OpenCV routine
	if (trigHeight != 0 && trigDist != 0 && !pause) { //check if limits have been initialized
		//and if pause is off
		if (rSig > trigDist || ypos > trigHeight) { //compare values to limits
			alm = true;
		}
		else {
			alm = false;
		}
	}

	/*
	if (alm == false) almTimer = millis() + 2000;  //reset alarm timer if alarm is off
	else if (millis() > almTimer) { //check if alarm timer has expired
		if (millis() - 2000 < almTimer) { //do this for additional 2 seconds
			Toolkit.getDefaultToolkit().beep(); //call the windows alarm sound
			delay(150);
		}
	}*/
}

int configureDefaults(Mat *mat_frame)
{
	trigDist = rSig + 3;
	trigHeight = ypos + 3;

	return 1;
}

int
get_connected_components(IplImage* img, IplImage* prev_img, CvRect window, CvSeq** comp)
{
	IplImage* _diff;

	cvZero(diff);

	/* apply search window to images */
	cvSetImageROI(img, window);
	cvSetImageROI(prev_img, window);
	cvSetImageROI(diff, window);

	/* motion analysis */
	cvSub(img, prev_img, diff, NULL);
	cvThreshold(diff, diff, 5, 255, CV_THRESH_BINARY);
	cvMorphologyEx(diff, diff, NULL, kernel, CV_MOP_OPEN, 1);

	/* reset search window */
	cvResetImageROI(img);
	cvResetImageROI(prev_img);
	cvResetImageROI(diff);

	_diff = (IplImage*)cvClone(diff);

	/* get connected components */
	int nc = cvFindContours(_diff, storage, comp, sizeof(CvContour),
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	cvClearMemStorage(storage);
	cvReleaseImage(&_diff);

	return nc;
}

int detectBlink(IplImage *frame)
{
	if (!frame)
		exit_nicely("cannot query frame!");
	frame->origin = 0;

	if (stage == STAGE_INIT)
		window = cvRect(0, 0, frame->width, frame->height);

	cvCvtColor(frame, gray, CV_BGR2GRAY);

	int nc = get_connected_components(gray, previous, window, &comp);

		if (stage == STAGE_INIT && is_eye_pair(comp, nc, &eye))
		{
			delay_frames(5);

			cvSetImageROI(gray, eye);
			cvCopy(gray, tpl, NULL);
			cvResetImageROI(gray);

			stage = STAGE_TRACKING;
			text_delay = 10;
		}

		if (stage == STAGE_TRACKING)
		{
			found = locate_eye(gray, tpl, &window, &eye);

			if (!found || key == 'r')
				stage = STAGE_INIT;

			if (is_blink(comp, nc, window, eye)) {
				++blink_count;
				text_delay = 1;
			}

			draw_rects(frame, diff, window, eye);
			//DRAW_TEXT(frame, "blink!", text_delay, 1);
		}

		cvShowImage(wnd_name, frame);
		cvShowImage(wnd_debug, diff);
		previous = (IplImage*)cvClone(gray);
		//key = cvWaitKey(15);

		return 1;

}
int main(void)
{
	Mat frame;
	init();
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			//printf(" --(!) No captured frame -- Break!");
			break;
		}

		IplImage *iplImgFrame = new IplImage(frame);
		detectAndDisplay(frame);
		detectBlink(iplImgFrame);

		int c = cvWaitKey(15);
		// escape
		int ll_millisec = 0;
		if ((char)c == 27) { 
			//GetSystemTimeAsFileTime(&et);
			//int ll_millisec = ((LONGLONG)et.dwLowDateTime + ((LONGLONG)(et.dwHighDateTime) << 32LL) - (LONGLONG)st.dwLowDateTime + ((LONGLONG)(st.dwHighDateTime) << 32LL))/10000;
			etime = GetTickCount();
			ll_millisec = etime - stime;
			printf("%ld blinks in %ld milliseconds at rate of %f blinks per minute", blink_count, ll_millisec, (blink_count*60.0*1000.0)/(ll_millisec));
			break;
		cvWaitKey(-1);
		} else if ((char)c == 'c') { 
			configureDefaults(&frame);
		}
	}
	return 0;
}

void detectAndDisplay(Mat mat_frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(mat_frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0, Size(30, 30));
	getProximity(mat_frame); 


	/*
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(mat_frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.05, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(mat_frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}
	*/
	//-- Show what you got
	//imshow(window_name, mat_frame);
	if (frame) delete frame;
	frame = new IplImage(mat_frame);
	cvShowImage(wnd_name, frame);
}

