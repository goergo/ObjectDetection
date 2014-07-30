#include "stdafx.h"
#include "windows.h"


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;


#define SCALE_FACTOR 1.0

#define DllExport   __declspec( dllexport ) 
#define FRAME_WIDTH		240*SCALE_FACTOR
#define FRAME_HEIGHT	180*SCALE_FACTOR
#define TPL_WIDTH 		16*SCALE_FACTOR
#define TPL_HEIGHT 		12*SCALE_FACTOR
#define WIN_WIDTH		TPL_WIDTH * 2 
#define WIN_HEIGHT		TPL_HEIGHT * 2 
#define TM_THRESHOLD	0.4
#define STAGE_INIT		1
#define STAGE_TRACKING	2

#define POINT_TL(r)		cvPoint(r.x, r.y)
#define POINT_BR(r)		cvPoint(r.x + r.width, r.y + r.height)
#define POINTS(r)		POINT_TL(r), POINT_BR(r)

bool detectProximityAndEye(Mat mat_frame);

typedef struct stats
{
	unsigned long total_time;
	unsigned long total_blink;
	unsigned long total_time_alarm_ambient;
	unsigned long total_time_alarm_proximity;
	unsigned int total_time_alarm_position;
}stats_t;

typedef struct posture
{
	unsigned int clock_ticks;
	FILETIME time;
	CvRect faceRect, eyeRect;
	bool alarm_ambient_light;
	bool alarm_proximity;
	bool alaram_sitting_posture;
	int eye_blinks;

} posture_t;

posture_t curr_posture;
posture_t buf_posture;
int buf_num = 0;
std::vector<posture_t> posture_vec;
std::list<posture_t> posture_with_blink_list;

String face_cascade_name = "C:/opencv/OpenCV/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
//String face_cascade_name = "C:/opencv/OpenCV/sources/data/haarcascades/haarcascade_profileface.xml";
String eyes_cascade_name = "C:/opencv/OpenCV/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
//String eyes_cascade_name = "C:/opencv/OpenCV/sources/data/haarcascades/haarcascade_mcs_eyepair_small.xml";
//String eyes_cascade_name = "C:/opencv/OpenCV/sources/data/haarcascades/haarcascade_mcs_lefteye.xml";

[event_source(native)]
class Csource {
public:
	__event void BlinkEvent(int nvalue);
};
VideoCapture capture;
unsigned int stime, ticks;
unsigned long total_ticks;
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;
int rSig, ypos;
int trigDist = 0;
int trigHeight = 0;
bool proximity_alm, ambient_light_alm, pause;
int		text_delay, stage = STAGE_INIT;
CvSeq*	comp = 0;
CvRect	window, eye, eye2;
int key, found;
ULONG64 total_blink_count;
unsigned int blink_count;
FILETIME st, currt;
int avg_luminance = 0;
int threshold_luminance = 0;
FILE* pFile;

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
void exit_nicely(char* msg);

 extern "C"  __declspec( dllexport )int statsdumps()
{
	char buf[1024];
	unsigned long total_time = 0;
	unsigned long total_blink = 0;
	unsigned long total_time_alarm_ambient = 0;
	unsigned long total_time_alarm_proximity = 0;
	unsigned int total_time_alarm_position = 0;
	for (int i = 1; i < posture_vec.size(); ++i) //ignore the first frame
	{
		total_time += posture_vec[i].clock_ticks;
		//total_blink += posture_vec[i].eye_blinks;
		if (posture_vec[i].alarm_proximity)
			total_time_alarm_proximity += posture_vec[i].clock_ticks;
		if (posture_vec[i].alarm_ambient_light)
			total_time_alarm_ambient += posture_vec[i].clock_ticks;
	}
	stats_t stats;
	stats.total_blink = total_blink_count;
	stats.total_time_alarm_ambient = total_time_alarm_ambient;
	stats.total_time = total_time;
	stats.total_time_alarm_position = total_time_alarm_position;
	stats.total_time_alarm_proximity = total_time_alarm_proximity;

	float perc_prox = (float)total_time_alarm_proximity * 100.0 / (float)total_time;
	float perc_ambi = (float)total_time_alarm_ambient * 100.0 / (float)total_time;

	sprintf(buf, "total time:%d\ntotal blinks:%d\n %%age time wrong proximity:%f\n %%age time wrong ambient light:%f\n",
		total_time/1000,total_blink_count, perc_prox, perc_ambi );
	fputs(buf, pFile);
	/*
	std::string stats_str;
	stats_str = "total blinks:" + total_blink + '\n';
	stats_str = "total time wrong proximity" + total_time_alarm_proximity + '\n';
	stats_str = "total time wrong ambient light" + total_time_alarm_ambient + '\n';
	fwrite(&stats, sizeof(stats_t), 1, pFile);
	*/
	return 1;
}

 extern "C"  __declspec(dllexport)int get_stats(int *blink, int *ambient_alarm, int *posture_alarm, int use_buf = 1 )
 {
	 if (!posture_with_blink_list.empty())
	 {
		 posture_t last_posture =posture_with_blink_list.front();
		 posture_with_blink_list.pop_front();
		 *blink = last_posture.eye_blinks;
		 *ambient_alarm = last_posture.alarm_ambient_light;
		 *posture_alarm = last_posture.alarm_proximity;
		 buf_num = 0;
		 return 1;
	 }
	 if (use_buf) {
		 if (buf_num > 0) 
		 {
			 *blink = buf_posture.eye_blinks;
			 *ambient_alarm = buf_posture.alarm_ambient_light;
			 *posture_alarm = buf_posture.alarm_proximity;
			 buf_num = 0;
			 return 1;
		 }
	 }
	 *blink = curr_posture.eye_blinks;
	 *ambient_alarm = curr_posture.alarm_ambient_light;
	 *posture_alarm = curr_posture.alarm_proximity;
	 buf_num = 0;
	 return 1;

 }
void draw_text(Mat &f, const char *t, int &d, bool use_bg)
{
	int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	int baseline = 0;
	double fontscale = 0.4;
	int thickness = 0.4;
	if (d)
	{
		cv::Size _size = cv::getTextSize(t, fontFace, fontscale, thickness, &baseline);
		if (use_bg)
		{
			cv::rectangle(f, cvPoint(0, f.cols),
				cvPoint(_size.width + 5,
				f.rows - _size.height * 2),
				CV_RGB(255, 0, 0), CV_FILLED, 8, 0);
		}
		cv::putText(f, t, cvPoint(2, f.rows - _size.height / 2), fontFace, fontscale, CV_RGB(255, 255, 0), thickness, 8);
		d--;
	}
}



void draw_text(IplImage *f, const char *t, int &d, bool use_bg)
{
	if (d)
	{
		CvSize _size;
		cvGetTextSize(t, &font, &_size, NULL);
		if (use_bg)
		{
			cvRectangle(f, cvPoint(0, f->height),
				cvPoint(_size.width + 5,
				f->height - _size.height * 2),
				CV_RGB(255, 0, 0), CV_FILLED, 8, 0);
		}
		cvPutText(f, t, cvPoint(2, f->height - _size.height / 2), &font, CV_RGB(255, 255, 0));
		d--;
	}
}

void delay_frames(int nframes)
{
	int i;

	for (i = 0; i < nframes; i++)
	{
		capture.read(mat_frame);
		if (frame)
			delete frame;
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
	if (abs(r1.width - r2.width) >= 5*SCALE_FACTOR)
		return 0;

	/* the height f the components are about the same */
	if (abs(r1.height - r2.height) >= 5*SCALE_FACTOR)
		return 0;

	/* vertical distance is small */
	if (abs(r1.y - r2.y) >= 5*SCALE_FACTOR)
		return 0;

	/* reasonable horizontal distance, based on the components' width */
	int dist_ratio = abs(r1.x - r2.x) / r1.width;
	if (dist_ratio < 2/SCALE_FACTOR || dist_ratio > 5*SCALE_FACTOR)
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
extern "C"  __declspec( dllexport ) int init()
{
	pFile = fopen("stats.txt", "a+");
	GetSystemTimeAsFileTime(&st);
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
	if (frame)
		delete frame;
	frame = new IplImage(mat_frame);
	if (!frame)
		exit_nicely("cannot query frame!");

	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.4, 0.4, 0, 1, 8);
	cvNamedWindow(wnd_name, 1);
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
		if (proximity_alm)
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
	//if (proximity_alm) stroke(255, 0, 0); //draw all lines red if alarm is active
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
			curr_posture.faceRect = faces[0];
			cv::Rect  deltaRect = cv::Rect(faces[i].x - delta / 2, faces[i].y - delta / 2, faces[i].width + delta, trigDist);
			if (proximity_alm) {
				curr_posture.alarm_proximity = true;
				buf_num = 10;
				cv::rectangle(mat_frame, deltaRect, CV_RGB(255, 0, 0), 2, 8,0);
			}
			else {
				curr_posture.alarm_proximity = false;
				cv::rectangle(mat_frame, deltaRect, CV_RGB(0, 255, 0), 2, 8, 0);
			}
		}
			//rect(faces[i].x - delta / 2, faces[i].y - delta / 2, faces[i].width + delta, trigDist);
	}
	//if (trigHeight != 0) line(0, trigHeight, width, trigHeight);
}

int getProximity(Mat mat_frame)
{
	getDistance(mat_frame,0);  // run the OpenCV routine
	if (trigHeight != 0 && trigDist != 0 && !pause) { //check if limits have been initialized
		//and if pause is off
		if (rSig > trigDist || ypos > trigHeight) { //compare values to limits
			proximity_alm = true;
		}
		else {
			proximity_alm = false;
		}
	}
		return (int)proximity_alm;
}

int configureDefaults(Mat &mat_frame)
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

int detectBlink(Mat &mat_frame, IplImage *frame, bool eye_hint)
{
	Csource source;
	if (!frame)
		exit_nicely("cannot query frame!");
	frame->origin = 0;

	if (stage == STAGE_INIT)
		window = cvRect(0, 0, frame->width, frame->height);

	cvCvtColor(frame, gray, CV_BGR2GRAY);

	int nc = get_connected_components(gray, previous, window, &comp);

		if (stage == STAGE_INIT)
		{
			if (is_eye_pair(comp, nc, &eye))
			{
				delay_frames(5);

				//cvReleaseImage(&tpl);
				//tpl = cvCreateImage(cvSize(eye.width, eye.height), 8, 1);
				cvSetImageROI(gray, eye);
				cvCopy(gray, tpl, NULL);
				cvResetImageROI(gray);

				curr_posture.eyeRect = eye;
				stage = STAGE_TRACKING;
				text_delay = 0;
			}
			else if (eye_hint) {
				//try deducing eye using eye hint from classifier output
				eye = eye2;

				delay_frames(5);

				//cvReleaseImage(&tpl);
				//tpl = cvCreateImage(cvSize(eye.width, eye.height), 8, 1);
				cvSetImageROI(gray, eye);
				cvCopy(gray, tpl, NULL);
				cvResetImageROI(gray);

				//nc = get_connected_components(gray, previous, eye, &comp);

				curr_posture.eyeRect = eye;
				stage = STAGE_TRACKING;
				text_delay = 0;
			}
		}

		if (stage == STAGE_TRACKING)
		{
			found = locate_eye(gray, tpl, &window, &eye);

			if (!found || key == 'r')
				stage = STAGE_INIT;

			if (is_blink(comp, nc, window, eye)) {
				__raise source.BlinkEvent(1);
				++blink_count;
				++total_blink_count;
				buf_num = 20;
				text_delay = 10;
			}

			curr_posture.eye_blinks = blink_count;
			curr_posture.eyeRect = eye;

			//draw_rects(frame, diff, window, eye);
			draw_text(mat_frame, "blink!", text_delay, 1);
		}

		imshow(wnd_name, mat_frame);
		cvShowImage(wnd_debug, diff);
		if (previous) 
			cvReleaseImage(&previous);
		previous = (IplImage*)cvClone(gray);
		//key = cvWaitKey(15);

		return 1;

}
int detectAmbientLight(Mat &mat_frame, IplImage *frame, bool configure = false)
{
	Mat frame_yuv;
    cvtColor( mat_frame, frame_yuv, CV_BGR2YUV );

    unsigned char* pixelPtr = (unsigned char*)frame_yuv.data;
    int cn = frame_yuv.channels();
	long int total_luminance = 0;
	int total_sample = 0;
    for(int i = 0; i < frame_yuv.rows; i+=10)
    {
        for(int j = 0; j < frame_yuv.cols; j += 10)
        {
            Scalar_<unsigned char> yuvPixel;
            yuvPixel.val[0] = pixelPtr[i*frame_yuv.cols*cn + j*cn + 0]; // Y
            yuvPixel.val[1] = pixelPtr[i*frame_yuv.cols*cn + j*cn + 1]; // U
            yuvPixel.val[2] = pixelPtr[i*frame_yuv.cols*cn + j*cn + 2]; // V

			total_luminance += yuvPixel.val[0];
			++total_sample;
        }
    }

	avg_luminance = total_luminance / (total_sample);
	if (configure) {
		threshold_luminance = avg_luminance;
		/*
		GUID *pPwrGUID = NULL;
		GUID subGUID = GUID_VIDEO_SUBGROUP;
		GUID BriGUID = GUID_DEVICE_POWER_POLICY_VIDEO_BRIGHTNESS;

		DWORD brigthness_out = PowerReadACValue(NULL, pPwrGUID, &subGUID, &BriGUID, NULL, NULL, NULL);
		*/
	}

	if ((threshold_luminance *0.8) > avg_luminance)
	{
		ambient_light_alm = true;
		buf_num = 10;
		//draw_text(frame, "*low light*", text_delay, 10);
	}
	else {
		ambient_light_alm = false;
	}
	curr_posture.alarm_ambient_light = ambient_light_alm;

	return 1;
}



 extern "C"  __declspec( dllexport ) int webCamMain(void)
{
	Mat mat_frame;
	init();
	while (capture.read(mat_frame))
	{
		GetSystemTimeAsFileTime(&currt);
		ticks = GetTickCount() - stime;
		memset((void*)&curr_posture, 0, sizeof(posture_t));
		curr_posture.clock_ticks = ticks;
		curr_posture.time = currt;
		if (mat_frame.empty())
		{
			//printf(" --(!) No captured frame -- Break!");
			break;
		}
		if (frame)
			delete frame;
		frame = new IplImage(mat_frame);

		bool eye_hint = detectProximityAndEye(mat_frame);

		detectBlink(mat_frame, frame, false);
		detectAmbientLight(mat_frame,frame);

		posture_vec.push_back(curr_posture);

		int c = cvWaitKey(15);
		// escape
		int ll_millisec = 0;
		if ((char)c == 27) { 
			//GetSystemTimeAsFileTime(&currt);
			//int ll_millisec = ((LONGLONG)et.dwLowDateTime + ((LONGLONG)(et.dwHighDateTime) << 32LL) - (LONGLONG)st.dwLowDateTime + ((LONGLONG)(st.dwHighDateTime) << 32LL))/10000;
			//ticks = GetTickCount();
			//ll_millisec = ticks - stime;
			//printf("%ld blinks in %ld milliseconds at rate of %f blinks per minute", blink_count, ll_millisec, (blink_count*60.0*1000.0)/(ll_millisec));
			statsdumps();
			fclose(pFile);
			break;
		} else if ((char)c == 'c') { 
			configureDefaults(mat_frame);
			detectAmbientLight(mat_frame, frame, true);
		}
		if (blink_count > 0) {
			posture_with_blink_list.push_back(curr_posture);
			blink_count = 0;
		}
		if (buf_num == 10)
			buf_posture = curr_posture;
		if (buf_num > 0)
			--buf_num;
	}
	return 0;
}

bool detectProximityAndEye(Mat mat_frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	bool eye_detected = false;

	cvtColor(mat_frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0, Size(30, 30));
	int is_near = getProximity(mat_frame); 


	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//ellipse(mat_frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		if (is_near) 
		{

			eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(6, 6));
		}
		else
		{
			eyes_cascade.detectMultiScale(faceROI, eyes, 1.03, 2, 0 | CASCADE_SCALE_IMAGE, Size(1, 1));
		}

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			eye_detected = true;
			eye2 = cvRect(
				eye_center.x - (TPL_WIDTH/2),
				eye_center.y - (TPL_WIDTH/2),
				TPL_WIDTH,
				TPL_HEIGHT
				//2*radius,
				//2*radius
			);
			//circle(mat_frame, eye_center, radius, Scalar(255, 0, 0), 1, 8, 0);
		}
		break;
	}
	/*
	*/
	//-- Show what you got
	//imshow(window_name, mat_frame);
	if (frame)
		delete frame;
	frame = new IplImage(mat_frame);
	cvShowImage(wnd_name, frame);
	return eye_detected;
}
