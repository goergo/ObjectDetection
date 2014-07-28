#include <stdio.h>

#if 0

CvCapture*		capture;
IplImage*		frame, *gray, *prev, *diff, *tpl;
CvMemStorage*	storage;
IplConvKernel*	kernel;
CvFont			font;
char*			wnd_name = "video";
char*			wnd_debug = "diff";

int  get_connected_components(IplImage* img, IplImage* prev, CvRect window, CvSeq** comp);
int	 is_eye_pair(CvSeq* comp, int num, CvRect* eye);
int  locate_eye(IplImage* img, IplImage* tpl, CvRect* window, CvRect* eye);
int	 is_blink(CvSeq* comp, int num, CvRect window, CvRect eye);
void delay_frames(int nframes);
void init();
void exit_nicely(char* msg);

/*
int _tmain(int argc, _TCHAR* argv[])
{
	CvSeq*	comp = 0;
	CvRect	window, eye;
	int		key = 0, nc, found;
	int		text_delay, stage = STAGE_INIT;

	init();

	while (key != 'q')
	{
		frame = cvQueryFrame(capture);
		if (!frame)
			exit_nicely("cannot query frame!");
		frame->origin = 0;

		if (stage == STAGE_INIT)
			window = cvRect(0, 0, frame->width, frame->height);

		cvCvtColor(frame, gray, CV_BGR2GRAY);

		nc = get_connected_components(gray, prev, window, &comp);

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

			if (is_blink(comp, nc, window, eye))
				text_delay = 1;

			DRAW_RECTS(frame, diff, window, eye);
			DRAW_TEXT(frame, "blink!", text_delay, 1);
		}

		cvShowImage(wnd_name, frame);
		cvShowImage(wnd_debug, diff);
		prev = (IplImage*)cvClone(gray);
		key = cvWaitKey(15);
	}

	exit_nicely(NULL);
	return 0;
}
*/




/**
* This is the wrapper function for cvFindContours
*
* @param	IplImage* img	  the current grayscaled frame
* @param	IplImage* prev	  previously saved frame
* @param	CvRect    window  search within this window
* @param	CvSeq**   comp    output parameter, will contain the connected components
* @return	int				  the number of connected components
*/
int
get_connected_components(IplImage* img, IplImage* prev, CvRect window, CvSeq** comp)
{
	IplImage* _diff;

	cvZero(diff);

	/* apply search window to images */
	cvSetImageROI(img, window);
	cvSetImageROI(prev, window);
	cvSetImageROI(diff, window);

	/* motion analysis */
	cvSub(img, prev, diff, NULL);
	cvThreshold(diff, diff, 5, 255, CV_THRESH_BINARY);
	cvMorphologyEx(diff, diff, NULL, kernel, CV_MOP_OPEN, 1);

	/* reset search window */
	cvResetImageROI(img);
	cvResetImageROI(prev);
	cvResetImageROI(diff);

	_diff = (IplImage*)cvClone(diff);

	/* get connected components */
	int nc = cvFindContours(_diff, storage, comp, sizeof(CvContour),
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	cvClearMemStorage(storage);
	cvReleaseImage(&_diff);

	return nc;
}

/**
* Experimentally-derived heuristics to determine whether
* the connected components are eye pair or not.
*
* @param	CvSeq*  comp the connected components
* @param	int     num  the number of connected components
* @param   CvRect* eye  output parameter, will contain the location of the
*                       first component
* @return	int          '1' if eye pair, '0' otherwise
*/
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
* Delay for the specified frame count. I have to write this custom
* delay function for these reasons:
* - usleep() is not available in Windows
* - usleep() and Sleep() will freeze the video for the given interval
*/
void
delay_frames(int nframes)
{
	int i;

	for (i = 0; i < nframes; i++)
	{
		frame = cvQueryFrame(capture);
		if (!frame)
			exit_nicely("cannot query frame");
		cvShowImage(wnd_name, frame);
		if (diff)
			cvShowImage(wnd_debug, diff);
		cvWaitKey(30);
	}
}

/**
* Initialize images, memory, and windows
*/
void
init()
{
	int delay, i;

	capture = cvCaptureFromCAM(0);
	if (!capture)
		exit_nicely("Cannot initialize camera!");

	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

	frame = cvQueryFrame(capture);
	if (!frame)
		exit_nicely("cannot query frame!");

	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.4, 0.4, 0, 1, 8);
	cvNamedWindow(wnd_name, 1);

	for (delay = 20, i = 0; i < 1; i++, delay = 20)
	while (delay)
	{
		frame = cvQueryFrame(capture);
		if (!frame)
			exit_nicely("cannot query frame!");
		//DRAW_TEXT(frame, msg[i], delay, 0);
		cvShowImage(wnd_name, frame);
		cvWaitKey(30);
	}

	storage = cvCreateMemStorage(0);
	if (!storage)
		exit_nicely("cannot allocate memory storage!");

	kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);
	gray = cvCreateImage(cvGetSize(frame), 8, 1);
	prev = cvCreateImage(cvGetSize(frame), 8, 1);
	diff = cvCreateImage(cvGetSize(frame), 8, 1);
	tpl = cvCreateImage(cvSize(TPL_WIDTH, TPL_HEIGHT), 8, 1);

	if (!kernel || !gray || !prev || !diff || !tpl)
		exit_nicely("system error.");

	gray->origin = frame->origin;
	prev->origin = frame->origin;
	diff->origin = frame->origin;

	cvNamedWindow(wnd_debug, 1);
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

	if (capture)
		cvReleaseCapture(&capture);
	if (gray)
		cvReleaseImage(&gray);
	if (prev)
		cvReleaseImage(&prev);
	if (diff)
		cvReleaseImage(&diff);
	if (tpl)
		cvReleaseImage(&tpl);
	if (storage)
		cvReleaseMemStorage(&storage);

	if (msg != NULL)
	{
		fprintf(stderr, msg);
		fprintf(stderr, "\n");
		exit(1);
	}

	exit(0);
}

#endif