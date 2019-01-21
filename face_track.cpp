#include<opencv2/opencv.hpp>
#include<opencv2/tracking.hpp>
#include<unistd.h>
using namespace std;
using namespace cv;

int main(int argc,char **argv){
    Ptr<Tracker> tracker = TrackerCSRT::create();

    VideoCapture video("test.mp4");

    if(!video.isOpened()){
        cout<<"Could not read video file"<<endl;
        exit(-1);
    }

    Mat frame;
    bool succ = video.read(frame);
    Rect2d bbox = selectROI(frame);
    tracker->init(frame,bbox);
    if(succ){
        
        imshow("Video",frame);

        while(video.read(frame)){
            bool ok = tracker->update(frame,bbox);

            if(ok){
                rectangle(frame,bbox,(255,0,0));
            }
            else{
                putText(frame,"Tracking failed",Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
            }
            imshow("Video",frame);
            // usleep(100000);
            int k = waitKey(1);
            if(k == 27){
                break;
            }
        }
    }
}