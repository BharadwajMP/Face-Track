#include <iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/tracking.hpp>
#include "face_detector.hpp"
#include "helpers.hpp"
#include<vector>
#include<fstream>
#define FRAME_LIFE 2
#define FRAME_RATE 30
using namespace std;
using namespace cv;

//Function to decrement tracker life
void desc(std::pair<Ptr<Tracker>,std::pair<Rect2d,int>> &it){
	it.second.second--;
}

//Checks is tracker is dead
bool isZero(std::pair<Ptr<Tracker>,std::pair<Rect2d,int>> it){
	return it.second.second <= 0;
}

//Computes centroid of 
Point centroid(Rect2d bbox){
	return Point((2*bbox.x+bbox.width)/2,(2*bbox.y+bbox.height)/2);
}


int main(int argc,char* argv[]) {
	Timer timer;
	int id = 0;
	int fc = 0;
	mtcnn::FaceDetector fd("./model/", 0.75f, 0.7f, 0.7f, true, true, 0);
	//Offline video
	VideoCapture video("sample1.mp4");
	//Use web cam
	// VideoCapture video(0);

	if(!video.isOpened()){
        cout<<"Could not read video file"<<endl;
        exit(-1);
    }

	//Face trackers
	vector<std::pair<Ptr<Tracker>,std::pair<Rect2d,int>>> trackers;
	
	//Frame
	cv::Mat frame;

	//Log file
	// ofstream log_file;
	// log_file.open("log.txt");

	while(video.read(frame)){
		//Process every FRAME_RATEth frame
		fc = (fc + 1) % FRAME_RATE;
		if(fc){
			continue;
		}
		id = 0;
		//Decrement life of all trackers by one
		for_each(trackers.begin(),trackers.end(),desc);
		
		timer.start();

		//Make copy of read frame
		cv::Mat mod_img = frame;

		//Call MTCNN to detect faces
		std::vector<mtcnn::Face> faces = fd.detect(mod_img, 40.f, 0.7f);
	
		std::cout << "Elapsed time (ms): " << timer.stop() << std::endl;
		for (int i = 0; i < faces.size(); ++i) {
			std::vector<cv::Point> pts;
			for (int p = 0; p < mtcnn::NUM_PTS; ++p) {
				pts.push_back(cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
			}
			Rect2d bbox = faces[i].bbox.getRect();
			if(bbox.x < 0 || bbox.y < 0 || bbox.y + bbox.height > frame.rows || bbox.x + bbox.width > frame.cols)
				continue;
			
			mod_img = drawAndShowFace(mod_img, bbox, pts);

			//flag is set if a tracker is present for the detected face
			int flag = 0;

			vector<std::pair<Ptr<Tracker>,std::pair<Rect2d,int>>>::iterator it;
			for(it = trackers.begin();it < trackers.end();it++){
				bool ok = it->first->update(frame,it->second.first); 
				if(ok){
					cout<<"Tracker updated successfully"<<endl;
        		}
        		else{
					//Delete tracker if updation fails
					it = trackers.erase(it);
				}

				//Compute centroid of tracker
				Point cen = centroid(it->second.first);

				//Check if centroid if inside bbox of detected face. The last condition checks if tracker is not reused
				if((cen.x <= (bbox.x + bbox.width) && cen.x >= bbox.x && cen.y <= (bbox.y + bbox.height) && cen.y >= bbox.y) && it->second.second < 2){
					it->second.second++;
					//Update tracker roi
					it->first->clear();
					Ptr<Tracker> tracker = TrackerCSRT::create();
					tracker->init(frame,bbox);
					it->first = tracker;
					flag = 1;
					break;
				}
			}
			//Adds a new tracker if no suitable tracker was found
			if(!flag /*&& fd.detect(frame(faces[i].bbox.getRect()), 10.f, 0.98f).size()*/){
				cout<<"Tracker added"<<endl;
				Ptr<Tracker> tracker = TrackerCSRT::create();
				tracker->init(frame,faces[i].bbox.getRect());
				rectangle(mod_img,faces[i].bbox.getRect(),Scalar(0,0,0),4);
				trackers.push_back(make_pair(tracker,make_pair(faces[i].bbox.getRect(),FRAME_LIFE)));
				//Save cropped image on disc
				cv::imwrite("faces/"+std::to_string(video.get(CAP_PROP_FRAME_COUNT))+"_"+std::to_string(id)+".jpeg",frame(bbox));
				id++;
			}
		}
		//Delete trackers whose life is 0
		trackers.erase(remove_if(trackers.begin(),trackers.end(),isZero),trackers.end());

		for(vector<std::pair<Ptr<Tracker>,std::pair<Rect2d,int>>>::iterator it = trackers.begin();it < trackers.end();it++){
			rectangle(mod_img,it->second.first,Scalar(0,255,0));
		}
		// log_file<<"DF: "+to_string(faces.size())+" T: "+to_string(trackers.size())<<endl;
		cv::imshow("Test",mod_img);

		int k = waitKey(1);
        if(k == 27){
            break;
        }
	}
	// log_file.close();
	return 0;
}
