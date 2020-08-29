#include "main.h"

//메인함수//
int main(int argc, char* argv[])
{
	//glut 초기화
	glutInitDisplayMode ( GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA ); // Display Mode
	glutInitWindowSize(_width*2,_height*2);
	glutInitWindowPosition(0,0);
	glutInit(&argc,argv);
	glLoadIdentity();
	glutCreateWindow("openGL");
	
	initGL();// texture 정보 읽어 오기
	
	////Camera part
	
	cvNamedWindow("Cam_original",0);
	cvResizeWindow("Cam_original",_width,_height);

	cvNamedWindow("imgBinary",0);
	cvResizeWindow("imgBinary",_width,_height);
	
	cvNamedWindow("final",0);
	cvResizeWindow("final",_width*2,_height*2);
	
	cvSet(final,cvScalar(255,255,255));


	// 초기 20프레임에 대한 영상의 평균값을 구한다.
	for(int i=0; i<INIT_TIME; i++)
		{
			imgCam = cvQueryFrame(capture);
			cvAcc(imgCam,imgAv);
		}
		cvConvertScale(imgAv,imgAv,1.0/INIT_TIME);

   // 절대값에 대한 영상의 평균값 구한다.
	for(int i=0; i<INIT_TIME; i++)
		{
			imgCam = cvQueryFrame(capture);
			cvConvert(imgCam, imgTmp);
			cvSub(imgTmp,imgAv,imgTmp);
			cvPow(imgTmp,imgTmp,2.0); // 제곱
			cvPow(imgTmp,imgTmp,0.5); // 루트
			cvAcc(imgTmp,imgSgm);
		}
	cvConvertScale(imgSgm,imgSgm,1.0/INIT_TIME);


	while(cvWaitKey(30)!=27) 
	{
		imgCam = cvQueryFrame(capture);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cvCvtColor(imgInput, imgBackgnd, CV_RGB2GRAY);
		cvCvtColor(imgCam, imgGray, CV_RGB2GRAY);
		
		cvThreshold(imgGray,imgBinary,Threshold,255,CV_THRESH_BINARY_INV); // Threshold 설정
		cvErode(imgBinary,imgBinary); // 침식
////////////////////////////////////////////////레이블링//////////////////////////////////////////////////////////

		/*if(GetAsyncKeyState(VK_UP))
		{
			if(Threshold < 255) 
				{
					Threshold ++; 
					printf("Threshold = %d \n",Threshold);
				}
		}

		else if(GetAsyncKeyState(VK_DOWN))
		{
				if(Threshold > 0) 
				{
					Threshold --; 
					printf("Threshold = %d \n",Threshold);
				}
		}
		else if(GetAsyncKeyState(VK_RIGHT))
		{
				if(rectsize < 470) 
				{
					rectsize += 10; 
					printf("rectsize = %d \n",rectsize);
				}
		}
		else if(GetAsyncKeyState(VK_LEFT))
		{
				if(rectsize > 10) 
				{
					rectsize -= 10; 
					printf("rectsize = %d \n",rectsize);
				}
		}*/

		blob.SetParam(imgBinary,100);
		blob.DoLabeling();

		//////레이블된 영역 설정
		for(int i=0; i<blob.m_nBlobs; i++)  // m_nBlobs = 레이블링의 갯수 , m_recBlobs = 각레이블의 사각형 왼쪽 위 좌표
		{
			contours = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour),sizeof(CvPoint),storage);

			//////레이블링 잡영 제거 지정범위안에 레이블만 사용한다.
			if(blob.m_recBlobs[i].width > 60 && blob.m_recBlobs[i].height > 50)
			{
				
				pt1 = cvPoint(blob.m_recBlobs[i].x,   blob.m_recBlobs[i].y);
				if(blob.m_recBlobs[i].y < (480-rectsize)) blob.m_recBlobs[i].height = rectsize;
				if(blob.m_recBlobs[i].x < (480-rectsize)) blob.m_recBlobs[i].width = rectsize;
				pt2 = cvPoint(pt1.x + blob.m_recBlobs[i].width,  pt1.y + blob.m_recBlobs[i].height);

				cvDrawRect(imgCam,pt1,pt2,CV_RGB(255,255,255),2); // 손범위 사각형 그리기

				hand = cvRect(pt1.x, pt1.y, blob.m_recBlobs[i].width,  blob.m_recBlobs[i].height); // 최종적으로 계산 된 손 범위를 hand 사각형 변수에 지정한다.

				cvSetImageROI(imgBinary,hand); // Region of Interest 즉 hand의 범위만큼 지역이미지로 설정
				subHand = cvCreateImage (cvSize(blob.m_recBlobs[i].width, blob.m_recBlobs[i].height), IPL_DEPTH_8U, 1); 
				cvCopy(imgBinary,subHand);// 지역이미지를 sub_hand 이미지 변수로 카피
				cvResetImageROI(imgBinary);

				///////////////////////////Subhand image에서 Moment 값을 구한다////////////////////////////////////////////////
				cvMoments(subHand, &moments);
				m00 = cvGetSpatialMoment(&moments, 0, 0);
				centerX = cvGetSpatialMoment(&moments, 1, 0)/m00; // 무게중심 x좌표
				centerY = cvGetSpatialMoment(&moments, 0, 1)/m00; // 무게중심 y좌표				
				cvCircle(imgCam, cvPoint(pt1.x+cvRound(centerX),pt1.y+cvRound(centerY)),5,cvScalarAll(255),CV_FILLED); // Hu moment를 이용한 무게 중심 점 원래 영상에 그리기
				////////////////////////////////////////////////////////////////////////////////////////////////////////////////

				for(int y=0; y< subHand->height; y++)
				{
					for(int x=0; x< subHand->width; x++)
					{
						pixel = cvGet2D(subHand,y,x); // x,y 좌표에 따른 pixel 색상 검출
						if(pixel.val[0] == 255)       // pixel의 Blue에 해당하는 변수가 255일 경우 [흰색=(255,255,255)] 
						{
							pt0.x = x;			      // ptseq에 엣지검출된 Subhand 이미지에서 흰색점에 해당되는 좌표들을 전부 저장한다.
							pt0.y = y;
							cvSeqPush(contours, &pt0);
						}
					}
				}

				/////////////convex hull 구하기//////////////////
				if(contours->total> 0)
				{	
					pt0.x = 0;
					pt0.y = 0;
					hull = cvConvexHull2(contours,0,CV_COUNTER_CLOCKWISE,0);

					for(int j=0; j<hull->total; j++)
					{
						hull_pt = **CV_GET_SEQ_ELEM(CvPoint*, hull, j);
				
						cvCircle(imgCam,cvPoint(pt1.x+hull_pt.x,pt1.y+hull_pt.y),2,CV_RGB(0,255,0),2); // hull 표시
						//cvShowImage("ROI",subHand); // ROI

						if(pt0.x == 0 && pt0.y ==0) // 초기값 설정
						{
							pt0 = hull_pt; end_pt = pt0;
						}
					
						cvLine(imgCam,cvPoint(pt0.x+pt1.x,pt0.y+pt1.y),cvPoint(hull_pt.x+pt1.x,hull_pt.y+pt1.y),cvScalarAll(255)); // convex hull 이어논 선 표시
						pt0 = hull_pt;

						if(j==hull->total-1)
							cvLine(imgCam,cvPoint(hull_pt.x+pt1.x,hull_pt.y+pt1.y),cvPoint(end_pt.x+pt1.x,end_pt.y+pt1.y),cvScalarAll(255)); // 끝점과 시작점 잇기

						/////////////// hull에서 중심까지 가장 긴 거리 찾기
						tmpDist_pt = **CV_GET_SEQ_ELEM(CvPoint*, hull, j);
						tmpDist = sqrt((double)((centerX-tmpDist_pt.x) * (centerX-tmpDist_pt.x) + (centerY - tmpDist_pt.y) *(centerY - tmpDist_pt.y)));
						if(tmpDist > MaxDist && tmpDist_pt.y < centerY)
						{
							MaxDist_pt = tmpDist_pt;
							MaxDist = tmpDist;
						}
					}
		
					cvLine(imgCam,cvPoint(pt1.x+cvRound(centerX),pt1.y+cvRound(centerY)),cvPoint(pt1.x+MaxDist_pt.x,pt1.y+MaxDist_pt.y),cvScalar(0,0,0),1); // 중심점에서 손가락 중심점으로 잇는 선  원래 영상에 그리기
				}

////////////////////////////////////////// circle과 비교한뒤 re-labeling 하는 작업 //////////////////////////////////////////////
				imgCircle = cvCreateImage (cvGetSize(subHand), IPL_DEPTH_8U, 1);
				cvSetZero(imgCircle);  // 비교원 기본값을 전부 0 (검정색)으로 한다.
				rad = MaxDist*7/12;// 비교원을 그릴 반지름
				cvCircle(imgCircle,cvPoint(centerX,centerY),rad,CV_RGB(255,255,255),2); // 비교원 그리기
			
				cvAnd(subHand,imgCircle,subHand,0); // And연산을 한다.
				cvDilate(subHand,subHand); // 증식연산
				//cvShowImage("ROI",subHand);
				//cvCircle(imgCam,cvPoint(pt1.x+centerX,pt1.y+centerY),rad,CV_RGB(255,255,255),2); // 비교원 원래 영상에 그리기
			

				blob.SetParam(subHand,30);   // And연산을 한 결과값을 레이블링 하여 흰색부분의 갯수를 검출한면 손목을 포함한 손가락의 갯수를 알 수 있다. threshold값으로 작은것은 쳐낼수 있다.
				blob.DoLabeling();

				if(blob.m_nBlobs >= 5) // 손바닥
				{
					cvSet(final,cvScalarAll(255));
					
					featureRight[0].x = pt1.x+cvRound(centerX);
					featureRight[0].y = pt1.y+cvRound(centerY);

					ptSelected[0].x = pt1.x+cvRound(centerX);
					ptSelected[0].y = pt1.y+cvRound(centerY);

					
					startflag = true;

					pt_line = cvPoint((pt1.x+MaxDist_pt.x)*2,(pt1.y+MaxDist_pt.y)*2);

					cvDrawRect(final,cvPoint(0,0),cvPoint(150,150),CV_RGB(255,0,0),2);
					cvDrawRect(final,cvPoint(1130,0),cvPoint(1280,150),CV_RGB(0,0,255),2);
					cvDrawRect(final,cvPoint(565,0),cvPoint(715,150),CV_RGB(0,255,0),2);

					lineColor = cvScalarAll(0);

					cvSet(drawCol,cvScalarAll(255));
				}
				
				else if(blob.m_nBlobs == 2) // 손가락 n+1 일때
				{
					
					
					featureRight[0].x = pt1.x+MaxDist_pt.x;
					featureRight[0].y = pt1.y+MaxDist_pt.y;

					featureRight[1].x = pt1.x+MaxDist_pt.x+200;
					featureRight[1].y = pt1.y+MaxDist_pt.y;

					ptSelected[0].x = pt1.x+MaxDist_pt.x;
					ptSelected[0].y = pt1.y+MaxDist_pt.y;

					ptSelected[1].x = pt1.x+MaxDist_pt.x+200;
					ptSelected[1].y = pt1.y+MaxDist_pt.y;

					if(startflag == true)
					{
						if( colorflag == 0 && pt_line.y < 150)
						{
							if(pt_line.x < 150) lineColor = cvScalar(0,0,255);
							if(pt_line.x > 1130) lineColor = cvScalar(255,0,0);
							if(pt_line.x > 565 && pt_line.x < 715) lineColor = cvScalar(0,255,0);

							colorflag=1;
						}
						if(pt_line.x == 0 && pt_line.y ==0)
						{
							pt_line = cvPoint((pt1.x+MaxDist_pt.x)*2,(pt1.y+MaxDist_pt.y)*2);
						}
						cvLine(drawMsk,pt_line,cvPoint((pt1.x+MaxDist_pt.x)*2,(pt1.y+MaxDist_pt.y)*2),cvScalarAll(255),2);
						cvLine(drawCol,pt_line,cvPoint((pt1.x+MaxDist_pt.x)*2,(pt1.y+MaxDist_pt.y)*2),lineColor,2);

						pt_line = cvPoint((pt1.x+MaxDist_pt.x)*2,(pt1.y+MaxDist_pt.y)*2);
					}
				}
				else if(blob.m_nBlobs == 3) // 손가락 n+1 일때
				{
					colorflag = 0;
					pt_line = cvPoint((pt1.x+MaxDist_pt.x)*2,(pt1.y+MaxDist_pt.y)*2);
				}
				
				cvCopy(drawCol,final, drawMsk);// Mask에 그려진 선 원래 영상에 복사하기

			}
			cvReleaseImage(&imgCircle);
			cvReleaseImage(&subHand);
			contours->total = 0;
			MaxDist = 0.0;
			MaxDist_pt.x = 0;
			MaxDist_pt.y = 0;
		}

		cvCopy(drawCol,final, drawMsk);// Mask에 그려진 선 원래 영상에 복사하기

		cvSetZero(drawMsk);
		cvSetZero(drawCol);
		
		cvCopy(imgGray,imgPre); // 현재 프레임을 이전프레임으로 설정

		cvShowImage("Cam_original",imgCam);
		cvShowImage("imgBinary",imgBinary);
		cvShowImage("final",final);		

		cvClearMemStorage(storage);
	}
	cvReleaseCapture(&capture);
	cvReleaseMemStorage(&storage);


	cvDestroyAllWindows();

	return 0;
}

//////////////////////////opengl 초기화 및 텍스쳐 설정 /////////////////////////////////////////
void initGL(void)
{
	 glShadeModel(GL_SMOOTH);       // Enable Smooth Shading
	 glClearColor(0.0f, 0.0f, 0.0f, 0.5f);    // Black Background
	 glEnable ( GL_COLOR_MATERIAL );
	 glColorMaterial ( GL_FRONT, GL_AMBIENT_AND_DIFFUSE );

	 glEnable ( GL_TEXTURE_2D );
	 glEnable(GL_BLEND);
	 glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); 
	 glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );
	 glGenTextures (1, texture);

	 AUX_RGBImageRec *TextureImage[1];     // Create Storage Space For The Texture

	 memset(TextureImage,0,sizeof(void *)*1);            // Set The Pointer To NULL

	 if (TextureImage[0]=LoadBMP("hrt.bmp"))//이미지 로딩
	 {
	  glGenTextures(1, &texture[0]);     //텍스쳐 생성

	  //텍스쳐에 이미지 넣기
	  glBindTexture(GL_TEXTURE_2D, texture[0]);
	  glTexImage2D(GL_TEXTURE_2D, 0, 3, TextureImage[0]->sizeX, TextureImage[0]->sizeY, 0, GL_RGB, GL_UNSIGNED_BYTE, TextureImage[0]->data);
	  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP);
 }

 if (TextureImage[0])         // If Texture Exists
 {
  if (TextureImage[0]->data)       // If Texture Image Exists
  {
   free(TextureImage[0]->data);     // Free The Texture Image Memory
  }

  free(TextureImage[0]);        // Free The Image Structure
 }

 glEnable ( GL_CULL_FACE ); 
}
//////////////// 반올림 함수//////////////////
double round(double number, int nPos)
{
	double num1,num2;

	num1 = 10.0;
	number *= pow(num1,(double)nPos);
	num2 = pow(num1,(double)nPos);
	number = (number>0) ? floor(number+0.5) : ceil(number-0.5);
	number = number/num2;

	return number;
}

////////////////////////////////////////////////////moment값 가져오기 ///////////////////////////////////////////////////////
//for(y_order=0; y_order<= 3; y_order++)
				//{
				//	for(x_order=0; x_order<=3; x_order++)
				//	{
				//		if(x_order + y_order >3)
				//				continue;
				//		M = cvGetSpatialMoment(&moments, x_order, y_order);  //CvMoments 구조체에서 특정 차수 모멘트를 반환한다. (단, 3차이하)

				//		if(x_order == 0 && y_order == 0)
				//			m00 = M;
				//		else if(x_order == 1 && y_order == 0)
				//			centerX = M;
				//		else if(x_order == 0 && y_order == 1)
				//			centerY = M;
				//	}
				//}
				//centerX /= m00;   //중심 x좌표
				//centerY /= m00;	 //중심 y좌표


/////////////////////////////////// circle과 비교한뒤 re labeling 하는 작업 //////////////////////////////////////////////
				//imgCircle = cvCreateImage (cvGetSize(subHand), IPL_DEPTH_8U, 1);
				//cvSetZero(imgCircle);  // 비교원 기본값을 전부 0 (검정색)으로 한다.
				//rad = MaxDist*2/3;// 비교원을 그릴 반지름
				//cvCircle(imgCircle,cvPoint(centerX,centerY),rad,CV_RGB(255,255,255),2); // 비교원 그리기
			
				//cvAnd(subHand,imgCircle,subHand,0); // And연산을 한다.
				//cvDilate(subHand,subHand); // 증식연산
				////cvShowImage("ROI",subHand);
				////cvCircle(imgCam,cvPoint(pt1.x+centerX,pt1.y+centerY),rad,CV_RGB(255,255,255),2); // 비교원 원래 영상에 그리기
			

				//blob.SetParam(subHand,10);   // And연산을 한 결과값을 레이블링 하여 흰색부분의 갯수를 검출한면 손목을 포함한 손가락의 갯수를 알 수 있다. threshold값으로 작은것은 쳐낼수 있다.
				//blob.DoLabeling();

				//if(blob.m_nBlobs >= 5) // 손바닥
				//{
				//	cvDrawRect(imgCam, pt1, pt2, CV_RGB(255,255,255),2); // 손범위 사각형 그리기
				//	//mode=1;
				//	pt_line.x = 0;
				//	pt_line.y = 0;
				//}
				//else if(blob.m_nBlobs == 2) // 손가락 n+1 일때
				//{
				//	if(pt_line.x == 0 && pt_line.y ==0)
				//	{
				//		pt_line = cvPoint((pt1.x+MaxDist_pt.x),(pt1.y+MaxDist_pt.y));
				//	}
				//	cvLine(drawMsk,pt_line,cvPoint((pt1.x+MaxDist_pt.x),(pt1.y+MaxDist_pt.y)),cvScalarAll(255),2);
				//	cvLine(drawCol,pt_line,cvPoint((pt1.x+MaxDist_pt.x),(pt1.y+MaxDist_pt.y)),lineColor,2);

				//	pt_line = cvPoint((pt1.x+MaxDist_pt.x),(pt1.y+MaxDist_pt.y));
				//}
				//else if(blob.m_nBlobs == 3) // 손가락 n+1 일때
				//{
				//	/*if(colorFlag == false)
				//	{
				//		lineColor = CV_RGB(255,0,0);
				//		colorFlag = true;
				//	}
				//	else 
				//	{
				//		lineColor = cvGet2D(imgCam,pt1.y+MaxDist_pt.y,pt1.x+MaxDist_pt.x);
				//		colorFlag = false;
				//	}*/
				//	//cvDrawRect(imgCam, pt1, pt2, CV_RGB(0,255,0),2); // 손범위 사각형 그리기
				//}

				//cvCopy(drawCol,imgCam, drawMsk);// Mask에 그려진 선 원래 영상에 복사하기
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//void PlayMusic(int tone)
//	{
//		switch(tone)
//		{
//		case 1: {
//					if(mode != 1)
//					{
//						mciSendString(TEXT("open piano/gun.mp3 type mpegvideo alias MediaFile"),NULL,0,NULL); // file load
//						mciSendString(TEXT("play MediaFile"),NULL,0,NULL);  // Play .mp3 background sound
//					}
//					if(mode == 1) 
//					{
//						moveStart = true; 
//						mciSendString(TEXT("open piano/background.mp3 type mpegvideo alias MediaFile"),NULL,0,NULL); // file load
//						mciSendString(TEXT("play MediaFile"),NULL,0,NULL);  // Play .mp3 background sound
//					}
//						counter++;
//						if(counter > 10)
//						{
//							mciSendString(TEXT("close MediaFile"), NULL, 0, NULL);
//							counter = 0;
//						}
//				}break;
//		case 2: {
//					mciSendString(TEXT("open piano/dog.mp3 type mpegvideo alias MediaFile"),NULL,0,NULL); // file load
//					mciSendString(TEXT("play MediaFile"),NULL,0,NULL);  // Play .mp3 background sound
//					
//						counter++;
//						if(counter > 10)
//						{
//							mciSendString(TEXT("close MediaFile"), NULL, 0, NULL);
//							
//							counter = 0;
//						}
//				}break;
//		case 3: {
//					mciSendString(TEXT("open piano/cat.mp3 type mpegvideo alias MediaFile"),NULL,0,NULL); // file load
//					mciSendString(TEXT("play MediaFile"),NULL,0,NULL);  // Play .mp3 background sound
//					
//						counter++;
//						if(counter > 10)
//						{
//							mciSendString(TEXT("close MediaFile"), NULL, 0, NULL);
//							
//							counter = 0;
//						}
//				}break;
//		case 4: {
//					mciSendString(TEXT("open piano/background.mp3 type mpegvideo alias MediaFile"),NULL,0,NULL); // file load
//					mciSendString(TEXT("play MediaFile"),NULL,0,NULL);  // Play .mp3 background sound
//					
//						counter++;
//						if(counter == 30)
//						{
//							mciSendString(TEXT("close MediaFile"), NULL, 0, NULL);
//							
//							counter = 0;
//						}
//				}break;
//		}
//
//	}

//sndPlaySound(TEXT("c:\\piano\\do.wav"),SND_ASYNC); // play .wav