#include "main.h"

//�����Լ�//
int main(int argc, char* argv[])
{
	//glut �ʱ�ȭ
	glutInitDisplayMode ( GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA ); // Display Mode
	glutInitWindowSize(_width*2,_height*2);
	glutInitWindowPosition(0,0);
	glutInit(&argc,argv);
	glLoadIdentity();
	glutCreateWindow("openGL");
	
	initGL();// texture ���� �о� ����
	
	////Camera part
	
	cvNamedWindow("Cam_original",0);
	cvResizeWindow("Cam_original",_width,_height);

	cvNamedWindow("imgBinary",0);
	cvResizeWindow("imgBinary",_width,_height);
	
	cvNamedWindow("final",0);
	cvResizeWindow("final",_width*2,_height*2);
	
	cvSet(final,cvScalar(255,255,255));


	// �ʱ� 20�����ӿ� ���� ������ ��հ��� ���Ѵ�.
	for(int i=0; i<INIT_TIME; i++)
		{
			imgCam = cvQueryFrame(capture);
			cvAcc(imgCam,imgAv);
		}
		cvConvertScale(imgAv,imgAv,1.0/INIT_TIME);

   // ���밪�� ���� ������ ��հ� ���Ѵ�.
	for(int i=0; i<INIT_TIME; i++)
		{
			imgCam = cvQueryFrame(capture);
			cvConvert(imgCam, imgTmp);
			cvSub(imgTmp,imgAv,imgTmp);
			cvPow(imgTmp,imgTmp,2.0); // ����
			cvPow(imgTmp,imgTmp,0.5); // ��Ʈ
			cvAcc(imgTmp,imgSgm);
		}
	cvConvertScale(imgSgm,imgSgm,1.0/INIT_TIME);


	while(cvWaitKey(30)!=27) 
	{
		imgCam = cvQueryFrame(capture);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cvCvtColor(imgInput, imgBackgnd, CV_RGB2GRAY);
		cvCvtColor(imgCam, imgGray, CV_RGB2GRAY);
		
		cvThreshold(imgGray,imgBinary,Threshold,255,CV_THRESH_BINARY_INV); // Threshold ����
		cvErode(imgBinary,imgBinary); // ħ��
////////////////////////////////////////////////���̺�//////////////////////////////////////////////////////////

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

		//////���̺�� ���� ����
		for(int i=0; i<blob.m_nBlobs; i++)  // m_nBlobs = ���̺��� ���� , m_recBlobs = �����̺��� �簢�� ���� �� ��ǥ
		{
			contours = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour),sizeof(CvPoint),storage);

			//////���̺� �⿵ ���� ���������ȿ� ���̺� ����Ѵ�.
			if(blob.m_recBlobs[i].width > 60 && blob.m_recBlobs[i].height > 50)
			{
				
				pt1 = cvPoint(blob.m_recBlobs[i].x,   blob.m_recBlobs[i].y);
				if(blob.m_recBlobs[i].y < (480-rectsize)) blob.m_recBlobs[i].height = rectsize;
				if(blob.m_recBlobs[i].x < (480-rectsize)) blob.m_recBlobs[i].width = rectsize;
				pt2 = cvPoint(pt1.x + blob.m_recBlobs[i].width,  pt1.y + blob.m_recBlobs[i].height);

				cvDrawRect(imgCam,pt1,pt2,CV_RGB(255,255,255),2); // �չ��� �簢�� �׸���

				hand = cvRect(pt1.x, pt1.y, blob.m_recBlobs[i].width,  blob.m_recBlobs[i].height); // ���������� ��� �� �� ������ hand �簢�� ������ �����Ѵ�.

				cvSetImageROI(imgBinary,hand); // Region of Interest �� hand�� ������ŭ �����̹����� ����
				subHand = cvCreateImage (cvSize(blob.m_recBlobs[i].width, blob.m_recBlobs[i].height), IPL_DEPTH_8U, 1); 
				cvCopy(imgBinary,subHand);// �����̹����� sub_hand �̹��� ������ ī��
				cvResetImageROI(imgBinary);

				///////////////////////////Subhand image���� Moment ���� ���Ѵ�////////////////////////////////////////////////
				cvMoments(subHand, &moments);
				m00 = cvGetSpatialMoment(&moments, 0, 0);
				centerX = cvGetSpatialMoment(&moments, 1, 0)/m00; // �����߽� x��ǥ
				centerY = cvGetSpatialMoment(&moments, 0, 1)/m00; // �����߽� y��ǥ				
				cvCircle(imgCam, cvPoint(pt1.x+cvRound(centerX),pt1.y+cvRound(centerY)),5,cvScalarAll(255),CV_FILLED); // Hu moment�� �̿��� ���� �߽� �� ���� ���� �׸���
				////////////////////////////////////////////////////////////////////////////////////////////////////////////////

				for(int y=0; y< subHand->height; y++)
				{
					for(int x=0; x< subHand->width; x++)
					{
						pixel = cvGet2D(subHand,y,x); // x,y ��ǥ�� ���� pixel ���� ����
						if(pixel.val[0] == 255)       // pixel�� Blue�� �ش��ϴ� ������ 255�� ��� [���=(255,255,255)] 
						{
							pt0.x = x;			      // ptseq�� ��������� Subhand �̹������� ������� �ش�Ǵ� ��ǥ���� ���� �����Ѵ�.
							pt0.y = y;
							cvSeqPush(contours, &pt0);
						}
					}
				}

				/////////////convex hull ���ϱ�//////////////////
				if(contours->total> 0)
				{	
					pt0.x = 0;
					pt0.y = 0;
					hull = cvConvexHull2(contours,0,CV_COUNTER_CLOCKWISE,0);

					for(int j=0; j<hull->total; j++)
					{
						hull_pt = **CV_GET_SEQ_ELEM(CvPoint*, hull, j);
				
						cvCircle(imgCam,cvPoint(pt1.x+hull_pt.x,pt1.y+hull_pt.y),2,CV_RGB(0,255,0),2); // hull ǥ��
						//cvShowImage("ROI",subHand); // ROI

						if(pt0.x == 0 && pt0.y ==0) // �ʱⰪ ����
						{
							pt0 = hull_pt; end_pt = pt0;
						}
					
						cvLine(imgCam,cvPoint(pt0.x+pt1.x,pt0.y+pt1.y),cvPoint(hull_pt.x+pt1.x,hull_pt.y+pt1.y),cvScalarAll(255)); // convex hull �̾�� �� ǥ��
						pt0 = hull_pt;

						if(j==hull->total-1)
							cvLine(imgCam,cvPoint(hull_pt.x+pt1.x,hull_pt.y+pt1.y),cvPoint(end_pt.x+pt1.x,end_pt.y+pt1.y),cvScalarAll(255)); // ������ ������ �ձ�

						/////////////// hull���� �߽ɱ��� ���� �� �Ÿ� ã��
						tmpDist_pt = **CV_GET_SEQ_ELEM(CvPoint*, hull, j);
						tmpDist = sqrt((double)((centerX-tmpDist_pt.x) * (centerX-tmpDist_pt.x) + (centerY - tmpDist_pt.y) *(centerY - tmpDist_pt.y)));
						if(tmpDist > MaxDist && tmpDist_pt.y < centerY)
						{
							MaxDist_pt = tmpDist_pt;
							MaxDist = tmpDist;
						}
					}
		
					cvLine(imgCam,cvPoint(pt1.x+cvRound(centerX),pt1.y+cvRound(centerY)),cvPoint(pt1.x+MaxDist_pt.x,pt1.y+MaxDist_pt.y),cvScalar(0,0,0),1); // �߽������� �հ��� �߽������� �մ� ��  ���� ���� �׸���
				}

////////////////////////////////////////// circle�� ���ѵ� re-labeling �ϴ� �۾� //////////////////////////////////////////////
				imgCircle = cvCreateImage (cvGetSize(subHand), IPL_DEPTH_8U, 1);
				cvSetZero(imgCircle);  // �񱳿� �⺻���� ���� 0 (������)���� �Ѵ�.
				rad = MaxDist*7/12;// �񱳿��� �׸� ������
				cvCircle(imgCircle,cvPoint(centerX,centerY),rad,CV_RGB(255,255,255),2); // �񱳿� �׸���
			
				cvAnd(subHand,imgCircle,subHand,0); // And������ �Ѵ�.
				cvDilate(subHand,subHand); // ���Ŀ���
				//cvShowImage("ROI",subHand);
				//cvCircle(imgCam,cvPoint(pt1.x+centerX,pt1.y+centerY),rad,CV_RGB(255,255,255),2); // �񱳿� ���� ���� �׸���
			

				blob.SetParam(subHand,30);   // And������ �� ������� ���̺� �Ͽ� ����κ��� ������ �����Ѹ� �ո��� ������ �հ����� ������ �� �� �ִ�. threshold������ �������� �ĳ��� �ִ�.
				blob.DoLabeling();

				if(blob.m_nBlobs >= 5) // �չٴ�
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
				
				else if(blob.m_nBlobs == 2) // �հ��� n+1 �϶�
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
				else if(blob.m_nBlobs == 3) // �հ��� n+1 �϶�
				{
					colorflag = 0;
					pt_line = cvPoint((pt1.x+MaxDist_pt.x)*2,(pt1.y+MaxDist_pt.y)*2);
				}
				
				cvCopy(drawCol,final, drawMsk);// Mask�� �׷��� �� ���� ���� �����ϱ�

			}
			cvReleaseImage(&imgCircle);
			cvReleaseImage(&subHand);
			contours->total = 0;
			MaxDist = 0.0;
			MaxDist_pt.x = 0;
			MaxDist_pt.y = 0;
		}

		cvCopy(drawCol,final, drawMsk);// Mask�� �׷��� �� ���� ���� �����ϱ�

		cvSetZero(drawMsk);
		cvSetZero(drawCol);
		
		cvCopy(imgGray,imgPre); // ���� �������� �������������� ����

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

//////////////////////////opengl �ʱ�ȭ �� �ؽ��� ���� /////////////////////////////////////////
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

	 if (TextureImage[0]=LoadBMP("hrt.bmp"))//�̹��� �ε�
	 {
	  glGenTextures(1, &texture[0]);     //�ؽ��� ����

	  //�ؽ��Ŀ� �̹��� �ֱ�
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
//////////////// �ݿø� �Լ�//////////////////
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

////////////////////////////////////////////////////moment�� �������� ///////////////////////////////////////////////////////
//for(y_order=0; y_order<= 3; y_order++)
				//{
				//	for(x_order=0; x_order<=3; x_order++)
				//	{
				//		if(x_order + y_order >3)
				//				continue;
				//		M = cvGetSpatialMoment(&moments, x_order, y_order);  //CvMoments ����ü���� Ư�� ���� ���Ʈ�� ��ȯ�Ѵ�. (��, 3������)

				//		if(x_order == 0 && y_order == 0)
				//			m00 = M;
				//		else if(x_order == 1 && y_order == 0)
				//			centerX = M;
				//		else if(x_order == 0 && y_order == 1)
				//			centerY = M;
				//	}
				//}
				//centerX /= m00;   //�߽� x��ǥ
				//centerY /= m00;	 //�߽� y��ǥ


/////////////////////////////////// circle�� ���ѵ� re labeling �ϴ� �۾� //////////////////////////////////////////////
				//imgCircle = cvCreateImage (cvGetSize(subHand), IPL_DEPTH_8U, 1);
				//cvSetZero(imgCircle);  // �񱳿� �⺻���� ���� 0 (������)���� �Ѵ�.
				//rad = MaxDist*2/3;// �񱳿��� �׸� ������
				//cvCircle(imgCircle,cvPoint(centerX,centerY),rad,CV_RGB(255,255,255),2); // �񱳿� �׸���
			
				//cvAnd(subHand,imgCircle,subHand,0); // And������ �Ѵ�.
				//cvDilate(subHand,subHand); // ���Ŀ���
				////cvShowImage("ROI",subHand);
				////cvCircle(imgCam,cvPoint(pt1.x+centerX,pt1.y+centerY),rad,CV_RGB(255,255,255),2); // �񱳿� ���� ���� �׸���
			

				//blob.SetParam(subHand,10);   // And������ �� ������� ���̺� �Ͽ� ����κ��� ������ �����Ѹ� �ո��� ������ �հ����� ������ �� �� �ִ�. threshold������ �������� �ĳ��� �ִ�.
				//blob.DoLabeling();

				//if(blob.m_nBlobs >= 5) // �չٴ�
				//{
				//	cvDrawRect(imgCam, pt1, pt2, CV_RGB(255,255,255),2); // �չ��� �簢�� �׸���
				//	//mode=1;
				//	pt_line.x = 0;
				//	pt_line.y = 0;
				//}
				//else if(blob.m_nBlobs == 2) // �հ��� n+1 �϶�
				//{
				//	if(pt_line.x == 0 && pt_line.y ==0)
				//	{
				//		pt_line = cvPoint((pt1.x+MaxDist_pt.x),(pt1.y+MaxDist_pt.y));
				//	}
				//	cvLine(drawMsk,pt_line,cvPoint((pt1.x+MaxDist_pt.x),(pt1.y+MaxDist_pt.y)),cvScalarAll(255),2);
				//	cvLine(drawCol,pt_line,cvPoint((pt1.x+MaxDist_pt.x),(pt1.y+MaxDist_pt.y)),lineColor,2);

				//	pt_line = cvPoint((pt1.x+MaxDist_pt.x),(pt1.y+MaxDist_pt.y));
				//}
				//else if(blob.m_nBlobs == 3) // �հ��� n+1 �϶�
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
				//	//cvDrawRect(imgCam, pt1, pt2, CV_RGB(0,255,0),2); // �չ��� �簢�� �׸���
				//}

				//cvCopy(drawCol,imgCam, drawMsk);// Mask�� �׷��� �� ���� ���� �����ϱ�
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