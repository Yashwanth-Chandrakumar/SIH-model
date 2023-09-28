# yolov7-pose-estimation and action detection

### Steps to run Code
- If you are using google colab then you will first need to mount the drive with mentioned command first, <b>(Windows or Linux users)</b> both can skip this step.
```
from google.colab import drive
drive.mount("/content/drive")
```
- Clone the repository.
```
!git clone https://github.com/Yashwanth-Chandrakumar/SIH-model.git
```

- Goto the cloned folder.
```
cd SIH-model
```

- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For Linux Users
python3 -m venv psestenv
source psestenv/bin/activate

### For Window Users
python3 -m venv psestenv
cd psestenv
cd Scripts
activate
cd ..
cd ..
```

- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```

- Install requirements with mentioned command below.

```
pip install -r requirements.txt
```

- Download yolov7 pose estimation weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {yolov7-pose-estimation}
- or !curl -L https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt -o yolov7-w6-pose.pt for # Colab
- or  curl -L https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt -o yolov7-w6-pose.pt for # PC

- Run the code with mentioned command below.
```
python pose-estimate.py

#if you want to change source file
python pose-estimate.py --source "your custom video.mp4"

#For CPU
python pose-estimate.py --source "your custom video.mp4" --device cpu

#For GPU
python pose-estimate.py --source "your custom video.mp4" --device 0

#For View-Image
python pose-estimate.py --source "your custom video.mp4" --device 0 --view-img

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python pose-estimate.py --source "your IP Camera Stream URL" --device 0 --view-img

#For WebCam
python pose-estimate.py --source 0 --view-img

#For External Camera
python pose-estimate.py --source 1 --view-img
```

- Output file will be created in the working directory with name <b>["your-file-name-without-extension"+"_keypoint.mp4"]</b>

#### RESULTS

<table>
  <tr>
    
  </tr>
  <tr>
    
  </tr>
 </table>

#### References
- https://github.com/RizwanMunawar/yolov7-pose-estimation.git (Special Thanks)
- https://github.com/WongKinYiu/yolov7
- https://github.com/augmentedstartups/yolov7
- https://github.com/augmentedstartups
- https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/
- https://github.com/ultralytics/yolov5

#### My Medium Articles
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f
