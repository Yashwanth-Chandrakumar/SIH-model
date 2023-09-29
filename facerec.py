import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts, colors, plot_one_box_kpt
from scipy.spatial import distance
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import tensorflow as tf
import io

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source="football1.mp4", device='cpu', view_img=False,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):
    face_2 = face_recognition.load_image_file("/content/1295.jpg")
    face_2_encodings = face_recognition.face_encodings(face_2)
    if face_2_encodings:
        face_2_encoding = face_2_encodings[0]
    else:
        print("No face encodings found in the second image.")

    face_4 = face_recognition.load_image_file("/content/400.jpg")
    face_4_encodings = face_recognition.face_encodings(face_4)
    if face_4_encodings:
        face_4_encoding = face_4_encodings[0]
    else:
        print("No face encodings found in the fourth image.")

    

    face_6 = face_recognition.load_image_file("/content/524.jpg")
    face_6_encodings = face_recognition.face_encodings(face_6)
    if face_6_encodings:
        face_6_encoding = face_6_encodings[0]
    else:
        print("No face encodings found in the sixth image.")

    known_face_encodings = [
         face_2_encoding,
    face_4_encoding, face_6_encoding
    
    ]
    known_face_names = [
        "Vignesh",
        "Vignesh- Suspect",
        "Vignesh -",
        

    ]
    frame_count = 0  # count no of frames
    total_fps = 0  # count total fps
    time_list = []   # list to store time
    fps_list = []    # list to store fps

    device = select_device(device)  # select device
    half = device.type != 'cpu'
    flowers = tf.keras.models.load_model("/content/drive/MyDrive/latestaction.h5")
    model = attempt_load(poseweights, map_location=device)  # Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    print(names)

    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))  # pass video to videocapture object
    else:
        cap = cv2.VideoCapture(source)  # pass video to videocapture object

    if not cap.isOpened():  # check if videocapture not opened
        print('Error while trying to read video. Please check the path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  # get video frame width
        frame_height = int(cap.get(4))  # get video frame height

        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]  # init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                             cv2.VideoWriter_fourcc(*'mp4v'), 30,
                             (resize_width, resize_height))

        # Define a threshold for jump detection (you can adjust this)
        jump_threshold = 50  # Adjust as needed

        while cap.isOpened:  # loop until cap opened or video not complete

            print("Frame {} Processing".format(frame_count + 1))

            ret, frame = cap.read()  # get frame and success from video capture

            if ret:  # if success is true, means frame exists
                orig_image = frame  # store frame

                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)  # convert image data to device
                image = image.float()  # convert image to float precision (cpu)
                start_time = time.time()  # start time for fps calculation

                with torch.no_grad():  # get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,   # Apply non-max suppression
                                                      0.25,   # Conf. Threshold.
                                                      0.65,   # IoU Threshold.
                                                      nc=model.yaml['nc'],  # Number of classes.
                                                      nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                                      kpt_label=True)

                output = output_to_keypoint(output_data)

                im0 = image[0].permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)

                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for i, pose in enumerate(output_data):  # detections per image
                    if len(pose):  # check if there are detected poses
                        text =""
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):  # loop over poses for drawing on the frame
                            c = int(cls)  # integer class
                            img = cv2.resize(im0, (224, 224))
                            img = np.reshape(img, [1, 224, 224, 3])
                            img = img / 255
                            preds = flowers.predict(img)
                            max_idx = np.argmax(preds)
                            act = np.max(preds)
                            acc = str(round(act*100,3))
                            image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
                            image_file_like = io.BytesIO(image_bytes)
                            unknown_image = face_recognition.load_image_file(image_file_like)
                            face_locations = face_recognition.face_locations(unknown_image)
                            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                            pil_image = Image.fromarray(unknown_image)
                            draw = ImageDraw.Draw(pil_image)

                            

# ... (previous code)

                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            # See if the face is a match for the known face(s)
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                            name = "Unknown"

                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]

                            # Convert the image to a PIL Image so you can use the draw module
                            pil_image = Image.fromarray(frame)

                            # Draw a rectangle around the face
                            draw = ImageDraw.Draw(pil_image)
                            draw.rectangle([left, top, right, bottom], outline=(255, 0, 0))

                            # Draw a label with a name below the face
                            text_width, text_height = draw.textsize(name)
                            draw.rectangle([(left, bottom - text_height - 10), (right, bottom)], fill=(0, 0, 255), outline=(0, 0, 255))
                            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

                            # Convert the PIL Image back to an OpenCV image
                            frame = np.array(pil_image)

# ... (rest of the code)


                            
                            kpts = pose[det_index, 6:]
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                             orig_shape=im0.shape[:2])

                        # Calculate the action for each person based on max_idx
                            if max_idx == 0:
                                text = "Jumping"
                                font_color = (0, 0, 255)
                                fill_color = (0, 0, 255)
                                x1, y1, x2, y2 = xyxy  # Extract coordinates
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
                                cv2.rectangle(im0, (x1, y1), (x2, y2), fill_color)
                            elif max_idx == 1:
                                text = "Running"
                                fill_color = (0, 0, 255)
                                font_color = (0, 0, 255)
                                x1, y1, x2, y2 = xyxy  # Extract coordinates
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
                                cv2.rectangle(im0, (x1, y1), (x2, y2), fill_color)
                            elif max_idx == 2:
                                text = "Walking"
                                font_color = (0, 255, 0)

                                # Add the action text above each person's bounding box
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
                            text_position = (int(xyxy[0]) + 50, int(xyxy[1]) - 10)  # Adjust the text position
                            text_position = (int(xyxy[0]) + 50, int(xyxy[1]) - 10)  # Adjust the text position
                            rect_position = (text_position[0], text_position[1] - text_height - baseline+30) # Adjust the rectangle position

                            # Draw the filled rectangle as the background for the text
                            cv2.rectangle(im0, (rect_position[0], rect_position[1] - text_height),
                                          (rect_position[0] + text_width, rect_position[1] + baseline),
                                          (200, 200, 200, 128), cv2.FILLED)
                            cv2.rectangle(im0, (40,50),(370,90),
                                          (200, 200, 200, 128), cv2.FILLED)
                            print(preds, " ", text)
                            atc = "Accuracy: "
                            cv2.putText(im0, atc, (50, 80), font, font_scale, (0, 255, 0), 2)
                            cv2.putText(im0, acc + "%", (220, 80), font, font_scale, (0, 255, 0), 2)
                            cv2.putText(im0, text, text_position, font, font_scale, font_color, 2)

                end_time = time.time()  # Calculation for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1

                fps_list.append(total_fps)  # append FPS in the list
                time_list.append(end_time - start_time)  # append time in the list

                # Stream results
                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)  # 1 millisecond

                out.write(im0)  # writing the video frame

            else:
                break

        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")

        # Plot the comparison graph
        plot_fps_time_comparison(time_list=time_list, fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam')  # video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')  # device arguments
    parser.add_argument('--view-img', action='store_true', help='display results')  # display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')  # save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')  # box line thickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # box hide label
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # box hide conf
    opt = parser.parse_args()
    return opt

# Function for plotting FPS and time comparison graph
def plot_fps_time_comparison(time_list, fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparison Graph')
    plt.plot(time_list, fps_list, 'b', label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparison_pose_estimate.png")

# Main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)

