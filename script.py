import seaborn as sns
import cv2
import numpy as np
import time
import concurrent.futures
import matplotlib.pyplot as plt
import folium

# Global variable for the classifier
car_cascade = None

def init():
    global car_cascade
    car_cascade = cv2.CascadeClassifier('cars2.xml')

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (400, 400))

def segment_image(img):
    return cv2.Canny(img, 50, 150)

def detect_cars(img):
    global car_cascade
    height, width = img.shape
    roi = img[int(height/2):, int(width*0.375):]
    cars = car_cascade.detectMultiScale(roi, 1.1, 3)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in cars:
        x += int(width*0.375)
        y += int(height/2)
        cv2.rectangle(img_bgr, (x,y), (x+w,y+h), (0,255,0), 2)
    return img_bgr, len(cars)

def detect_cars_full_frame(img):
    global car_cascade
    cars = car_cascade.detectMultiScale(img, 1.1, 3)
    return len(cars)

def apply_background_subtraction(previous_frame, current_frame, alpha):
    return cv2.absdiff(previous_frame, current_frame), cv2.addWeighted(previous_frame, 1-alpha, current_frame, alpha, 0)

def apply_optical_flow(previous_frame_gray, current_frame_gray):
    return cv2.calcOpticalFlowFarneback(previous_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def draw_flow(img, flow, step=16):
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    h, w = vis.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    return vis

def overlay_images(image1, image2):
    return cv2.addWeighted(image1, 0.7, image2, 0.3, 0)

def overlay_images(image1, image2):
    return cv2.addWeighted(image1, 0.7, image2, 0.3, 0)

def calculate_land_cover_change_ratio(foreground_mask):
    non_black_pixels = np.sum(foreground_mask != 0)
    total_pixels = foreground_mask.size
    return non_black_pixels / total_pixels

def main():
    video_file = 'dataset/video-C1-20221011-1.mp4'
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = 1
    frame_count = 0
    counts = []
    time_stamps = []

    video_length_in_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_length_in_seconds = video_length_in_frames / fps
    print(f"Video length: {video_length_in_seconds} seconds")

    out = open('output/segmented_output.raw', 'wb')

    ret, frame = cap.read()
    processed_frame = process_image(frame)
    avg = np.float32(processed_frame)
    previous_frame_gray = np.copy(processed_frame)

    start_time = time.time()

    counts_full_frame = []
    time_stamps_full_frame = []
    land_cover_change_ratios = []  # list to store the land cover change ratio for each frame
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=init) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_rate == 0:
                processed_frame = process_image(cv2.resize(frame, (400, 400)))
                foreground_mask, avg = apply_background_subtraction(avg, np.float32(processed_frame), alpha=0.02)

                land_cover_change_ratio = calculate_land_cover_change_ratio(foreground_mask)
                land_cover_change_ratios.append(land_cover_change_ratio)  # store the ratio
                
                future_optical_flow = executor.submit(apply_optical_flow, previous_frame_gray, processed_frame) if frame_count > 1 else None
                future_detect_cars = executor.submit(detect_cars, processed_frame)  # ROI for vehicle detection
                future_detect_cars_full_frame = executor.submit(detect_cars_full_frame, processed_frame)  # Full frame for vehicle detection
                future_segment_image = executor.submit(segment_image, processed_frame)
                
                optical_flow_frame = None
                if future_optical_flow:
                    flow = future_optical_flow.result()
                    optical_flow_frame = draw_flow(processed_frame, flow)
                
                detected_frame, count = future_detect_cars.result()
                counts.append(count)
                time_stamps.append(frame_count / fps)

                count_full_frame = future_detect_cars_full_frame.result()
                counts_full_frame.append(count_full_frame)
                time_stamps_full_frame.append(frame_count / fps)

                if optical_flow_frame is not None:
                    optical_flow_and_detection = overlay_images(optical_flow_frame, detected_frame)
                    cv2.imshow('Optical Flow and Vehicle Detection', optical_flow_and_detection)

                segmented_frame = future_segment_image.result()
                out.write(segmented_frame.tobytes())

                cv2.imshow('Foreground Mask', foreground_mask)

                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

                previous_frame_gray = np.copy(processed_frame)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"processing factor: {elapsed_time/video_length_in_seconds}")
    
    # After processing all frames, plot the land cover change ratios
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, land_cover_change_ratios)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Land Cover Change Ratio')
    plt.title('Land Cover Change Ratio Over Time')
    plt.savefig('output/land_cover_change_ratio.png')
    
    heatmap_data = np.array(counts_full_frame).reshape(-1, 1)
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(heatmap_data, cmap='viridis', yticklabels=False)  # Turn off default y-tick labels
    n = len(time_stamps_full_frame)
    ticks = ax.get_yticks()
    ax.set_yticks([n-1 if t == max(ticks) else t for t in ticks])  # Re-adjust y-ticks to match timestamps
    ax.set_yticklabels([str(time_stamps_full_frame[int(t)]) for t in ax.get_yticks()])  # Set y-tick labels as timestamps
    plt.ylabel('Time (seconds)')
    plt.xlabel('Vehicle Count')
    plt.title('Vehicle Count Over Time (Full Frame Heatmap)')
    plt.savefig('output/vehicle_count_full_frame_heatmap.png')

    # Existing plot for ROI vehicle count
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, counts)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Vehicle Count (Lane)')
    plt.title('Vehicle Count Over Time (Lane)')
    plt.savefig('output/vehicle_count_lane.png')
    
    # lat, lon = 43.160133, 12.949957

    # # Create a map centered at the location
    # m = folium.Map(location=[lat, lon], zoom_start=13)

    # # Add a marker to the map for the location
    # folium.Marker([lat, lon], popup='Traffic Congestion').add_to(m)

    # # Save it as html
    # m.save('map.html')
    
    cap.release()
    out.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()