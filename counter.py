import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


class Counter:

    def __init__(self, color_lower=np.array([25, 80, 20]), color_upper=np.array([40, 180, 255])):
        self.counter = 0
        self.count = '0'
        self.x_centers = []
        self.y_centers = []
        self.backSub = cv2.createBackgroundSubtractorKNN()
        self.background_subtract = True
        self.colorLower = color_lower
        self.colorUpper = color_upper

    def count_juggles(self, frame) -> int:
        # background subtract
        fg_mask = self.backSub.apply(frame)
        # bitwise multiplication to grab moving part
        new_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

        # identify the ball
        mask = self.ball_finder(new_frame, self.colorLower, self.colorUpper)

        # canny edge detection
        edges = cv2.Canny(mask, 100, 200)
        # find contours of the ball
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # write frame number in top left
        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)

        new_frame = cv2.drawContours(new_frame, contours, -1, (0, 255, 0), 3)

        # get rid of excess contours and do analysis
        if len(contours) > 0:
            # get the largest contour
            c = max(contours, key=cv2.contourArea)
            # find the center of the ball
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # save the center coordinates to a list
            self.x_centers.append(center[0])
            self.y_centers.append(center[1])
            peaks = self.peak_calculator(self.y_centers, int(self.count))

            # draw center and contour onto frame
            cv2.circle(frame, center, 8, (255, 255, 0), -1)
            img = cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

            # put number of peaks on frame
            cv2.putText(frame, peaks, (150, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 2)

            # say outloud the count
            if self.count != peaks:
                self.count = peaks

        return self.counter

    # curtosy of https://github.com/Enoooooormousb
    @staticmethod
    def ball_finder(frame, hsv_lower, hsv_upper):
        # blur the image to reduce noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        # convert to hsv color space for color filtering
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color specified
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        # eliminate low color signal
        mask = cv2.erode(mask, None, iterations=2)
        # expand the strong color signal
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    # count number of peaks in trace so far. Returns # of peaks
    @staticmethod
    def peak_calculator(height, cur_num_peaks):
        if len(height) > 9:
            # invert and filter input
            y = savgol_filter(height, 9, 2)
            # use peak finder
            peaks, _ = find_peaks(y)
            if len(peaks) < cur_num_peaks:
                peaks = [cur_num_peaks]
        else:
            peaks = [0]
        return str(len(peaks))
