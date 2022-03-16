import logging
import cv2
import os

from counter import Counter


def config_logs():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s][Thread %(threadName)s %(thread)d] %(filename)s:%(lineno)s -> %(message)s')


def main():
    config_logs()
    logging.info("Starting ball juggling counter.")
    path = os.getcwd()
    video_path = path + "/test_data/Outside_Trash.mp4"
    cap = cv2.VideoCapture(video_path)
    juggle_counter = Counter()
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret:
                juggle_counter.count_juggles(frame)
                cv2.imshow("Ball Juggling Counter", frame)
            if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Ball Juggling Counter', cv2.WND_PROP_VISIBLE) < 1:
                break
        except ZeroDivisionError as ex:
            logging.warning(ex)
        except Exception as ex:
            logging.error(ex)
            exit(-1)
    cv2.destroyAllWindows()
    exit(0)


if __name__ == "__main__":
    main()
