import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter


MODEL_PATH = "exported_models/my_ssd_mobilenet/saved_model"

LABELS = {
    1: "hello",
    2: "thanks",
    3: "yes",
    4: "no",
    5: "iloveyou"
}

CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_FRAMES = 8


def smooth_box(previous_box, current_box, alpha=0.7):
    """
    Smooth bounding box movement.
    alpha closer to 1 means more stable but slower movement.
    """
    if previous_box is None:
        return current_box

    return [
        int(alpha * previous_box[0] + (1 - alpha) * current_box[0]),
        int(alpha * previous_box[1] + (1 - alpha) * current_box[1]),
        int(alpha * previous_box[2] + (1 - alpha) * current_box[2]),
        int(alpha * previous_box[3] + (1 - alpha) * current_box[3]),
    ]


def main():
    print("Loading model...")
    detect_fn = tf.saved_model.load(MODEL_PATH)
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    label_history = deque(maxlen=SMOOTHING_FRAMES)
    score_history = deque(maxlen=SMOOTHING_FRAMES)

    last_box = None
    stable_label = "Detecting..."
    stable_score = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            break

        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy()
            for key, value in detections.items()
        }

        detection_boxes = detections["detection_boxes"]
        detection_classes = detections["detection_classes"].astype(np.int64)
        detection_scores = detections["detection_scores"]

        height, width, _ = frame.shape

        best_score = 0
        best_class = None
        best_box = None

        # Pick only the best detection from this frame
        for i in range(num_detections):
            score = detection_scores[i]

            if score > best_score:
                best_score = score
                best_class = detection_classes[i]
                best_box = detection_boxes[i]

        if best_score >= CONFIDENCE_THRESHOLD and best_class in LABELS:
            label = LABELS[best_class]

            ymin, xmin, ymax, xmax = best_box

            current_box = [
                int(xmin * width),
                int(ymin * height),
                int(xmax * width),
                int(ymax * height)
            ]

            label_history.append(label)
            score_history.append(best_score)

            # Most common label in recent frames
            most_common_label, count = Counter(label_history).most_common(1)[0]

            # Only accept label if it appears repeatedly
            if count >= 4:
                stable_label = most_common_label
                stable_score = sum(score_history) / len(score_history)
                last_box = smooth_box(last_box, current_box)

        else:
            # Do not instantly remove box; wait a little
            label_history.append("none")

            if label_history.count("none") >= 6:
                stable_label = "Detecting..."
                stable_score = 0
                last_box = None

        # Draw stable box
        if last_box is not None and stable_label != "Detecting...":
            left, top, right, bottom = last_box

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            text = f"{stable_label}: {stable_score * 100:.1f}%"

            cv2.putText(
                frame,
                text,
                (left, max(top - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "Detecting...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow("Real-Time Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()