import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import time
import pyttsx3


MODEL_PATH = "exported_models/my_ssd_mobilenet/saved_model"

LABELS = {
    1: "hello",
    2: "thanks",
    3: "yes",
    4: "no",
    5: "iloveyou"
}

CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_FRAMES = 8


def smooth_box(previous_box, current_box, alpha=0.7):
    if previous_box is None:
        return current_box

    return [
        int(alpha * previous_box[0] + (1 - alpha) * current_box[0]),
        int(alpha * previous_box[1] + (1 - alpha) * current_box[1]),
        int(alpha * previous_box[2] + (1 - alpha) * current_box[2]),
        int(alpha * previous_box[3] + (1 - alpha) * current_box[3]),
    ]


def draw_panel(frame, stable_label, stable_score, sentence, fps):
    height, width, _ = frame.shape

    cv2.rectangle(frame, (0, 0), (width, 115), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Detected: {stable_label}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {stable_score * 100:.1f}%",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (width - 140, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    sentence_text = " ".join(sentence)
    if len(sentence_text) > 60:
        sentence_text = sentence_text[-60:]

    cv2.putText(
        frame,
        f"Sentence: {sentence_text}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


def draw_controls(frame):
    height, width, _ = frame.shape

    controls = "A: Add word | S: Speak | C: Clear | B: Backspace | Q: Quit"

    cv2.rectangle(frame, (0, height - 35), (width, height), (0, 0, 0), -1)

    cv2.putText(
        frame,
        controls,
        (20, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


def main():
    print("Loading model...")
    detect_fn = tf.saved_model.load(MODEL_PATH)
    print("Model loaded successfully.")

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    label_history = deque(maxlen=SMOOTHING_FRAMES)
    score_history = deque(maxlen=SMOOTHING_FRAMES)

    sentence = []

    last_box = None
    stable_label = "Detecting..."
    stable_score = 0.0

    last_added_label = None
    last_added_time = 0

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

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

            most_common_label, count = Counter(label_history).most_common(1)[0]

            if count >= 4:
                stable_label = most_common_label
                stable_score = sum(score_history) / len(score_history)
                last_box = smooth_box(last_box, current_box)

        else:
            label_history.append("none")

            if label_history.count("none") >= 6:
                stable_label = "Detecting..."
                stable_score = 0.0
                last_box = None

        if last_box is not None and stable_label != "Detecting...":
            left, top, right, bottom = last_box

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{stable_label}: {stable_score * 100:.1f}%",
                (left, max(top - 10, 130)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

        draw_panel(frame, stable_label, stable_score, sentence, fps)
        draw_controls(frame)

        cv2.imshow("AI-Powered Real-Time Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("a"):
            if stable_label != "Detecting...":
                # Avoid accidental duplicate additions too quickly
                now = time.time()
                if stable_label != last_added_label or now - last_added_time > 1.5:
                    sentence.append(stable_label)
                    last_added_label = stable_label
                    last_added_time = now
                    print(f"Added word: {stable_label}")

        elif key == ord("c"):
            sentence.clear()
            print("Sentence cleared.")

        elif key == ord("b"):
            if sentence:
                removed = sentence.pop()
                print(f"Removed word: {removed}")

        elif key == ord("s"):
            if sentence:
                text = " ".join(sentence)
                print(f"Speaking: {text}")
                engine.say(text)
                engine.runAndWait()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()