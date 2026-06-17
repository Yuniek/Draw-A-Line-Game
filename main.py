import cv2
import math
import numpy as np
from ball import Ball
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import random
import pymunk

from functions import update_state, point_line_distance

MODEL_PATH = "hand_landmarker.task"
HAND_MODE = "right"   # "right", "left"

# Generate on the right side logically, so it mirrors to the bottom left visually
BIN_W = 80
BIN_H = 80
BIN_X = random.randint(450, 540)
BIN_Y = random.randint(300, 380)
hand_state = "Hover"

drawn_lines = []
physics_lines = []
MAX_LINES = 200 # Optimization: limit active lines
START_GESTURE_TIME = 3.0
WIN_TIME = 2.0

x = random.randint(250, 390)
y = random.randint(20, 80)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

success, init_frame = cap.read()
if success:
    h, w, _ = init_frame.shape
else:
    w, h = 640, 480

space = pymunk.Space()
space.gravity = (0, 1500)

boundaries = [
    pymunk.Segment(space.static_body, (0, 0), (w, 0), 0),
    pymunk.Segment(space.static_body, (w, 0), (w, h), 0),
    pymunk.Segment(space.static_body, (w, h), (0, h), 0),
    pymunk.Segment(space.static_body, (0, h), (0, 0), 0)
]
for b in boundaries:
    b.elasticity = 0.9
    b.friction = 0.5
space.add(*boundaries)

bin_walls = [
    pymunk.Segment(space.static_body, (BIN_X, BIN_Y), (BIN_X, BIN_Y + BIN_H), 4),
    pymunk.Segment(space.static_body, (BIN_X + BIN_W, BIN_Y), (BIN_X + BIN_W, BIN_Y + BIN_H), 4),
    pymunk.Segment(space.static_body, (BIN_X, BIN_Y + BIN_H), (BIN_X + BIN_W, BIN_Y + BIN_H), 4)
]
for bw in bin_walls:
    bw.elasticity = 0.5
    bw.friction = 0.5
space.add(*bin_walls)

ball = Ball(space, x, y)

prevX, prevY = (None, None)
game_started = False
start_gesture_timer = 0
time_in_bin = 0

with HandLandmarker.create_from_options(options) as landmarker:

    frame_count = 0

    while cap.isOpened():
        success, CameraFrame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(CameraFrame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = frame_count * 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        h, w, _ = CameraFrame.shape

        GameScreen = np.zeros_like(CameraFrame)

        if result.hand_landmarks:

            # Same landmark connections as MediaPipe
            connections = [
                (0,1),(1,2),(2,3),(3,4),       # Thumb
                (0,5),(5,6),(6,7),(7,8),       # Index
                (5,9),(9,10),(10,11),(11,12),  # Middle
                (9,13),(13,14),(14,15),(15,16),# Ring
                (13,17),(17,18),(18,19),(19,20), # Pinky
                (0,17)                         # Palm base
            ]

            for hand_index, hand in enumerate(result.hand_landmarks):

                handedness = result.handedness[hand_index][0].category_name.lower()

                if HAND_MODE == "right" and handedness != "right":
                    continue

                if HAND_MODE == "left" and handedness != "left":
                    continue

                hand_state = update_state(hand, HAND_MODE)

                points = []

                # Convert normalized coordinates to pixels with sensitivity scaling
                # This allows reaching the edges of the screen without the hand leaving the camera view
                for landmark in hand:

                    # Scale outward from the center (0.5) to increase reach
                    nx = (landmark.x - 0.5) * 1.3 + 0.5
                    ny = (landmark.y - 0.5) * 1.4 + 0.5

                    x = int(nx * w)
                    y = int(ny * h)

                    points.append((x, y))

                # Draw white lines
                for start, end in connections:

                    cv2.line(
                        GameScreen,
                        points[start],
                        points[end],
                        (255, 255, 255),
                        2
                    )

                # Draw larger white dots
                for x, y in points:

                    cv2.circle(
                        GameScreen,
                        (x, y),
                        4,
                        (255, 255, 255),
                        -1
                    )

                if hand_state == "draw":
                    if prevX is None:
                        prevX, prevY = points[8]
                    else:
                        new_line = ((prevX, prevY), points[8])
                        drawn_lines.append(new_line)
                        
                        segment = pymunk.Segment(space.static_body, (prevX, prevY), points[8], 4)
                        segment.elasticity = 0.8
                        segment.friction = 0.5
                        space.add(segment)
                        physics_lines.append(segment)

                        # Cap lines to avoid lag
                        if len(drawn_lines) > MAX_LINES:
                            drawn_lines.pop(0)
                            old_segment = physics_lines.pop(0)
                            space.remove(old_segment)
                            
                        prevX, prevY = points[8]
                else:
                    prevX, prevY = (None, None)

        # Handle Start Gesture
        if not game_started:
            if hand_state == "start_gesture":
                if start_gesture_timer == 0:
                    start_gesture_timer = time.time()
                elif time.time() - start_gesture_timer >= START_GESTURE_TIME:
                    game_started = True
            else:
                start_gesture_timer = 0

        # Step Physics
        if ball.active:
            if game_started:
                dt = 1.0 / 30.0
                space.step(dt / 3)
                space.step(dt / 3)
                space.step(dt / 3)

            in_bin = (
                BIN_X + ball.radius < ball.x < BIN_X + BIN_W - ball.radius and
                BIN_Y < ball.y < BIN_Y + BIN_H
            )
            if in_bin:
                if time_in_bin == 0:
                    time_in_bin = time.time()
                elif time.time() - time_in_bin >= WIN_TIME:
                    ball.deactivate(space)
            else:
                time_in_bin = 0

            if ball.active:
                # Draw ball
                cv2.circle(
                    GameScreen,
                    (int(ball.x), int(ball.y)),
                    ball.radius,
                    (0,255,255),
                    -1
                )
        

        # Draw dynamic green lines
        for p1, p2 in drawn_lines:
            cv2.line(GameScreen, p1, p2, (0, 255, 0), 8)

        # Bin        
        cv2.line(
            GameScreen,
            (BIN_X, BIN_Y),
            (BIN_X, BIN_Y + BIN_H),
            (0,0,255),
            5
        )

        cv2.line(
            GameScreen,
            (BIN_X + BIN_W, BIN_Y),
            (BIN_X + BIN_W, BIN_Y + BIN_H),
            (0,0,255),
            5
        )

        cv2.line(
            GameScreen,
            (BIN_X, BIN_Y + BIN_H),
            (BIN_X + BIN_W, BIN_Y + BIN_H),
            (0,0,255),
            5
        )


        GameScreen = cv2.flip(GameScreen, 1)

        # Win Mechanism (Draw text AFTER flipping so it is readable)
        if not ball.active:
            # Draw background box for text
            cv2.rectangle(GameScreen, (190, 140), (510, 220), (0, 0, 0), -1)
            cv2.putText(
                GameScreen,
                "YOU WIN",
                (200,200),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0,255,0),
                4
            )

        cv2.imshow("Hand Tracking", GameScreen)

        

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()