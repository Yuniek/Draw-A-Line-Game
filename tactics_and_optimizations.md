# Draw A Line Game: Tactics & Optimizations

This document outlines the mechanics, optimizations, and libraries utilized to build this interactive hand-tracking physics game. The goal of this game was to provide a smooth, lag-free experience where the player dynamically interacts with simulated physics using their hands via computer vision.

## 🚀 Modules Used

- **OpenCV (`cv2`)**: Used for all the drawing, image processing, video capture from the webcam, and rendering the `GameScreen`.
- **MediaPipe (`mediapipe`)**: Powered by Google, MediaPipe's `HandLandmarker` is the core AI that tracks hand nodes in real-time. It accurately locates 21 landmarks on the hands to detect states (like "draw" mode vs "hover" mode).
- **Pymunk (`pymunk`)**: A robust, industry-standard 2D physics engine built on top of Chipmunk2D. It handles all gravity, friction, elasticity (bouncing), and rigid body collision detection, ensuring flawless interactions without tunneling or bugs.
- **NumPy (`numpy`)**: Used to create blank image arrays (`np.zeros_like`) to generate the canvas for the game overlay efficiently.
- **Math (`math`)**: Handles vector normalizations.

## 🧠 Tactics & Game Logic

### Mirroring the Real World
The camera frame remains untouched for the physics math, but the final `GameScreen` is horizontally mirrored right before display (`cv2.flip(GameScreen, 1)`). Doing this ensures that the camera perfectly matches the mirrored reality, giving the user intuitive hand-eye coordination.

### Bin Mechanics
The objective "bin" is randomly generated at the bottom left. Instead of being an abstract visual object, it is actively injected into the physics engine:
- The walls of the bin (Left, Right, Bottom) act as solid `pymunk.Segment`s that bounce the ball away.
- The top boundary of the bin is open, and if the ball successfully clears the vertical boundaries, the game logic registers a win, deactivating the ball.

### Dynamic Interaction System
Rather than keeping all drawn lines forever, the game features a dynamic decaying line mechanic. Only the newest `MAX_LINES` (e.g., 150) are retained. If the player keeps drawing, the older lines disappear. This forces continuous interaction and prevents the player from "trapping" the ball easily, keeping the game challenging and engaging.

## ⚙️ Performance Optimizations (Avoiding Lag)

To keep the application running at 30+ FPS, the following optimizations were critical:

### 1. Robust Physics Stepping (Anti-Tunneling)
When gravity accelerates a ball to high velocities, a standard frame-by-frame distance check will often fail because the ball jumps entirely past a thin drawn line between frames (tunneling). To fix this, `pymunk`'s solver is stepped multiple times per frame (`space.step(dt / 3)`). This ensures continuous collision accuracy without sacrificing speed.

### 2. Line Limits (Garbage Collection)
The `drawn_lines` list and `physics_lines` list act as a queue. When their length exceeds the `MAX_LINES` threshold, the oldest segment is popped (`drawn_lines.pop(0)`) and safely removed from the `pymunk.Space`. This puts a hard ceiling on the number of computations required by the collision detection loop, preventing memory leaks and physics lag.

### 3. Canvas Redrawing vs. Copying
Instead of maintaining a massive static `GameCanvas` that needs copying every frame, the game dynamically reconstructs the screen layout per frame (`GameScreen = np.zeros_like(CameraFrame)`). By leveraging OpenCV's hyper-optimized C++ backend, redrawing the active queue of lines directly on the `GameScreen` each frame is actually faster and more flexible than copying and managing a persistent NumPy buffer.
