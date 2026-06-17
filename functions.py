def get_finger_state(hand, HAND_MODE):

    fingers = []

    if HAND_MODE == "left":
        fingers.append(hand[4].x < hand[3].x)
    else:
        fingers.append(hand[4].x > hand[3].x)

    fingers.append(hand[8 ].y < hand[6 ].y)
    fingers.append(hand[12].y < hand[10].y)
    fingers.append(hand[16].y < hand[14].y)
    fingers.append(hand[20].y < hand[18].y)

    return "".join([str(int(i)) for i in fingers])


def update_state(hand, HAND_MODE):
    fingerState = get_finger_state(hand, HAND_MODE)

    if fingerState == "01000":
        return "draw"
    elif fingerState == "01110":
        return "start_gesture"
    else:
        return "hover"

def point_line_distance(px, py, x1, y1, x2, y2):
    import math
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)

    t = max(0, min(1, t))

    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    return math.hypot(
        px - nearest_x,
        py - nearest_y
    )