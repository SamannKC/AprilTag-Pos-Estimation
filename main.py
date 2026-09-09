import cv2
import numpy as np
from pupil_apriltags import Detector

# tags ids
TOP_LEFT = 0
TOP_RIGHT = 1
BOTTOM_RIGHT = 2
BOTTOM_LEFT = 3
MIDDLE_TAG = 4


# Distance between the cneter of the four tags 

FIELD_WIDTH_MM = 500.0
FIELD_HEIGHT_MM = 500.0


# tag size and phyiscal marker size are taken differently as the program detects the inner tag and not eh entire physical boundary

# physical marker size
OUTER_TAG_WIDTH_MM = 85.0
OUTER_TAG_HEIGHT_MM = 90.0

#inner april tag size
INNER_TAG_WIDTH_MM = 62.0
INNER_TAG_HEIGHT_MM = 62.0

FOCAL_LENGTH_PX = 700.0

cap = cv2.VideoCapture(2)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1920
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    1080
)


#opencv
WINDOW_NAME = "AprilTag Height Estimation"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1600,
    900
)


# tag detector 
detector = Detector(
    families="tagStandard41h12"
)


def get_outer_corners(corners):

    center = np.mean(
        corners,
        axis=0
    )

    scale_x = (
        OUTER_TAG_WIDTH_MM /
        INNER_TAG_WIDTH_MM
    )

    scale_y = (
        OUTER_TAG_HEIGHT_MM /
        INNER_TAG_HEIGHT_MM
    )

    outer = corners.copy()

    outer[:, 0] = (
        center[0]
        +
        (corners[:, 0] - center[0])
        * scale_x
    )

    outer[:, 1] = (
        center[1]
        +
        (corners[:, 1] - center[1])
        * scale_y
    )

    return outer


def draw_marker(frame,tag_id,center,outer_corners):

    for i in range(4):

        p1 = tuple(
            outer_corners[i].astype(int)
        )

        p2 = tuple(
            outer_corners[
                (i + 1) % 4
            ].astype(int)
        )

        cv2.line(
            frame,
            p1,
            p2,
            (0, 255, 0),
            4
        )



    x = int(center[0])
    y = int(center[1])

    cv2.circle(
        frame,
        (x, y),
        7,
        (0, 0, 255),
        -1
    )


    #id
    cv2.putText(
        frame,
        f"ID {tag_id}",
        (x + 12, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


def estimate_depth(corners):

    # four side lengths
    side_1 = np.linalg.norm(
        corners[0] - corners[1]
    )

    side_2 = np.linalg.norm(
        corners[1] - corners[2]
    )

    side_3 = np.linalg.norm(
        corners[2] - corners[3]
    )

    side_4 = np.linalg.norm(
        corners[3] - corners[0]
    )


    # average apparent size
    pixel_size = (
        side_1 +
        side_2 +
        side_3 +
        side_4
    ) / 4.0


    if pixel_size <= 0:
        return None


    depth = (
        FOCAL_LENGTH_PX *
        INNER_TAG_WIDTH_MM
    ) / pixel_size


    return depth


def draw_distance_line(frame,p1,p2,distance_mm):

    p1 = tuple(
        np.asarray(p1, dtype=int)
    )

    p2 = tuple(
        np.asarray(p2, dtype=int)
    )


    cv2.line(
        frame,
        p1,
        p2,
        (255, 0, 0),
        3
    )

    #midpoint approx
    mx = int(
        (p1[0] + p2[0]) / 2
    )

    my = int(
        (p1[1] + p2[1]) / 2
    )


    text = f"{distance_mm:.1f} mm"


    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    cv2.rectangle(
        frame,
        (
            mx - 5,
            my - th - 7
        ),
        (
            mx + tw + 5,
            my + baseline + 5
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        text,
        (mx, my),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera read failed.")
        break


    height, width = frame.shape[:2]


    # detect tags
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    detections = detector.detect(
        gray
    )


    tags = {}


    # store detections
    for detection in detections:

        tag_id = detection.tag_id


        if tag_id not in [
            0,
            1,
            2,
            3,
            4
        ]:
            continue


        inner_corners = (
            detection.corners.astype(
                np.float32
            )
        )


        center = np.array(
            detection.center,
            dtype=np.float32
        )


        outer_corners = get_outer_corners(
            inner_corners
        )

        depth = estimate_depth(
            inner_corners
        )


        tags[tag_id] = {

            "center": center,

            "inner": inner_corners,

            "outer": outer_corners,

            "depth": depth
        }


    for tag_id, tag in tags.items():

        draw_marker(
            frame,
            tag_id,
            tag["center"],
            tag["outer"]
        )


    reference_ids = [
        TOP_LEFT,
        TOP_RIGHT,
        BOTTOM_RIGHT,
        BOTTOM_LEFT
    ]


    reference_depths = []


    for tag_id in reference_ids:

        if tag_id in tags:

            depth = tags[tag_id]["depth"]

            if depth is not None:
                reference_depths.append(
                    depth
                )


    if len(reference_depths) > 0:

        reference_depth = np.mean(
            reference_depths
        )

    else:

        reference_depth = None


    y_text = 210

    for tag_id in reference_ids:

        if tag_id not in tags:
            continue

        depth = tags[tag_id]["depth"]

        if depth is None:
            continue

        cv2.putText(
            frame,
            f"ID {tag_id} Depth: {depth:.1f} mm",
            (20, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        y_text += 30



    if MIDDLE_TAG in tags:

        middle = tags[
            MIDDLE_TAG
        ]["center"]

        middle_depth = tags[
            MIDDLE_TAG
        ]["depth"]


        if middle_depth is not None:

            cv2.putText(
                frame,
                f"Middle Depth: {middle_depth:.1f} mm",
                (20, y_text + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        if (
            reference_depth is not None
            and
            middle_depth is not None
        ):

            middle_height = (
                reference_depth -
                middle_depth
            )


            cv2.putText(
                frame,
                f"HEIGHT: {middle_height:.1f} mm",
                (20, y_text + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )


            cv2.putText(
                frame,
                f"Reference Depth: "
                f"{reference_depth:.1f} mm",
                (20, y_text + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        for tag_id in reference_ids:

            if tag_id not in tags:
                continue


            outer_center = tags[
                tag_id
            ]["center"]



    all_references = all(
        tag_id in tags
        for tag_id in reference_ids
    )


    if all_references:

        image_points = np.array([
            tags[TOP_LEFT]["center"],
            tags[TOP_RIGHT]["center"],
            tags[BOTTOM_RIGHT]["center"],
            tags[BOTTOM_LEFT]["center"]
        ], dtype=np.float32)


        world_points = np.array([
            [0, 0],

            [
                FIELD_WIDTH_MM,
                0
            ],

            [
                FIELD_WIDTH_MM,
                FIELD_HEIGHT_MM
            ],

            [
                0,
                FIELD_HEIGHT_MM
            ]
        ], dtype=np.float32)


        H, status = cv2.findHomography(
            image_points,
            world_points
        )


        if H is not None:
            if MIDDLE_TAG in tags:

                middle = tags[
                    MIDDLE_TAG
                ]["center"]


                middle_pixel = np.array(
                    [[middle]],
                    dtype=np.float32
                )


                middle_world = (
                    cv2.perspectiveTransform(
                        middle_pixel,
                        H
                    )
                )


                middle_x = (
                    middle_world[0][0][0]
                )

                middle_y = (
                    middle_world[0][0][1]
                )



                cv2.putText(
                    frame,
                    f"Middle X: "
                    f"{middle_x:.1f} mm",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Middle Y: "
                    f"{middle_y:.1f} mm",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2
                )


                for tag_id in reference_ids:

                    outer_center = tags[
                        tag_id
                    ]["center"]


                    outer_pixel = np.array(
                        [[outer_center]],
                        dtype=np.float32
                    )


                    outer_world = (
                        cv2.perspectiveTransform(
                            outer_pixel,
                            H
                        )
                    )


                    outer_x = (
                        outer_world[0][0][0]
                    )

                    outer_y = (
                        outer_world[0][0][1]
                    )


                    dx = (
                        middle_x -
                        outer_x
                    )

                    dy = (
                        middle_y -
                        outer_y
                    )


                    distance_mm = np.sqrt(
                        dx * dx +
                        dy * dy
                    )



                    draw_distance_line(
                        frame,
                        middle,
                        outer_center,
                        distance_mm
                    )

    cv2.putText(
        frame,
        f"Camera: {width} x {height}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


cap.release()

cv2.destroyAllWindows()
