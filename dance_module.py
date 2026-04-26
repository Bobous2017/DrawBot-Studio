import math
import os

import pygame


SKELETON_EDGES = [
    ("head", "neck"),
    ("neck", "shoulder_l"),
    ("neck", "shoulder_r"),
    ("shoulder_l", "elbow_l"),
    ("elbow_l", "hand_l"),
    ("shoulder_r", "elbow_r"),
    ("elbow_r", "hand_r"),
    ("neck", "hip_l"),
    ("neck", "hip_r"),
    ("hip_l", "hip_r"),
    ("hip_l", "knee_l"),
    ("knee_l", "foot_l"),
    ("hip_r", "knee_r"),
    ("knee_r", "foot_r"),
]


REQUIRED_KEYS = {
    "head",
    "neck",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "hand_l",
    "hand_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "foot_l",
    "foot_r",
}


DEFAULT_DANCE_PROFILE = {
    "bob_amp": 8.0,
    "bob_speed": 4.0,
    "sway_amp": 10.0,
    "sway_speed": 2.0,
    "arm_wave_amp": 20.0,
    "arm_wave_speed": 8.0,
    "leg_step_amp": 14.0,
    "leg_step_speed": 6.0,
}


def _validate_joints(joints):
    missing = REQUIRED_KEYS - set(joints.keys())
    if missing:
        raise ValueError(f"Missing joints: {sorted(missing)}")


def _normalize_to_screen(joints, width, height, margin=80):
    xs = [float(joints[name][0]) for name in REQUIRED_KEYS]
    ys = [float(joints[name][1]) for name in REQUIRED_KEYS]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    src_w = max(max_x - min_x, 1.0)
    src_h = max(max_y - min_y, 1.0)

    target_w = width - margin * 2
    target_h = height - margin * 2
    scale = min(target_w / src_w, target_h / src_h)

    offset_x = (width - src_w * scale) * 0.5
    offset_y = (height - src_h * scale) * 0.5

    normalized = {}
    for name, point in joints.items():
        x = (float(point[0]) - min_x) * scale + offset_x
        y = (float(point[1]) - min_y) * scale + offset_y
        normalized[name] = [x, y]

    return normalized


def _animate_pose(base_pose, t, profile):
    pose = {name: [point[0], point[1]] for name, point in base_pose.items()}

    bob = math.sin(t * profile["bob_speed"]) * profile["bob_amp"]
    sway = math.sin(t * profile["sway_speed"]) * profile["sway_amp"]
    wave_l = math.sin(t * profile["arm_wave_speed"]) * profile["arm_wave_amp"]
    wave_r = math.sin(t * profile["arm_wave_speed"] + math.pi) * profile["arm_wave_amp"]
    step_l = math.sin(t * profile["leg_step_speed"]) * profile["leg_step_amp"]
    step_r = math.sin(t * profile["leg_step_speed"] + math.pi) * profile["leg_step_amp"]

    for name in pose:
        pose[name][1] += bob

    pose["head"][0] += sway * 0.2
    pose["shoulder_l"][0] += sway * 0.3
    pose["shoulder_r"][0] += sway * 0.3
    pose["hip_l"][0] -= sway * 0.15
    pose["hip_r"][0] -= sway * 0.15

    pose["elbow_l"][1] += wave_l * 0.6
    pose["hand_l"][1] += wave_l
    pose["hand_l"][0] += wave_l * 0.2

    pose["elbow_r"][1] += wave_r * 0.6
    pose["hand_r"][1] += wave_r
    pose["hand_r"][0] += wave_r * 0.2

    pose["knee_l"][1] += step_l * 0.5
    pose["foot_l"][1] += step_l
    pose["foot_l"][0] += step_l * 0.15

    pose["knee_r"][1] += step_r * 0.5
    pose["foot_r"][1] += step_r
    pose["foot_r"][0] += step_r * 0.15

    return pose


def _draw_pose(screen, pose):
    edge_color = (50, 90, 255)
    joint_color = (255, 120, 60)

    for a, b in SKELETON_EDGES:
        ax, ay = pose[a]
        bx, by = pose[b]
        pygame.draw.line(screen, edge_color, (ax, ay), (bx, by), 4)

    for point in pose.values():
        pygame.draw.circle(screen, joint_color, (int(point[0]), int(point[1])), 7)


def _pose_center(pose):
    hx = (pose["hip_l"][0] + pose["hip_r"][0]) * 0.5
    hy = (pose["hip_l"][1] + pose["hip_r"][1]) * 0.5
    nx = pose["neck"][0]
    ny = pose["neck"][1]
    return ((hx + nx) * 0.5, (hy + ny) * 0.5)


def _pose_torso_angle(pose):
    hx = (pose["hip_l"][0] + pose["hip_r"][0]) * 0.5
    hy = (pose["hip_l"][1] + pose["hip_r"][1]) * 0.5
    nx = pose["neck"][0]
    ny = pose["neck"][1]
    return math.atan2(ny - hy, nx - hx)


def _pose_scale_metric(pose):
    shoulder = math.hypot(
        pose["shoulder_r"][0] - pose["shoulder_l"][0],
        pose["shoulder_r"][1] - pose["shoulder_l"][1],
    )
    torso = math.hypot(
        pose["neck"][0] - ((pose["hip_l"][0] + pose["hip_r"][0]) * 0.5),
        pose["neck"][1] - ((pose["hip_l"][1] + pose["hip_r"][1]) * 0.5),
    )
    return max((shoulder + torso) * 0.5, 1.0)


def _load_figure_sprite(figure_image_path):
    if not figure_image_path:
        return None
    if not os.path.exists(figure_image_path):
        return None

    loaded = pygame.image.load(figure_image_path).convert_alpha()

    # Crop to the actual non-transparent figure bounds so the sprite is not a big rectangle.
    mask = pygame.mask.from_surface(loaded)
    bounds = mask.get_bounding_rects()
    if bounds:
        rect = bounds[0]
        for bound in bounds[1:]:
            rect.union_ip(bound)
        loaded = loaded.subsurface(rect).copy()

    return loaded


def start_dancing(
    joints,
    figure_image_path=None,
    width=900,
    height=900,
    duration_sec=20,
    fps=60,
    show_skeleton=False,
):
    _validate_joints(joints)
    base_pose = _normalize_to_screen(joints, width, height)
    profile = dict(DEFAULT_DANCE_PROFILE)

    pygame.init()
    pygame.display.set_caption("Robot Agent Dance")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    sprite = _load_figure_sprite(figure_image_path)
    base_angle = _pose_torso_angle(base_pose)
    base_scale = _pose_scale_metric(base_pose)

    running = True
    elapsed = 0.0

    while running:
        dt = clock.tick(fps) / 1000.0
        elapsed += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if duration_sec and elapsed >= duration_sec:
            running = False

        pose = _animate_pose(base_pose, elapsed, profile)

        screen.fill((245, 247, 255))

        if sprite is not None:
            center = _pose_center(pose)
            angle = _pose_torso_angle(pose)
            scale_metric = _pose_scale_metric(pose)

            rel_angle_deg = math.degrees(angle - base_angle)
            rel_scale = max(0.7, min(1.5, scale_metric / base_scale))

            transformed = pygame.transform.rotozoom(sprite, -rel_angle_deg, rel_scale)
            # Move sprite with torso center motion from the animated pose.
            draw_x = int(center[0])
            draw_y = int(center[1])
            rect = transformed.get_rect(center=(draw_x, draw_y))
            screen.blit(transformed, rect)

        if show_skeleton or sprite is None:
            _draw_pose(screen, pose)

        pygame.display.flip()

    pygame.quit()
