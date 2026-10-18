"""AM1 inverse kinematics extracted from the competition controller."""

import math


def solve(x, y, z):
    """Return PWM for joints 0..3; coordinates are millimetres."""
    if not all(math.isfinite(v) for v in (x, y, z)) or y <= 0:
        raise ValueError("Coordinates must be finite and y must be positive")
    best = None
    for alpha in range(0, -136, -1):
        # Preserve the original AM1 base calibration (270/pi, not 180/pi).
        base = math.atan(x / y) * 270 / math.pi
        planar = math.hypot(x, y) * 10 - 1550 * math.cos(math.radians(alpha))
        height = z * 10 - 1000 - 1550 * math.sin(math.radians(alpha))
        radius = math.hypot(planar, height)
        if height < -1000 or radius < 1e-9 or radius > 1930:
            continue
        shoulder_cos = (radius**2 + 1050**2 - 880**2) / (2 * 1050 * radius)
        elbow_cos = -(radius**2 - 1050**2 - 880**2) / (2 * 1050 * 880)
        if not (-1 <= shoulder_cos <= 1 and -1 <= elbow_cos <= 1):
            continue
        elevation = math.acos(max(-1, min(1, planar / radius)))
        shoulder = math.degrees(elevation * (-1 if height < 0 else 1)
                                + math.acos(shoulder_cos))
        elbow = 180 - math.degrees(math.acos(elbow_cos))
        wrist = alpha - shoulder + elbow
        if not (0 <= shoulder <= 180 and -135 <= elbow <= 135 and -90 <= wrist <= 90):
            continue
        pwm = (int(1500 - 2000 * base / 270),
               int(1500 + 2000 * (shoulder - 90) / 270),
               int(1500 + 2000 * elbow / 270),
               3000 - int(1500 - 2000 * wrist / 270))
        if all(500 <= value <= 2500 for value in pwm):
            best = pwm
    if best is None:
        raise ValueError("Target pose is unreachable")
    return best
