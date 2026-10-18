"""Run from this directory; defaults to dry-run, never opens hardware."""
from organoid_arm import ArmController

with ArmController(dry_run=True) as arm:
    arm.home()
    arm.execute("right")
    arm.execute("up")
    arm.execute("grab")
    arm.execute("cw")
    arm.execute("release")
