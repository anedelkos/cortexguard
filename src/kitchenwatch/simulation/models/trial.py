from pydantic import BaseModel


class Trial(BaseModel):
    trial_id: str
    rgb_path: str
    depth_path: str | None
    timestamp_ns: int
    force_x: float
    force_y: float
    force_z: float
    torque_x: float
    torque_y: float
    torque_z: float
    pos_x: float
    pos_y: float
    pos_z: float
