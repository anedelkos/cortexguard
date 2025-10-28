from pydantic import BaseModel


class Trial(BaseModel):
    trial_id: str
    subject: str | None = None
    food_type: str | None = None
    trial_num: int | None = None

    # file paths
    sensor_file: str
    image_folder: str
    depth_folder: str | None = None

    # processing options
    fusion_window: float = 0.03
    anomaly_scenario: str | None = None
    seed: int | None = None
