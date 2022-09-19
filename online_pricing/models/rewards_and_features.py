from pydantic import BaseModel


class RewardsAndFeatures(BaseModel):
    """The rewards and features of a user."""

    reward: int
    features: list[int]
