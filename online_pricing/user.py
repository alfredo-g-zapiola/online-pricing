from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic.class_validators import root_validator


class User(BaseModel):

    id: int = Field(default_factory=uuid4, description="The user's id.")
    group: int = Field(..., description="The user's group.")
    landing_product: int = Field(..., description="The user's landing product.")

    feature_0: bool = Field(0, description="The user's feature 0.")
    feature_1: bool = Field(0, description="The user's feature 1.")

    @root_validator()
    def set_features(cls, values: dict[str, Any]) -> dict[str, Any]:
        features = [int(feature) for feature in "{0:02b}".format(values["group"])]
        values.update({"feature_0": features[0], "feature_1": features[1]})
        return values
