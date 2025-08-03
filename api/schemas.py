from pydantic import BaseModel, conlist, ValidationError


class IrisInput(BaseModel):
    features: conlist(float, min_length=4, max_length=4)
