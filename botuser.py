
from datetime import datetime

from pydantic.errors import StrError
from pyobjectid import PyObjectId
from bson import ObjectId
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from configurations import logger


class BotUser(BaseModel):  # class names in CamelCase

    # alias the _id of mongodb to id
    dbid: Optional[PyObjectId] = Field(alias='_id')
    id: int
    first_name: str
    is_bot: bool
    username: str
    language_code: str
    viewedurls: Optional[List[str]]
    authorised: Optional[bool]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {   # mapping for json serialization
            ObjectId: str
        }

    def set_authorised(self, authorised: bool):
        self.authorised = authorised
        return self

    def set_viewedurls(self, viewedurls: list):
        self.viewedurls = viewedurls
        return self

    def add_viewedurl(self, viewedurl: str):
        self.viewedurls.extend(viewedurl)
        return self
