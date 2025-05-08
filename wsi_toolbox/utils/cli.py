import os
import sys
import re
from string import capwords
import inspect
import asyncio
from typing import Callable, Type
import argparse

from pydantic import BaseModel, Field
from pydantic_autocli import AutoCLI

from .seed import fix_global_seed, get_global_seed


class BaseMLArgs(BaseModel):
    seed: int = get_global_seed()

class BaseMLCLI(AutoCLI):
    class CommonArgs(BaseMLArgs):
        pass

    def _pre_common(self, a:BaseMLArgs):
        fix_global_seed(self.a.seed)
        super()._pre_common(a)
