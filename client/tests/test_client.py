import csv
import os
import shutil
import stat
import sys
import time
from tempfile import NamedTemporaryFile
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "client/src/")
from featureform import Client
import serving_cases as cases
import featureform as ff
from featureform.serving import check_feature_type, Row, Dataset

class MockStub:
    req = None
    def AddTrigger(self, req):
        req = req

def test_create_trigger_proto():
    client = ff.Client()
    client._stub = MockStub()
    trigger = ff.ScheduleTrigger("trigger_name", "1 1 * * *")
    f = ("name", "variant", ff.ResourceType.FEATURE.value)
    client.add_trigger(trigger, f)

