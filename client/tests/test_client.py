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
from featureform.enums import ResourceType


class MockStub:
    req = None

    def AddTrigger(self, req):
        req = req

    def RemoveTrigger(self, req):
        req = req

    def UpdateTrigger(self, req):
        req = req

    def DeleteTrigger(self, req):
        req = req


def test_add_trigger():
    client = ff.Client()
    client._stub = MockStub()
    trigger = ff.ScheduleTrigger("trigger_name", "1 1 * * *")
    f = ("name", "variant", "FEATURE_VARIANT")
    client.add_trigger(trigger, f)


def test_create_trigger_proto_trigger():
    client = ff.Client()
    trigger = ff.ScheduleTrigger("trigger", "1 1 * * *")
    proto = client._create_trigger_proto(trigger)
    assert type(proto) == ff.metadata_pb2.Trigger
    assert proto.name == "trigger"


def test_create_trigger_proto_string():
    client = ff.Client()
    trigger_name = "trigger_name"
    proto = client._create_trigger_proto(trigger_name)
    assert type(proto) == ff.metadata_pb2.Trigger
    assert proto.name == "trigger_name"


def test_create_trigger_proto_error():
    client = ff.Client()
    not_a_trigger = ff.Feature(("name", "variant", "other_info"), ff.String)
    with pytest.raises(ValueError) as e:
        client._create_trigger_proto(not_a_trigger)
    assert (
        str(e.value)
        == f"Invalid trigger type: {type(not_a_trigger)}. Please use the trigger name or TriggerResource"
    )


def test_create_resource_proto_tuple():
    client = ff.Client()
    feature_id = ("name", "variant", "FEATURE_VARIANT")
    proto = client._create_resource_proto(feature_id)
    assert type(proto) == ff.metadata_pb2.ResourceID
    assert proto.resource.name == "name"
    assert proto.resource.variant == "variant"
    assert proto.resource_type == 4


class MockRegistrar:
    def __init__(self, *args, **kwargs):
        pass


@mock.patch("featureform.register.Registrar", mock.MagicMock(side_effect=MockRegistrar))
def test_create_resource_proto_feature():
    client = ff.Client()

    # Arguments
    class User:
        feature_obj = ff.Feature(
            (
                MockRegistrar(),
                ("source_name", "source_variant"),
                ["column1", "column2"],
            ),
            variant="variant",
            type=ff.String,
        )

    User.feature_obj.name = "feature_obj"

    proto = client._create_resource_proto(User.feature_obj)

    # Expected output
    assert type(proto) == ff.metadata_pb2.ResourceID
    assert proto.resource.name == "feature_obj"
    assert proto.resource.variant == "variant"
    assert proto.resource_type == ResourceType.FEATURE.value


@mock.patch("featureform.register.Registrar", mock.MagicMock(side_effect=MockRegistrar))
def test_create_resource_proto_label():
    client = ff.Client()

    # Arguments
    class User:
        label_obj = ff.Label(
            (
                MockRegistrar(),
                ("source_name", "source_variant"),
                ["column1", "column2"],
            ),
            variant="variant",
            type=ff.String,
        )

    User.label_obj.name = "label_obj"

    proto = client._create_resource_proto(User.label_obj)

    # Expected output
    assert type(proto) == ff.metadata_pb2.ResourceID
    assert proto.resource.name == "label_obj"
    assert proto.resource.variant == "variant"
    assert proto.resource_type == ResourceType.LABEL.value


class MockFeature:
    name = "feature_name"
    variant = "feature_variant"
    registrar = MockRegistrar()
    source = ("source_name", "source_variant")
    columns = ["column1", "column2"]


# class MockLabel:
#     # name = "label_name"
#     variant = "label_variant"
#     registrar = MockRegistrar()
#     source = ("source_name", "source_variant")

# @mock.patch(
#     "featureform.resources.FeatureVariant", mock.MagicMock(side_effect=MockFeature)
# )
# @mock.patch(
#     "featureform.register.LabelColumnResource", mock.MagicMock(side_effect=MockLabel)
# )
# @mock.patch(
#     "featureform.register.Registrar", mock.MagicMock(side_effect=MockRegistrar)
# )
# def test_create_resource_proto_trainingset():
#     client = ff.Client()

#     # Arguments
#     class User:
#         ts_obj = ff.register_training_set("ts_obj", "variant", features=[MockFeature()], label=("label_name", "label_variant"))

#     proto = client._create_resource_proto(User.ts_obj)

#     # Expected output
#     assert type(proto) == ff.metadata_pb2.ResourceID
#     assert proto.resource.name == "ts_obj"
#     assert proto.resource.variant == "variant"
#     assert proto.resource_type == ResourceType.TRAINING_SET.value
