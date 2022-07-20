import os
from unittest import TestCase
import shutil
import pandas as pd
import pytest
from featureform import serving, ResourceClient
from tempfile import NamedTemporaryFile
import csv
from localmode_cases import features_no_ts, features_with_ts, feature_invalid_entity, feature_invalid_value, \
    feature_invalid_ts, feature_e2e


class TestIndividualFeatures(TestCase):
    def test_process_feature_no_ts(self):
        for name, case in features_no_ts.items():
            with self.subTest(name):
                file_name = create_temp_file(case)
                client = serving.LocalClient()
                dataframe_mapping = client.process_feature_csv(file_name, case['entity'], case['entity'], case['value_col'], [], "test_name_variant", "")
                expected = pd.DataFrame(case['expected'])
                actual = dataframe_mapping[0]
                expected = expected.values.tolist()
                actual = actual.values.tolist()
                assert all(elem in expected for elem in actual), \
                    "Expected: {} Got: {}".format(expected, actual)

    def test_process_feature_with_ts(self):
        for name, case in features_with_ts.items():
            with self.subTest(msg=name):
                file_name = create_temp_file(case)
                client = serving.LocalClient()
                dataframe_mapping = client.process_feature_csv(file_name, case['entity'], case['entity'], case['value_col'], [], "test_name_variant", case['ts_col'])
                expected = pd.DataFrame(case['expected'])
                actual = dataframe_mapping[0]
                expected = expected.values.tolist()
                actual = actual.values.tolist()
                assert all(elem in expected for elem in actual), \
                    "Expected: {} Got: {}".format(expected, actual)

    def test_invalid_entity_col(self):
        case = feature_invalid_entity
        file_name = create_temp_file(case)
        client = serving.LocalClient()
        with pytest.raises(KeyError) as err:
            client.process_feature_csv(file_name, case['entity'], case['value_col'], case['name'], [], "test_name_variant", case['ts_col'])
        assert "column does not exist" in str(err.value)

    def test_invalid_value_col(self):
        case = feature_invalid_value
        file_name = create_temp_file(case)
        client = serving.LocalClient()
        with pytest.raises(KeyError) as err:
            client.process_feature_csv(file_name, case['entity'], case['value_col'], case['name'], [], "test_name_variant", case['ts_col'])
        assert "column does not exist" in str(err.value)

    def test_invalid_ts_col(self):
        case = feature_invalid_ts
        file_name = create_temp_file(case)
        client = serving.LocalClient()
        with pytest.raises(KeyError) as err:
            client.process_feature_csv(file_name, case['entity'], case['value_col'], case['name'], [], "test_name_variant", case['ts_col'])
        assert "column does not exist" in str(err.value)


class TestFeaturesE2E(TestCase):
    def test_features(self):
        for name, case in feature_e2e.items():
            with self.subTest(msg=name):
                file_name = create_temp_file(case)
                res = e2e_features(file_name, case['entity'], case['entity_loc'], case['features'], case['value_cols'], case['entities'], case['ts_col'])
                expected = case['expected']
                assert all(elem in expected for elem in res), \
                    "Expected: {} Got: {}".format(expected, res)
            shutil.rmtree('.featureform')



    def test_timestamp_doesnt_exist(self):
        case = {
            'columns': ['entity', 'value'],
            'values': [
                ['a', 1],
                ['b', 2],
                ['c', 3],

            ],
            'value_cols': ['value'],
            'entity': 'user',
            'entity_loc': 'entity',
            'features': [("avg_transactions", "quickstart")],
            'entities': [("user", "a"), ("user", "b"), ("user", "c")],
            'expected': [[1], [2], [3]],
            'ts_col': "ts"
        }
        file_name = create_temp_file(case)
        with pytest.raises(KeyError) as err:
            e2e_features(file_name, case['entity'], case['entity_loc'], case['features'], case['value_cols'], case['entities'], case['ts_col'])
        assert "column does not exist" in str(err.value)



    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(tmpdir):
        """Fixture to execute asserts before and after a test is run"""
        # Remove any lingering Databases
        try:
            shutil.rmtree('.featureform')
        except:
            print("File Already Removed")
        yield
        try:
            shutil.rmtree('.featureform')
        except:
            print("File Already Removed")


def create_temp_file(test_values):
    file = NamedTemporaryFile(delete=False)
    with open(file.name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        writer.writerow(test_values['columns'])
        for row in test_values['values']:
            writer.writerow(row)

    return file.name


def e2e_features(file, entity_name, entity_loc, name_variants, value_cols, entities, ts_col):
    ff = ResourceClient("")
    ff.register_user("featureformer").make_default_owner()
    local = ff.register_local()
    transactions = local.register_file(
        name="transactions",
        variant="quickstart",
        description="A dataset of fraudulent transactions",
        path=file
    )
    entity = ff.register_entity(entity_name)
    for i, variant in enumerate(name_variants):
        transactions.register_resources(
            entity=entity,
            entity_column=entity_loc,
            inference_store=local,
            features=[
                {"name": variant[0], "variant": variant[1], "column": value_cols[i], "type": "float32"},
            ],
            timestamp_column=ts_col
        )
    ff.state().create_all_local()
    client = serving.LocalClient()
    results = []
    for entity in entities:
        results.append(client.features(name_variants, entity))
    return results
