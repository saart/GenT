import csv
import os
import shutil

from ml.app_normalizer import normalize_data
from ml.app_utils import GenTConfig


def test_normalize_data(tmp_path, test_data):
    source = tmp_path / "source"
    source.mkdir()

    shutil.copy2(os.path.join(test_data, "../test_data/app.json"), str(source))
    config = GenTConfig(
        chain_length=2, metadata_int_size=3, metadata_str_size=5, is_test=True
    )

    normalize_data(str(source), config)

    with open(
        os.path.join(config.get_raw_normalized_data_dir(), "app.csv"), "r"
    ) as csvfile:
        csv_data = csv.reader(csvfile)
        headers = next(csv_data)
        assert headers == [
            "traceId",
            "txStartTime",
            "chain",
            "gapFromParent_0",
            "duration_0",
            "hasError_0",
            "metadata_0_0",
            "metadata_0_1",
            "metadata_0_2",
            "metadata_0_3",
            "metadata_0_4",
            "metadata_0_5",
            "metadata_0_6",
            "metadata_0_7",
            "gapFromParent_1",
            "duration_1",
            "hasError_1",
            "metadata_1_0",
            "metadata_1_1",
            "metadata_1_2",
            "metadata_1_3",
            "metadata_1_4",
            "metadata_1_5",
            "metadata_1_6",
            "metadata_1_7",
        ]
        normalized_data = sorted(list(csv_data))
        assert normalized_data[0] == [
            "1958265259256010132023419797",
            "1675588335685",
            "eventBridge#wild-rydes-app-calcSalaries",
            "0",
            "175765",
            "0",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "0",
            "0",
            "0",
            "0",
            "129",
            "1",
            "nodejs12.x",
            "False",
            "AccessDeniedException",
            "Scheduled Event",
            "aws.events",
            "337",
            "1",
            "1024",
        ]
        assert len(normalized_data) == 6
