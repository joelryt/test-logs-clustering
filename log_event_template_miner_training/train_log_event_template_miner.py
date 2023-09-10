import os
import pandas as pd
from tqdm import tqdm
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence


def parse_templates_from_logs(directory: str, persistence_file: str):
    """Trains drain3 log event template miner model with all log lines
    from all .parquet file logs in the directory.

    :param directory: directory from which the all the parquet files are parsed
    :param persistence_file: persistence file path for the log template miner
    """

    def _parse_templates(df: pd.DataFrame):
        """Parses log event templates from a single log dataframe.

        :param df: pandas dataframe that contains the log lines that should be parsed
            by the log template miner on 'parsed_log_line' column
        """
        nonlocal template_miner
        for line in df["parsed_log_line"]:
            template_miner.add_log_message(line)

    config = TemplateMinerConfig()
    config.load(os.path.join("..", "log_template_miner_config", "drain3.ini"))
    persistence = FilePersistence(persistence_file)
    template_miner = TemplateMiner(persistence, config)
    num_of_files = len([name for name in os.listdir(directory) if name.endswith(".parquet")])
    with tqdm(total=num_of_files) as pbar:
        for filename in os.scandir(directory):
            if filename.path.endswith(".parquet"):
                df = pd.read_parquet(filename.path)
                _parse_templates(df)
            pbar.update()


def main():
    persistence_file_failed_cases = os.path.join("")
    persistence_file_passed_cases = os.path.join("")
    failed_cases_directory = os.path.join("")
    passed_cases_directory = os.path.join("")

    parse_templates_from_logs(failed_cases_directory, persistence_file_failed_cases)
    parse_templates_from_logs(passed_cases_directory, persistence_file_passed_cases)


if __name__ == "__main__":
    main()
