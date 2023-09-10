import os
import string
import re
import lzma
import pandas as pd
from tqdm import tqdm


def parse_log_line(log_line: str, severity: str):
    """Parses datetime, logging entity and log message from the log line and removes non-ASCII characters.

    :param log_line: String that needs to be parsed
    :param severity: Severity of the log message as a string,
        this is needed to split the log message after the severity information
    :return: Returns the parsed timestamp, logging entity and parsed log message as a tuple
    """
    # Remove non-ASCII characters and other extra stuff that are inside some of the container logs
    printable_chars = [char for char in log_line if char in string.printable]
    printable_line = "".join(printable_chars)
    regex = re.compile(r"\[\d+m")
    cleaned_log = regex.sub("", printable_line)

    # Parse timestamp
    regex = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{9}Z)"
    timestamp, log_msg = re.split(regex, cleaned_log, maxsplit=1)[1:]

    # Remove timestamps inside the log message
    timestamp_in_log_line_regex = re.compile(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}")
    log_msg = timestamp_in_log_line_regex.sub("", log_msg)

    # Separate log message and logging entity, if entity part is in the log message
    log_msg = log_msg.split(severity, maxsplit=1)[-1].strip()
    if log_msg[0] == "|":
        entity_regex = r"\|[ ]{0,1}(\w+)[ ]{0,1}\|"
        entity, log_msg = re.split(entity_regex, log_msg)[-2:]
    else:
        entity = None

    return timestamp, entity, log_msg.strip()


def parse_container_log(filename: str, save_path: str, extracted_severities: list):
    """Parses all the lines that are of specified logging severities (e.g., 'ERROR')
    from a compressed container log file into a parquet file with timestamps, severities
    and log messages parsed into separate columns.

    :param filename: Path of the file that needs to be parsed
    :param save_path: Path where the parsed parquet file is saved
    :param extracted_severities: List of logging severities that should be extracted
    """
    with lzma.open(filename, mode="rt", encoding="utf-8") as file:
        times = []
        severities = []
        logging_entities = []
        log_messages = []
        raw_logs = []
        try:
            for line in file:
                extracted_severity = []
                try:
                    extracted_severity = [severity for severity in extracted_severities if severity in line][0]
                except IndexError:
                    continue
                try:
                    time, logging_entity, log_message = parse_log_line(line, extracted_severity)
                    times.append(time)
                    severities.append(extracted_severity)
                    logging_entities.append(logging_entity)
                    log_messages.append(log_message)
                    raw_logs.append(line)
                except ValueError as exc:
                    raise ValueError(f"Failed to parse file: {filename}, line\n{line}.") from exc
        except UnicodeError as error:
            print(f"Failed to parse file {filename}. Error:\n{error}")
        else:
            df = pd.DataFrame(
                {
                    "timestamp": times,
                    "severity": severities,
                    "logging_entity": logging_entities,
                    "parsed_log_line": log_messages,
                    "raw_log": raw_logs,
                }
            )
            new_filename = f"{os.path.split(filename)[-1].split('.')[0]}.parquet"
            df.to_parquet(os.path.join(save_path, new_filename))


def parse_container_logs(directory: str, save_directory: str, extracted_severities: list = ["ERROR"]):
    """Iterates through all the .xz files in the directory and parses them
    into parquet files in save_directory.

    :param directory: Directory from which all the .xz files are parsed
    :param save_directory: Directory where the parsed parquet files are saved
    :param extracted_severities: List of logging severities that should be extracted from the logs,
        by default only extracting ERROR lines
    """
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    print(f"Parsing container logs from {directory}...")
    num_of_files = len([name for name in os.listdir(directory) if name.endswith(".xz")])
    with tqdm(total=num_of_files) as pbar:
        for filename in os.scandir(directory):
            if filename.path.endswith(".xz"):
                parse_container_log(filename.path, save_directory, extracted_severities)
                pbar.update()


def main():
    failed_cases_directory = os.path.join("")
    failed_cases_save_directory = os.path.join("")

    passed_cases_directory = os.path.join("")
    passed_cases_save_directory = os.path.join("")

    severities = ["CRITICAL", "ERROR"]
    parse_container_logs(failed_cases_directory, failed_cases_save_directory, severities)
    parse_container_logs(passed_cases_directory, passed_cases_save_directory, severities)


if __name__ == "__main__":
    main()
