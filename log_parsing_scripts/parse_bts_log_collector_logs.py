import os
import re
import pandas as pd
from tqdm import tqdm


def parse_sw_component_and_severity_from_error_line(log_line: str) -> str:
    """Parses the SW component identifier and logging severity from the log line using regex.
    Expects to find the component identifier after a severity flag
    ('ERR/', 'WRN/', 'INF/', 'DBG/' or 'VIP/').

    :param log_line: log line to be parsed
    :return: returns the component identifier and severity as a tuple
    """
    regex = r"(ERR|WRN|INF|DBG|VIP)(/+\(?\w+[/\w:*]*\)?)"
    return re.findall(regex, log_line)[0]


def parse_timestamp_from_log_line(log_line: str) -> str:
    """Parses timestamp from the log line using regex.
    Expects to find the timestamp inside angle brackets (<>).

    :param log_line: log line to be parsed
    :return: returns the timestamp without the surrounding brackets as a string
    """
    regex = r"<(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z)>"
    return re.findall(regex, log_line)[0]


def parse_ip_address_from_log_line(log_line: str) -> str:
    """Parses IP address from the log line using regex.
    Expects to find the IP address at the start of the log line.

    :param log_line: log line to be parsed
    :return: returns the IP address as a string
    """
    regex = r"\A(([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|))"
    regex_match = re.findall(regex, log_line)[0]
    ip_address = regex_match[0] if isinstance(regex_match, tuple) else regex_match
    return ip_address


def parse_log_message_from_log_line(log_line: str, component: str) -> str:
    """Parses log message from the log line starting after the component identifier.

    :param log_line: log line to be parsed
    :param component: component identifier after which the log line is split
    :return: returns the log message as a string
    """
    return log_line.split(component)[1]


def extract_log_lines(df: pd.DataFrame, extracted_keywords: list) -> pd.DataFrame:
    """Extracts interesting log lines from the dataframe.
    Extracted lines are lines containing a keyword in the extracted_keywords list.

    :param df: dataframe from which the lines are extracted
    :param extracted_keywords: list of keywords that are searched from log lines,
        log lines that contain a keyword from this list are extracted
    :return: returns a dataframe containing only the extracted rows
    """

    def _append_extracted_line(line: str):
        """Appends the parsed parts of the extracted log line to the corresponding lists.

        :param line: the log line that is being parsed
        """
        result_ids.append(result_id)
        timestamps.append(parse_timestamp_from_log_line(line))
        ip_addresses.append(ip_address)
        severity, component = parse_sw_component_and_severity_from_error_line(line)
        components.append(component)
        severities.append(severity)
        parsed_log_lines.append(parse_log_message_from_log_line(line, component))
        log_lines.append(line)

    result_ids = []
    result_id = df.iloc[0]["resultId"]
    ip_addresses = []
    timestamps = []
    components = []
    severities = []
    parsed_log_lines = []
    log_lines = []

    for line in df["log_line"]:
        if any(keyword in line for keyword in extracted_keywords):
            ip_address = parse_ip_address_from_log_line(line)
            if "\r\n" in line:
                split_log_lines = line.split("\r\n")
                for split_line in split_log_lines:
                    if any(keyword in split_line for keyword in extracted_keywords):
                        _append_extracted_line(split_line)
            else:
                _append_extracted_line(line)

    extracted_df = pd.DataFrame(
        {
            "resultId": result_ids,
            "ip_address": ip_addresses,
            "timestamp": timestamps,
            "logging_entity": components,
            "severity": severities,
            "parsed_log_line": parsed_log_lines,
            "raw_log": log_lines,
        }
    )
    return extracted_df


def parse_bts_log_collector_logs(directory: str, save_directory: str, extracted_keywords: list = ["ERR/"]):
    """Parses all the bts log collector logs in the directory.
    Saves the parsed logs in the save directory with the same name as the original log.

    :param directory: directory from which the logs are parsed
    :param save_directory: directory where the parsed logs are saved
    :param extracted_keywords: list of keywords that are searched from log lines,
        log lines that contain a keyword from this list are extracted
    """
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    num_of_files = len([name for name in os.listdir(directory) if name.endswith(".parquet")])
    with tqdm(total=num_of_files) as pbar:
        for filename in os.scandir(directory):
            if filename.path.endswith(".parquet"):
                save_path = os.path.join(save_directory, os.path.basename(filename.path))
                if os.path.exists(save_path):
                    pbar.update()
                else:
                    df = pd.read_parquet(filename.path)
                    try:
                        parsed_df = extract_log_lines(df, extracted_keywords)
                    except Exception as e:
                        print(f"Failed to parse file: {os.path.basename(filename)}")
                        print(e)
                    else:
                        parsed_df.to_parquet(save_path)
                    pbar.update()


def main():
    directory = os.path.join("")
    save_directory = os.path.join("")
    parse_bts_log_collector_logs(directory, save_directory)


if __name__ == "__main__":
    main()
