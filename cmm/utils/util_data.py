import logging
from typing import List
import json


class UtilData:
    def __init__(self):
        pass

    @staticmethod
    def read_raw_jsonl_file(file_name, verbose=True) -> List[dict]:
        if verbose:
            logging.info("reading jsonl from: {}".format(file_name))
        with open(file_name, 'r', encoding='utf-8') as file:
            file_content = [json.loads(line) for line in file]
        return file_content

    @staticmethod
    def read_raw_json_file(file_name, verbose=True) -> dict:
        if verbose:
            logging.info("reading json from: {}".format(file_name))
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    @staticmethod
    def write_json_file(file_name, data, verbose=True):
        if verbose:
            logging.info("writing json to: {}".format(file_name))
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    @staticmethod
    def decimal_to_rgb(decimal):
        hexadecimal = "{:06x}".format(decimal)
        r = int(hexadecimal[0:2], 16)
        g = int(hexadecimal[2:4], 16)
        b = int(hexadecimal[4:6], 16)
        return r, g, b

    @staticmethod
    def write_jsonl_file_line_error_catching(file_name, data, default_line=None, verbose=True):
        # if default_line is None, then ignore
        if verbose:
            logging.info("writing jsonl to: {}".format(file_name))
        with open(file_name, 'w', encoding='utf-8') as file:
            count = 0
            for line in data:
                try:
                    file.write(json.dumps(line, ensure_ascii=False))
                except UnicodeEncodeError:
                    logging.error('UnicodeEncodeError at file {} at line {}'.format(file_name, count))
                    try:
                        file.write(json.dumps(line, ensure_ascii=False).encode('utf-8', 'surrogateescape').decode('utf-8', 'replace'))
                    except UnicodeEncodeError:
                        logging.error('UnicodeEncodeError again with utf8-replace at file {} at line {}'.format(file_name, count))
                        if default_line is not None:
                            file.write(json.dumps(default_line, ensure_ascii=False))
                            logging.error('Write default {} at file {} at line {}'.format(default_line, file_name, count))
                        else:
                            logging.error('Ignore file {} at line {}'.format(file_name, count))
                            count -= 1
                file.write('\n')
                count += 1
