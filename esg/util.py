import logging
import os
import configparser
from pathlib import Path
import json
from typing import List
import hashlib


class Util:
    current_path = Path(os.path.split(os.path.realpath(__file__))[0] + '/../')
    logging.info("Current Path: {}".format(current_path))
    config = configparser.ConfigParser()

    @classmethod
    def if_not_exist_then_creat(cls, path):
        if not os.path.exists(path):
            logging.info("Path not exist!!! : {}, creating...".format(path))
            os.makedirs(path)

    @classmethod
    def get_current_path(cls):
        return cls.current_path

    @classmethod
    def get_data_path(cls):
        return cls.current_path / 'data'


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
    def compute_are_of_rectangle(r: list[float]):
        area = (r[2] - r[0]) * (r[3] - r[1])
        return area

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

    @staticmethod
    def str_to_bool(s):
        s = s.lower()
        if s in ['true', 't', 'yes', 'y', '1']:
            return True
        elif s in ['false', 'f', 'no', 'n', '0']:
            return False
        else:
            return None

    @staticmethod
    def compute_md5(data: str) -> str:
        md5 = hashlib.md5()
        md5.update(data.encode('utf-8'))
        return md5.hexdigest()
