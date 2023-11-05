from esg.ESG_PDF_downloader import ESGPDFDownloader
from esg.ESG_dataset_parser import ESGdatasetParser
from esg.util import Util
import argparse
import importlib
import logging
importlib.reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


def run(args):
    downloader = ESGPDFDownloader()
    parser = ESGdatasetParser()
    def process_fun(se):
        return parser.combine_one(se)
    if args.only_download:
        process_fun = None
    downloader.download_all(skip_downloaded_file=args.skip_downloaded_file, process_fun=process_fun)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--skip_downloaded_file", type=str, required=False, default='True', help="whether to skip downloaded file")
    arg_parser.add_argument("--only_download", type=str, required=False, default='False', help="whether to only download PDF")
    args = arg_parser.parse_args()
    args.skip_downloaded_file = Util.str_to_bool(args.skip_downloaded_file)
    args.only_download = Util.str_to_bool(args.only_download)
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)
