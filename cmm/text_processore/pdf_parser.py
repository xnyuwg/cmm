from typing import List, Dict, Tuple
from cmm.conf.PDF_parser_conf import PDFParserConfig
from cmm.data_example.PDF_example import PDFBlockExample


class PDFParser():
    def __init__(self, config: PDFParserConfig):
        self.config = config

    def get_report_toc_as_example(self, report, toc) -> Tuple[List[List[PDFBlockExample]], List[list]]:
        report = self.convert_json_to_block_example(report)
        if toc is not None:
            toc = self.convert_json_to_toc_example(toc)
        return report, toc

    def convert_json_to_block_example(self, report) -> List[List[PDFBlockExample]]:
        new_report = []
        for page in report:
            new_page = []
            for block in page:
                new_block = PDFBlockExample()
                new_block.from_json(block)
                new_page.append(new_block)
            new_report.append(new_page)
        return new_report

    def convert_json_to_toc_example(self, report) -> List[list]:
        new_tocs = []
        for toc in report:
            new_tocs.append([toc['level'], toc['heading'], toc['page'], toc['block_id']])
        return new_tocs