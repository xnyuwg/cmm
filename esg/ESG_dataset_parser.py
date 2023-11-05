import logging
import fitz
from esg.util import Util
import re
from tqdm import tqdm


class ESGdatasetParser:
    def __init__(self,
                 ):
        self.report_summary_file = 'document_info.jsonl'
        self.pdf_folder = Util.get_data_path() / 'pdf'
        self.report_folder = Util.get_data_path() / 'report_info'
        self.new_report_folder = Util.get_data_path() / 'report_full'
        self.toc_folder = Util.get_data_path() / 'toc'
        self.summary_examples = Util.read_raw_jsonl_file(Util.get_data_path() / self.report_summary_file, verbose=False)
        Util.if_not_exist_then_creat(self.new_report_folder)

    def generate_dataset(self):
        need_to_check_pdf = []
        for se in tqdm(self.summary_examples):
            success, ratio = self.combine_one(se)
            if not success:
                need_to_check_pdf.append(str(se['doc_id']) + f"({ratio})")
        if len(need_to_check_pdf) > 0:
            logging.error('The following PDF need to check:')
            for x in need_to_check_pdf:
                logging.error(str(x))

    def get_span_key_with_span_block(self, block, no_color=False):
        return self.get_span_key_with_para(block['page_count'], block['size'], block['font'], block['color'], block['span_bbox'], no_color=no_color)

    def get_span_key_with_para(self, page_count, size, font, color, box, no_color=False):
        key = str(page_count) + '_' + str(font) + '_' + str(int(color)) + '_'
        if no_color:
            key = str(page_count) + '_' + str(font) + '_'
        box = box
        box = [str(int(x)) for x in box]
        pos = '_'.join(box)
        key += pos
        return key

    def combine_one(self, summary_example):
        doc_id = summary_example['doc_id']
        toc_page = summary_example['toc_page']
        # logging.info('processing {}'.format(doc_id))
        pdf_path = self.pdf_folder / (doc_id + '.pdf')
        report_path = self.report_folder / (doc_id + '.jsonl')
        toc_path = self.toc_folder / (doc_id + '.jsonl')
        report_jsonl = Util.read_raw_jsonl_file(report_path, verbose=False)
        toc_jsonl = Util.read_raw_jsonl_file(toc_path, verbose=False)
        raw_text, pdf_text = self.pdf_read(pdf_path=pdf_path)
        if pdf_text is None:
            return False

        ids_to_toc = {}
        for toc in toc_jsonl:
            ids_to_toc['block_id'] = toc

        brlc_to_pdf_block = {}
        brlc_to_pdf_block_no_color = {}
        for page in pdf_text:
            for block in page:
                key = self.get_span_key_with_span_block(block)
                if key in brlc_to_pdf_block and re.sub(' +', '', block['text'].strip()) != re.sub(' +', '', brlc_to_pdf_block[key]['text'].strip()) and len(re.sub(' +', '', brlc_to_pdf_block[key]['text'].strip())) > 4:
                    logging.error('PDF key in brlc_to_pdf_block with block_key: {} and block_in {}'.format(block, brlc_to_pdf_block[key]))
                    pass
                else:
                    brlc_to_pdf_block[key] = block
                    brlc_to_pdf_block_no_color[self.get_span_key_with_span_block(block, no_color=True)] = block
        fake_root = {"id": "r0", "text_str": "root", "text": "root", "texts": ["root"], "size": 100.0, "font": "", "color": 0, "span_bbox": [0, 1, 0, 1], "page_count": -1, "xy_cut_sequence_number": -1, "page_height": 1, "page_width": 1, "block_no": -1}
        brlc_to_pdf_block[self.get_span_key_with_span_block(fake_root)] = fake_root
        brlc_to_pdf_block_no_color[self.get_span_key_with_span_block(fake_root, no_color=True)] = fake_root

        page_mapping = [i for i in range(len(pdf_text)) if i not in toc_page]
        pos_find_scope = 5

        for page in report_jsonl:
            for block in page:
                if block['id'] in ids_to_toc:
                    toc = ids_to_toc[block['id']]
                    block['texts'] = [toc['heading']]
                    block['text'] = toc['heading']
                    continue
                texts = []
                for pos in block['bboxes']:
                    find_match = False
                    page = block['page']
                    if page >= len(page_mapping):
                        return False, 0
                    page = page_mapping[page]
                    key = self.get_span_key_with_para(page, block['size'], block['font'], block['color'], pos)
                    if key in brlc_to_pdf_block:
                        text = brlc_to_pdf_block[key]['text']
                        texts.append(text)
                        find_match = True
                    if find_match:
                        continue
                    key = self.get_span_key_with_para(page, block['size'], block['font'], block['color'], pos, no_color=True)
                    if key in brlc_to_pdf_block_no_color:
                        text = brlc_to_pdf_block_no_color[key]['text']
                        texts.append(text)
                        find_match = True
                    if find_match:
                        continue
                    last_try_pos_list = [[]]
                    for i, p in enumerate(pos):
                        new_try_pos_list = []
                        for ltpos in last_try_pos_list:
                            for m in range(0, pos_find_scope * 2 + 1):
                                new_try_pos_list.append(ltpos + [p + m - pos_find_scope])
                        last_try_pos_list = new_try_pos_list
                    for try_pos in last_try_pos_list:
                        assert len(try_pos) == 4
                        key = self.get_span_key_with_para(page, block['size'], block['font'], block['color'], try_pos, no_color=True)
                        if key in brlc_to_pdf_block_no_color:
                            text = brlc_to_pdf_block_no_color[key]['text']
                            texts.append(text)
                            find_match = True
                    # logging.error('Map key not in brlc_to_pdf_block for block {}'.format(block))
                    pass
                block['texts'] = texts

        report_jsonl = self.text_list_to_str_clean(report_jsonl)

        total_block_count = 0
        text_map_count = 0
        for page in report_jsonl:
            for block in page:
                # if 'check_text' in block and block:
                total_block_count += 1
                if block['check_text'] == Util.compute_md5(str(block['text'])):
                    text_map_count += 1
                else:
                    # logging.error(f"check_text {block['check_text']} != text |{str(block['text'])}| with md5 |{Util.compute_md5(str(block['text']))}|")
                    logging.error(f"check_text != text |{(block)}| with md5 |{Util.compute_md5(str(block['text']))}|")
                    pass
        if total_block_count > 0 and text_map_count / total_block_count < 0.9:
            logging.warning('text map ratio {} for {}'.format(text_map_count/total_block_count, doc_id))
        elif total_block_count > 0 and text_map_count / total_block_count < 0.99:
            logging.info('text map ratio {} for {}'.format(text_map_count / total_block_count, doc_id))
        if total_block_count > 0 and text_map_count / total_block_count < 0.5:
            success = False
        else:
            Util.write_jsonl_file_line_error_catching(self.new_report_folder / (doc_id + '.jsonl'), report_jsonl, default_line=[], verbose=False)
            success = True
        return success, text_map_count / total_block_count

    def pdf_read(self, pdf_path):
        report = self.extract_raw_text_from_pdf(pdf_path=pdf_path)
        if report is None:
            return None, None
        raw_text = report
        report = self.extract_text_from_raw_text(report)
        return raw_text, report

    def extract_raw_text_from_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            read_pages = []
            for page in doc:
                text = page.get_text('dict')
                read_pages.append(text)
            doc.close()
        except Exception as e:
            logging.error('Error on {}: {}'.format(pdf_path, e))
            return None

        image_count = -1
        page_count = -1
        block_report_level_count = -1
        for read_page in read_pages:
            page_count += 1
            read_page['page_count'] = page_count
            blocks = read_page['blocks']
            block_page_level_count = -1
            for block in blocks:
                block_page_level_count += 1
                block_report_level_count += 1
                if block['type'] == 1:
                    image_count += 1
                    block['image'] = str(image_count)
                if block['type'] != 1 and block['type'] != 0:
                    logging.warning("type is not 1 and 2:")
                    logging.warning(block)
                block['page_count'] = page_count
                block['block_page_level_count'] = block_page_level_count
                block['block_report_level_count'] = block_report_level_count
        return read_pages

    def extract_text_from_raw_text(self, read_pages):
        out_pages = []
        page_count = -1
        block_report_level_count = -1
        for read_page in read_pages:
            page_count += 1
            if 'blocks' not in read_page:
                out_pages.append([])
                continue
            blocks = read_page['blocks']

            out_blocks = []
            block_page_level_count = -1

            for i, block in enumerate(blocks):
                block_type = block['type']
                if block_type == '1' or block_type == 1:
                    continue
                if 'lines' not in block:
                    continue
                lines = block['lines']
                block_number = block['number']
                for line in lines:
                    for spans in line['spans']:
                        if isinstance(spans, dict):
                            spans = [spans]
                        for span in spans:
                            span_bbox = span['bbox']
                            block_page_level_count += 1
                            block_report_level_count += 1
                            out_block = dict(
                                page_count=page_count,
                                block_page_level_count=block_page_level_count,
                                block_report_level_count=block_report_level_count,
                                block_number=block_number,
                                text=span['text'],
                                type='text',
                                size=span['size'],
                                font=span['font'],
                                color=span['color'],
                                span_bbox=span_bbox
                            )
                            out_blocks.append(out_block)
            out_pages.append(out_blocks)
        return out_pages

    def text_list_to_str_clean(self, report):
        for page in report:
            for block in page:
                texts = block['texts']
                text_str = ' '.join(texts)
                text_str = str(text_str)
                if ' ' in text_str:
                    text_str = text_str.replace(' ', ' ')
                text_str = re.sub(' +', ' ', text_str)
                if '-' in text_str:
                    text_str = text_str.replace('- ', '').replace(' -', '').replace(' ,', ',').replace(' .', '.')
                text_str = text_str.strip()
                block['text'] = text_str
        return report

