from esg.util import Util
import wget
import time
import random
import logging
import os
import fitz
from unidecode import unidecode


class ESGPDFDownloader:
    def __init__(self,
                 ):
        self.report_summary_file = 'document_info.jsonl'
        self.pdf_folder = Util.get_data_path() / 'pdf'
        self.summary_examples = Util.read_raw_jsonl_file(Util.get_data_path() / self.report_summary_file, verbose=False)
        Util.if_not_exist_then_creat(self.pdf_folder)

        self.url_archive_ticker_template = 'https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/{}/{}_{}_{}.pdf'
        self.url_archive_name_template = 'https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/{}/{}_{}.pdf'
        self.url_pdf_ticker_template = 'https://www.responsibilityreports.com/HostedData/ResponsibilityReports/PDF/{}_{}_{}.pdf'
        self.url_pdf_name_template = 'https://www.responsibilityreports.com/HostedData/ResponsibilityReports/PDF/{}_{}.pdf'

    def download_all(self, skip_downloaded_file=True, process_fun=None):
        skip_count = 0
        total_ratio = 0
        run_count = 0
        not_success = []
        for i, se in enumerate(self.summary_examples):
            path = self.pdf_folder / (se['doc_id'] + '.pdf')
            if skip_downloaded_file and os.path.exists(path):
                skip_count += 1
                continue
            else:
                if skip_count > 0:
                    logging.info('skipped {} PDF file'.format(skip_count))
                skip_count = 0
            success, errors, used_url, ratio = self.download_one_with_se(se=se, pdf_path=path, process_fun=process_fun)
            run_count += 1
            total_ratio += ratio
            if not success:
                not_success.append((se['pdf_link'], path, errors))
            else:
                logging.info(f"{i}-th PDF downloaded from {used_url} and saved at {path}")
        if len(not_success) > 0:
            logging.error(f"Error downloading the following files (total {len(not_success)}):")
            for ns in not_success:
                logging.error(ns)
        else:
            logging.info("All PDF files downloaded with matched text ratio {}".format(total_ratio / run_count))

    def download_one_with_se(self, se, pdf_path, max_retries=3, process_fun=None):
        urls = []
        if se['pdf_link'][-9] != '_' and '/Click/' not in se['pdf_link']:
            urls.append(se['pdf_link'])
        try_urls = self.url_generate(date=se['download_report_date'], ticker=se['ticker'], exchange=se['exchange'], company_name=se['company_name'])
        urls += try_urls
        if '/Click/' not in se['pdf_link']:
            urls.append(se['pdf_link'])
        ratio = 0
        url = None
        success = False
        error_infos = []
        retries = 0
        while retries < max_retries:
            for url in urls:
                sleep_duration = random.randint(3, 6)
                time.sleep(sleep_duration)
                try:
                    wget.download(url, str(pdf_path))
                    if process_fun is None:
                        doc = fitz.open(pdf_path)
                        for page in doc:
                            page.get_text('dict')
                        doc.close()
                    else:
                        process_success, ratio = process_fun(se)
                        if not process_success:
                            raise Exception(f'process not success {ratio}!')
                    success = True
                    break
                except Exception as e:
                    if retries > 0:
                        logging.warning(f"Error downloading {url} to save at {pdf_path}. Error: {e}")
                        logging.warning(f"Retrying... ({retries}/{max_retries})")
                    error_infos.append(f"URL={url}: Exception={e}")
                if not success:
                    if os.path.isfile(pdf_path):
                        try:
                            os.remove(pdf_path)
                            logging.info(f'File at {pdf_path} deleted.')
                        except Exception as e:
                            logging.error(f'Deleting {pdf_path} encounter the error: {e}')
            if success:
                break
            else:
                retries += 1
        else:
            logging.error(f"Failed to download {urls} to save at {pdf_path} after {max_retries} attempts.")
            success = False
        return success, error_infos, url, ratio

    def url_generate(self, date, ticker, exchange, company_name):
        has_company_name = company_name is not None and len(company_name) > 0 and company_name != 'null'
        has_ticker_exchange = ticker is not None and len(ticker) > 0 and ticker != 'null' and exchange is not None and len(exchange) > 0 and exchange != 'null'
        company_name = unidecode(company_name) if has_company_name else None
        company_name = company_name.lower().replace(' ', '-').replace('.', '') if has_company_name else None
        ticker = ticker.replace(':', '') if has_ticker_exchange else None
        if has_company_name and has_ticker_exchange:
            out = [
                self.url_archive_ticker_template.format(company_name[0], exchange, ticker, date),
                self.url_archive_name_template.format(company_name[0], company_name, date),
                self.url_pdf_ticker_template.format(exchange, ticker, date),
                self.url_pdf_name_template.format(company_name, date),
                self.url_archive_ticker_template.format(company_name[0].upper(), exchange, ticker, date),
                self.url_archive_name_template.format(company_name[0].upper(), company_name, date),
            ]
        elif has_company_name and not has_ticker_exchange:
            out = [
                self.url_archive_name_template.format(company_name[0], company_name, date),
                self.url_pdf_name_template.format(company_name, date),
                self.url_archive_name_template.format(company_name[0].upper(), company_name, date),
            ]
        elif not has_company_name and has_ticker_exchange:
            out = [
                self.url_archive_ticker_template.format(ticker[0].lower(), exchange, ticker, date),
                self.url_archive_ticker_template.format(ticker[0], exchange, ticker, date),
                self.url_pdf_ticker_template.format(exchange, ticker, date)
            ]
        else:
            out = []
        PDF_out = []
        for o in out:
            PDF_out.append(o[:-3] + 'PDF')
        final_out = out + PDF_out
        return final_out

