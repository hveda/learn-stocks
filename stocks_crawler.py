"""Yahoo finance stock data crawler."""

import requests                  # [handles the http interactions](http://docs.python-requests.org/en/master/)
from bs4 import BeautifulSoup    # beautiful soup handles the html to text conversion and more
import re                        # regular expressions are necessary for finding the crumb (more on crumbs later)
from datetime import datetime    # string to datetime object conversion
from time import mktime          # mktime transforms datetime objects to unix timestamps
import csv
import sys
import random
from lxml.html import fromstring

user_agent_list = [
    # Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    # Firefox
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)'
]


def get_proxies():
    """
    Get free proxies dictionary.

    parameters: none

    returns dictionary of proxies.
    """
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = []
    for i in parser.xpath('//tbody/tr')[:10]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.append(proxy)
    return proxies


def _get_crumbs_and_cookies(stock, proxy):
    """
    Get crumb and cookies for historical data csv download from yahoo finance.

    parameters: stock - short-handle identifier of the company

    returns a tuple of header, crumb and cookie
    """
    url = 'https://finance.yahoo.com/quote/{}/history'.format(stock)
    with requests.session():
        user_agent = random.choice(user_agent_list)
        header = {'Connection': 'keep-alive',
                  'Expires': '-1',
                  'Upgrade-Insecure-Requests': '1',
                  'User-Agent': user_agent
                  }

        website = requests.get(url, headers=header, proxies={"http": proxy, "https": proxy})
        soup = BeautifulSoup(website.text, 'lxml')
        crumb = re.findall('"CrumbStore":{"crumb":"(.+?)"}', str(soup))

        return (header, crumb[0], website.cookies)


def convert_to_unix(date):
    """
    Convert date to unix timestamp.

    parameters: date - in format (dd-mm-yyyy)

    returns integer unix timestamp
    """
    datum = datetime.strptime(date, '%d-%m-%Y')

    return int(mktime(datum.timetuple()))


def load_csv_data(stock, interval='1d', day_begin='00-00-0000', day_end='28-04-2019'):
    """
    Query yahoo finance api to receive historical data in csv file format.

    parameters:
        stock - short-handle identifier of the company

        interval - 1d, 1wk, 1mo - daily, weekly monthly data

        day_begin - starting date for the historical data (format: dd-mm-yyyy)

        day_end - final date of the data (format: dd-mm-yyyy)

    returns a list of comma seperated value lines
    """
    try:
        day_begin_unix = convert_to_unix(day_begin)
    except Exception:
        day_begin_unix = 0

    day_end_unix = convert_to_unix(day_end)

    stock_yf = stock + '.JK'
    proxies = get_proxies()
    proxy = random.choice(proxies)
    header, crumb, cookies = _get_crumbs_and_cookies(stock_yf, proxy)

    with requests.session():
        url = 'https://query1.finance.yahoo.com/v7/finance/download/' \
              '{stock}?period1={day_begin}&period2={day_end}&interval={interval}&events=history&crumb={crumb}' \
              .format(stock=stock_yf, day_begin=day_begin_unix, day_end=day_end_unix, interval=interval, crumb=crumb)

        website = requests.get(url, headers=header, cookies=cookies, proxies={"http": proxy, "https": proxy})

        # save and look at the html coding, find the location of what you need to get, apply the struture and grab
        data = website.text.split('\n')[:-1]
        # print(len(data))
        filename = '{}.csv'.format(stock)
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in range(len(data)):
                writer.writerow(data[i].split(','))

        # print(website.text)
        # return data


if __name__ == '__main__':
    # print(sys.argv[1])
    load_csv_data(stock=sys.argv[1])
