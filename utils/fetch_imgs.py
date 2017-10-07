import sys
import re
import urllib.request as request
import os
import json
from bs4 import BeautifulSoup

class GoogleImageSearch(object):

    def __init__(self):
        self.headers = { 'User-Agent': 
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36" }

    def _get_soup(self, url, headers):
        return BeautifulSoup(request.urlopen(request.Request(url, headers=headers)), 'html.parser')

    def fetch_img_links(self, query):
        base_url = "https://www.google.com/search"
        path_query = "?as_st=y&tbm=isch&hl=en&as_q=ai+face&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=isz:m,itp:face,sur:f".format(query)
        soup = self._get_soup(base_url+path_query, self.headers)
        image_links = []
        for link in soup.find_all("div", { "class": "rg_meta" }):
            link, image_type = json.loads(link.text)["ou"], json.loads(link.text)["ity"]
            image_links.append((link, image_type))
        return image_links

    def fetch_and_write(self, image_urls, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i, (url, image_type) in enumerate(image_urls):
            if i % 10 == 0:
                print('Downloading {}/{} images...'.format(i+10, len(image_urls)))
            try:
                req = request.Request(url, headers=self.headers)
                resp = request.urlopen(req)
                raw_img = resp.read()
             
                cntr = len([i for i in os.listdir(directory) if image_type in i]) + 1
                path = os.path.join(directory, image_type + "_" + str(cntr) + "." + (image_type or 'jpg'))
                f = open(path, 'wb')
                f.write(raw_img)
                f.close()
            except Exception as e:
                print("Failed to load: <{}>\nError: {}".format(url, e))

    def download(self, query):
        image_links = self.fetch_img_links(query)
        self.fetch_and_write(image_links, query)

if __name__ == '__main__':
    n_args = len(sys.argv)
    query = 'ai+face'
    if n_args > 1:
        query = sys.argv[1]
    gis = GoogleImageSearch()
    gis.download(query)

