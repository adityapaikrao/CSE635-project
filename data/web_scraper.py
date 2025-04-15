from requests_html import HTMLSession
import trafilatura
import requests
from sections import vault
from collections import defaultdict
import json
from urllib.parse import urljoin


def ScrapeFromLink(full_url):
    try:
        # Download the HTML content of the page
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()

        # Use Trafilatura to extract text from the HTML
        downloaded = trafilatura.fetch_url(full_url)
        if downloaded:
            content = trafilatura.extract(downloaded)
            return content
        else:
            print(f"Could not extract content from {full_url}")
            return None
    except Exception as e:
        print(f"Error fetching {full_url}: {e}")
        return None
    
def GetArticleLinks(base_url, xpath):
    response = sess.get(base_url, headers=headers)
    # print(response.status_code)
    if response.status_code == 200:
        print(f'fetching article {base_url}')
    else:
        print(f'failed with {response.status_code}')
        return {}

    articles = response.html.xpath(xpath)
    
    if not articles:
        return {}
    
    data_combined = defaultdict(str)
    for article in articles:
        anchors = article.xpath('.//a[@href]')
        for anchor in anchors:
            href = anchor.attrs.get("href")
            is_spanish = anchor.text.strip().lower() == 'spanish'
            if href and not is_spanish:
                full_url = urljoin(base_url, href)
                if full_url not in links:
                    # Start scraping the data here
                    data = ScrapeFromLink(full_url)
                    data_combined[full_url] = data
                    links.add(full_url)

    return data_combined


if __name__ == "__main__":
    sess = HTMLSession()
    links = set()

    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

    scraped_text = {}

    for site, params in vault.items():
        count = 0
        scraped_text[site] = []
        print(f'Scraping for {site}')
        BASE_URL = params.get('BASE_URL', None)
        
        # scrape based on xpaths
        xpaths = params.get('xpaths', [])
        if xpaths:
            for xpath in xpaths:
                data = GetArticleLinks(BASE_URL, xpath)
                scraped_text[site].append(data)
                count += len(data)
        
        print(f'Scraped {count} articles.......')
    
        with open('scraped_text.json', "w") as f:
            json.dump(scraped_text, f, indent=4)
    
    print(f'Scraped a total of {len(links)} articles')
    
