from requests_html import HTMLSession
from sections import xpaths


def GetArticleLinks(base_url, xpath):
    response = sess.get(base_url)
    print(response.status_code)
    response.html.render(sleep=1)
    
    with open("rendered_page.html", "w", encoding="utf-8") as file:
        file.write(response.html.html)
    articles = response.html.xpath(xpath)  
    
    article_links = set()
    # Check if we found articles
    if not articles:
        return article_links
    
    for article in articles:
        article_links.update(article.absolute_links)
    
    return article_links

if __name__ == "__main__":
    BASE_URL = 'https://medlineplus.gov/diabetes.html'
    sess = HTMLSession()
    links = set()

    for xpath in xpaths:
        print(f'scraping for section {xpath}')
        article_links = GetArticleLinks(BASE_URL, xpath)
        links.update(article_links)
    
    print(links)
    
