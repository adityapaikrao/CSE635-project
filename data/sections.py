vault = {
    'medline_plus': {
        'BASE_URL': 'https://medlineplus.gov/diabetes.html',
        'xpaths': [
            '//*[@id="more_encyclopedia"]',
            '//*[@id="related-topics"]',
            '//*[@id="section77"]',
            '//*[@id="section51"]',
            '//*[@id="section92"]',
            '//*[@id="section78"]',
            '//*[@id="section82"]',
            '//*[@id="section69"]'
        ]
    },
    'cdc_diabetes': {
        'BASE_URL': 'https://www.cdc.gov/diabetes/basics/index.html',
        'xpaths': [
            '//div[contains(@class, "syndicate")]',  # main article blocks
            '//div[contains(@class, "card-body")]'   # sub-article links
        ]
    },
    'mayo_diabetes': {
        'BASE_URL': 'https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444',
        'xpaths': [
            '//div[contains(@class, "m-card--content")]',
            '//a[@href]'  # general fallback
        ]
    },
    'nih_diabetes': {
        'BASE_URL': 'https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes',
        'xpaths': [
            '//ul[contains(@class, "list--inline")]',  # article cards
            '//div[contains(@class, "content")]'
        ]
    },
    'ada_diabetes': {
        'BASE_URL': 'https://diabetes.org/diabetes',
        'xpaths': [
            '//div[contains(@class, "field-content")]',
            '//a[@href]'
        ]
    }
}
