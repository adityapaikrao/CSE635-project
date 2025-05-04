import json
with open("data/scraped_text.json") as f:
    data = json.load(f)
print("Total sites:", len(data))

# Check the structure of the data first
if isinstance(data, dict):
    # If data is a dictionary with lists as values
    count = sum(len(articles) if isinstance(articles, list) else 0 for articles in data.values())
    print("Total articles:", count)
else:
    # If data itself is a list
    count = len(data)
    print("Total articles:", count)

# Alternatively, you can first print the structure to understand it better
print("Data structure:", type(data))
if isinstance(data, dict):
    print("First key example:", next(iter(data)))
    print("First value type:", type(next(iter(data.values()))))