import os
import hashlib
from collections import Counter

def hash(s):
    return hashlib.sha1(s.encode()).hexdigest()

# get all files under directory and sort them by their name before the first _ (e.g., 0_9.txt should be 0), length first (e.g., 2.txt before 10.txt) then lexicographically
def get_files(d):
    files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    files.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), len(x.split('/')[-1]), x.split('/')[-1]))
    return files

# read file content
def read_file(f):
    with open(f, 'r') as fin:
        return fin.read()
    
# get all files
pos_files = get_files('./data/imdb_ori/aclImdb/train/pos')
neg_files = get_files('./data/imdb_ori/aclImdb/train/neg')

# get all file content
pos_contents = [read_file(f) for f in pos_files]
neg_contents = [read_file(f) for f in neg_files]

# get all file hash
pos_hashes = [hash(c) for c in pos_contents]
neg_hashes = [hash(c) for c in neg_contents]

# get corresponding urls from ./data/imdb_ori/aclImdb/train/urls_*.txt
def get_urls(f):
    with open(f, 'r') as fin:
        return fin.readlines()
    
pos_urls = get_urls('./data/imdb_ori/aclImdb/train/urls_pos.txt')
neg_urls = get_urls('./data/imdb_ori/aclImdb/train/urls_neg.txt')

pos_urls = [u.strip().split('/')[-2] for u in pos_urls]
neg_urls = [u.strip().split('/')[-2] for u in neg_urls]

# create a dictionary to store hash (as key) and url (as value)
pos_hash_url = dict(zip(pos_hashes, pos_urls))
neg_hash_url = dict(zip(neg_hashes, neg_urls))

# get duplicate hash and their indexes in pos_hashes and neg_hashes
pos_counter = Counter(pos_hashes)
neg_counter = Counter(neg_hashes)

pos_duplicate_hashes = [k for k, v in pos_counter.items() if v > 1]
neg_duplicate_hashes = [k for k, v in neg_counter.items() if v > 1]

pos_duplicate_indexes = {k: [i for i, h in enumerate(pos_hashes) if h == k] for k in pos_duplicate_hashes}
neg_duplicate_indexes = {k: [i for i, h in enumerate(neg_hashes) if h == k] for k in neg_duplicate_hashes}


# create a dictionary to store url (as key) and movie category (as value)
category_map = open('./data/imdb_ori/aclImdb/imdb_categories.txt').readlines()
category_map = {line.split(' ')[0].split('/')[-1]: line.split(' ')[1].strip() for line in category_map}

# iterate over duplicate hashes and check if their corresponding content are the same
# for h, indexes in pos_duplicate_indexes.items():
#     contents = [(pos_contents[i][:500], pos_urls[i]) for i in indexes]
#     if len(set(contents)) > 1:
#         # check if the category is the same
#         categories = [category_map[pos_urls[i]] for i in indexes]
#         if len(set(categories)) > 1:
#             print(h, indexes, contents)

# iterate over data and assign category based on hash
def assign_category(example, idx):
    text_hash = hash(example['text'])
    # special case handling
    if text_hash == "0c4e635d35a7c10e28d3c0ce50cd94da68e2c7f6":
        example['category'] = category_map["tt0086662"]
        if idx == 24960:
            example['category'] = category_map["tt0096659"]
        return example
    # general case handling
    if text_hash in pos_hashes:
        example['category'] = category_map[pos_hash_url[text_hash]]
    elif text_hash in neg_hashes:
        example['category'] = category_map[neg_hash_url[text_hash]]
    else:
        raise ValueError(f"Hash {text_hash} not found in pos_hashes or neg_hashes")
    return example
