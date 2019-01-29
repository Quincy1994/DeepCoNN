# coding=utf-8

import os
import json
import pandas as pd
import pickle
import numpy as np

TPS_DIR = '/media/iiip/文档/data/yelp2017/yelp_dataset'
TP_file = os.path.join(TPS_DIR,  'review.json')

f = open(TP_file)
users_id = []
items_id = []
ratings = []
reviews = []
np.random.seed(2017)

for line in f:
    print(line)

    js = json.loads(line)
    if str(js['user_id']) == 'unknown':
        print("unknown")
        continue
    if str(js['business_id']) == 'unknown':
        print("unknown2")
        continue

    reviews.append(js['text'])
    users_id.append(str(js['user_id']) + ",")
    items_id.append(str(js['business_id']) + ",")
    ratings.append(str(js['stars']))

data = pd.DataFrame(
    {
        'user_id': pd.Series(users_id),
        'item_id': pd.Series(items_id),
        'ratings': pd.Series(ratings),
        'reviews': pd.Series(reviews)
    }
)[['user_id', 'item_id', 'ratings', 'reviews']]


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

MIN_USER_COUNT = 5
MIN_ITEM_COUNT = 5


def filter_triplets(tp, min_uc=MIN_USER_COUNT, min_ic=MIN_ITEM_COUNT):

    # Only keep this triplets for items which were commented by at least min_ic users.
    itemcount = get_count(tp, 'item_id')
    tp = tp[tp['item_id'].isin(itemcount.index[itemcount >= min_ic])]

    # Only keep this triplets for users who commented on at least min_uc items
    usercount = get_count(tp, "user_id")
    tp = tp[tp['user_id'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'user_id'), get_count(tp, 'item_id')
    return tp, usercount, itemcount


data, usercount, itemcount = filter_triplets(data)

print(data.shape[0])
print(usercount.shape[0])
print(itemcount.shape[0])

unique_uid = usercount.index
unique_sid = itemcount.index
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))

def numerize(tp):
    uid = list(map(lambda x: user2id[x], tp['user_id']))
    sid = list(map(lambda x: item2id[x], tp['item_id']))
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp

data = numerize(data)
tp_rating = data[['user_id', 'item_id', 'ratings']]

n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train = tp_rating[~test_idx]
data = data[~test_idx]

n_ratings = tp_1.shape[0]
print(n_ratings)
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]

tp_train.to_csv(os.path.join(TPS_DIR, 'yelp_train.csv'), index=False, header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'yelp_valid.csv'), index=False, header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'yelp_test.csv'), index=False, header=None)

user_reviews = {}
item_reviews = {}
user_rid = {}
item_rid = {}
for i in data.values:
    user_id = i[0]
    item_id = i[1]
    review = i[3]
    if user_id in user_reviews:
        user_reviews[user_id].append(review)
        user_rid[user_id].append(item_id)
    else:
        user_rid[user_id] = [item_id]
        user_reviews[user_id] = [review]

    if item_id in item_reviews:
        item_reviews[item_id].append(review)
        item_rid[item_id].append(user_id)
    else:
        item_reviews[item_id] = [review]
        item_rid[item_id] = [user_id]

print(user_reviews[11])
print(item_reviews[11])

pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), "wb"))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), "wb"))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')

print(np.sort(np.array(usercount.values)))
print(np.sort(np.array(itemcount.values)))