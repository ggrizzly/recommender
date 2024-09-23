# User-Product Recommender
> Recommender for user / product / rating data, with EDA + algorithms

--------

## Q1: Provide a list of at least 1, and at most the top 5, recommended products for each of the users found below:
### Notes:
- I used cosine similarity using collaborative filtering for this for just users. (aka user-user collab filtering).
- Here we may note that there aren't necessarily enough products bought on average by users that are similar enough (overlapping) (the average comes out to about ~11 ish per user)
    - There are 32k unique users and 320k unique products, so the overlap is tinyyyy
    - Even then, I started with a cosine similarity collab filtering on users (used C to calculate).
    - In Q2, I provide a more complex algorithm

My recommendation for these particular users, would be given the sparsity matrix, if their similarity is low, we should just recommend the most popular objects to start.

For example, see below, I've conducted collab. filtering, and 2 of the users don't have super clear reccomendations (B00HFF691C, B017F8361O)

For those two, I'd honestly just recommend the top 5 purchased products (and rated high (>=4)) (The classic "See what others are buying!" on Amazon). These namely are:

| productId      |  number bought |
| -------------- | -------------- |
| A1KSC91G9AIY2Z |             23 |
| A3I3BI5PFL3MSH |             21 |
| A2FBRED718N0OW |             18 |
| AFADXZJYP5VHW  |             17 |
| A3KPMMEYYH8EGM |             14 |

--------

## Q1:
### userId: B00DGW1SFK
| productId      |  similarity |
| -------------- | ----------- |
| A1KSC91G9AIY2Z |    2.172414 |
| A3I3BI5PFL3MSH |    0.482759 |
| A2FBRED718N0OW |    0.275862 |
| AFADXZJYP5VHW  |    0.206897 |
| A3KPMMEYYH8EGM |    0.172414 |


### userId: B00HFF691C - HERE I WOULD REPLACE THIS ONE WITH TOP PURCHASED AND RATED PRODUCTS
| productId      |  similarity |
| -------------- | ----------- |
| A1W6AL80O62QZM |    0.172414 |
| A1H1YXZVMWLTM3 |    0.172414 |
| A2DPB0B9MKYFN1 |    0.172414 |
| AB0DXFCFRFMZK  |    0.172414 |
| AIA7Q13PWP1LI  |    0.172414 |


### userId: B017F8361O - HERE I WOULD REPLACE THIS ONE WITH TOP PURCHASED AND RATED PRODUCTS
| productId      |  similarity |
| -------------- | ----------- |
| ALW6C3AV7847A  |    0.172414 |
| A2H9B8WR4ZW1FL |    0.172414 |
| A262RJ0QP44NZR |    0.172414 |
| A3SUAH89YINHX8 |    0.172414 |
| A2DUEJ0TDUMWFW |    0.172414 |


### userId: B00F1XUQDC
| productId      |  similarity |
| -------------- | ----------- |
| A1KSC91G9AIY2Z |    2.137931 |
| A3R3NUN0I5XM81 |    0.482759 |
| A141SBKDA5AGQY |    0.310345 |
| A1WLXPXLKLJH67 |    0.241379 |
| AFADXZJYP5VHW  |    0.206897 |


### userId: B00NNLKABW
| productId      |  similarity |
| -------------- | ----------- |
| A1GLZSLS2JHS7X |    0.827586 |
| A35VCRI7V71UN3 |    0.517241 |
| A25NWRWLOUAILI |    0.448276 |
| A17DKZHKTO2B24 |    0.344828 |
| A3S2TK8BDV81FE |    0.344828 |

--------

## Q2: Assume you have a new user with no historical information on them. What product would you recommend if they purchased a product from the list below and rated it highly (5/5). Please provide a list of top recommendations for each of the products below.

### Notes:
- Sparsity in buying items may have an impact. For example, in the first case, the cosine similarity of items is fine, but the fact that only one other person has bought it or the fact that it's just ONE product is probably not enough.
- To this end, I've tried hybrid techniques as well:
    - natural log of popularity metrics as the lambda weight for cosine similarity, 
    - hybrid collab filtering (user-user + item-item)
    - combo of the two ^
- I would say if this is the first time buying this item, it may be recommended again in the style of "buy it again!" Granted, you run into the same goofiness as Amazon, when it recommends you buy another TV / chair, when you just bought it, but for some items (especially if people frequently buy them over and over again, its worth it)
    - I could play more with the frequency + rating, especially using something like TF-IDF, but I didn't have enough time to full explore.
    - Likewise, against TF-IDF (esp. augmented) (K + K * (tf-idf)) as this is used to bias against heavy buyers, but that is honestly what we want - whoever these heavier buyers are, we want more people like em.

One thing I did not have time to do is verify f-scores, rmse, etc. I spent quite a bit of time on cosine similarity, optimization, and making sure things work well.
But even then, this is a start. Given that I have the hyperparameters (albeit chosen "randomly"), we can now tune using RMSE / F-Score metrics.

For the following items, I included:
- user-user similarity collab filtering (done by creating new "users" that just purchased those items, nothing more or less)
- product-product similarity collab filtering (I don't have 1Tb of space for a full matrix calculation. Luckily, we only needed to multiply the vector of the product we were looking at by the matrix, which is significantly less expensive.)
- (0.7\*product_cf + 0.3\*user_cf)\*log(count\*rating/5)
- In the jupyter notebook, you'll see how top results compare - just cosine similarity may not be enough, so we look at all the metrics

Here, the tables are sorted in order of hybrid_cf_weighted_by_popularity (aka the formula I gave above). Some of the regular product-product collab. filtering returned really low connective data, so the weighing absolutely helps in that regard, balancing product relationships, popularity, and products bought by similar users.

### item: A696ZTE6VBU4U
| productId      | counts | product_cf | hybrid_cf | product_cf_weighted_by_popularity | **hybrid_cf_weighted_by_popularity** |
| -------------- | ------ | ---------- | --------- | --------------------------------- | ------------------------------------ |
| AFNGCFXDPE55P  |      6 |   0.272166 |  0.523849 |                          0.925689 |                       **1.781715** * |
| AILBHLCU1VM03  |      3 |   0.410305 |  0.620547 |                          1.082818 |                       **1.637659** * |
| A2WXKLHX5SQC3  |     10 |   0.105409 |  0.240453 |                          0.412363 |                       **0.940658** * |
| A1UX6MRX503HNP |      4 |   0.192450 |  0.301382 |                          0.576529 |                       **0.902859** * |
| A3JLDJMZ4FVG96 |      4 |   0.192450 |  0.301382 |                          0.576529 |                       **0.902859** * |


### item: A6CEOJ5ISIGRB
| productId      | counts | product_cf | hybrid_cf | product_cf_weighted_by_popularity | **hybrid_cf_weighted_by_popularity** |
| -------------- | ------ | ---------- | --------- | --------------------------------- | ------------------------------------ |
| AROYPRQ35VSAT  |      7 |   0.912871 |  1.472343 |                          3.245574 |                       **5.234692** * |
| A2IGYO5UYS44RW |      6 |   1.000000 |  1.533333 |                          3.401197 |                       **5.215169** * |
| A3UXFQUZ6P1JRB |      6 |   1.000000 |  1.533333 |                          3.401197 |                       **5.215169** * |
| A165FHUTQU6L2Z |      6 |   1.000000 |  1.533333 |                          3.401197 |                       **5.215169** * |
| A2RWJPXMBFGCF0 |      6 |   1.000000 |  1.533333 |                          3.401197 |                       **5.215169** * |


### item: A2PLGB52VCSYHG
| productId      | counts | product_cf | hybrid_cf | product_cf_weighted_by_popularity | **hybrid_cf_weighted_by_popularity** |
| -------------- | ------ | ---------- | --------- | --------------------------------- | ------------------------------------ |
| A1CJPRUT6GHTGO |    	2 |   0.577350 |  0.570812 |                          1.329398 |                       **1.314343** * |
| A2065HBMYDXJ1S |    	8 |   0.297746 |  0.375089 |                          0.981321 |                       **1.236231** * |
| A2OS3TIVAKUAHG |    	2 |   0.495074 |  0.513218 |                          1.029477 |                       **1.067207** * |
| A26CPEEWB2WKRE |    	2 |   0.450835 |  0.482251 |                          0.990585 |                       **1.059614** * |
| A35NI8OWUTR1XB |    	2 |   0.408248 |  0.452440 |                          0.940026 |                       **1.041783** * |

### item: A2PUZMHH482FU7
| productId      | counts | product_cf | hybrid_cf | product_cf_weighted_by_popularity | **hybrid_cf_weighted_by_popularity** |
| -------------- | ------ | ---------- | --------- | --------------------------------- | ------------------------------------ |
| A2GOEDQ35EBF1R |	    6 |   1.000000 |  1.533333 |                          3.401197 |                       **5.215169** * |
| A7IB2KI9HJZW   |	    6 |   1.000000 |  1.533333 |                          3.401197 |                       **5.215169** * |
| A1L0QECT7J93ZP |	   10 |   0.730297 |  1.177874 |                          2.856938 |                       **4.607872** * |
| A30VYJQW4XWDQ6 |	   10 |   0.819920 |  1.240611 |                          2.982530 |                       **4.512828** * |
| AMYTL79JMGQ6   |	    9 |   0.735215 |  1.181317 |                          2.730272 |                       **4.386905** * |


### item: A2Q2V2KMKOQDI0
| productId      | counts | product_cf | hybrid_cf | product_cf_weighted_by_popularity | **hybrid_cf_weighted_by_popularity** |
| -------------- | ------ | ---------- | --------- | --------------------------------- | ------------------------------------ |
| A2UERU5XY1CNCG |	    3 |   0.870388 |  0.942605 |	                      2.357055 |                       **2.552622** * |
| A3Z0NDVCF8U8X  |	    4 |   0.710669 |  0.830802 |	                      2.128974 |                       **2.488859** * |
| A3T15AD35VVLWC |	    5 |   0.550482 |  0.718671 |	                      1.771933 |                       **2.313312** * |
| A3BTG5QT11003T |	    3 |   0.710669 |  0.830802 |	                      1.924527 |                       **2.249853** * |
| A3C4692KE4G6TD |	    3 |   0.710669 |  0.830802 |	                      1.924527 |                       **2.249853** * |

> If you find any limitations or an inability to produce a recommendation for Q1 or Q2, please state the reasoning and if you would propose any alternatives to the goal of identifying the best product recommendation for user(s).

One massive limitation is this is technically done on those "per product" bases - e.g. I couldn't calculate the cosine similarity matrix (even if using multiprocessing techniques). I tried using Dask, and that failed as well, so I resorted to using C, and being clever and not calculating the world.

Likewise, if a new product is introduced or a new user, one would have to recalculate a lot of the cosine distance similarities, even if for that one item / user. Some form of hashing could be used to speed up comparisons between users, if specific to different "fingerprints"

There are certainly other algorithms I could've looked at (in fact I did).
- SVD/ALS crashed trying to compute everything, even if it was a sparse matrix.
- Neural Collaborative Filtering was another option, I didn't have time exactly to explore, so I went with this ^ approach.
- KNN / K-means could've just extended the above logic. The problem, unfortunately is it too crashed when I tried to run it lol. 
- The Apriori algorithm could've worked great if I grouped products and users based on timestamps
    - This would be more aligned if I saw what users BOUGHT together. The timestamps, however are for when a product was rated, not when it was bought.
- Page Rank
    - Graph algorithms could do wonders here - weights for the relationships (edges) would be based on how many times it was bought by a user and how it was reviewed. Probably would have to normalize the data from 1-5 to -1 to 1.
- Autoencoders
    - Similar to Neural Collab filtering, with the cosine similarity matrix. However, didn't really have a chance to run this.
- Euclidean Transformations (instead of cosine):
    - Based on this paper - http://ulrichpaquet.com/Papers/SpeedUp.pdf
- Any of the numerous algorithms / approaches with collaborative filtering listed in the paper below:
    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9287091/
- Multi-armed bandit Algorithm
    - Could've helped with the cold start problem in Q2.
- Time aware collaborative filtering
    - Like I mentioned, I didn't have time to incorporate time as an important variable. Here's are two good reads on it I looked at, that I was considering:
        - https://paginas.fe.up.pt/~prodei/dsie12/papers/paper_18.pdf
        - https://cdn.techscience.cn/files/cmc/2019/v61n2/20191015025650_32258.pdf
    - TimeSVD++ would've worked great for this as well, that's one I looked into.
        - https://ieeexplore.ieee.org/document/8610156

That's about it!