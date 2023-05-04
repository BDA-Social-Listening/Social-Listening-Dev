from pyspark import SparkContext
from pyspark.sql import SparkSession
import json
import mmh3
import sys
import numpy as np
from collections import defaultdict


'''
Trial Data: 
A34A1UP40713F8_B000NCTOUM, 
A3LGZ8M29PBNGG_B000W3P4AQ, 
AFUVGAUNQVT0S_B004XLDE5A

Test Data:
A26S0R5B6D0BP9_B00006LVF1,
A3SS6919NRQ2MF_B00006I5SA,
AYM76JWI220Z4_B0000WR5EW

'''


trial_data = ['A34A1UP40713F8_B000NCTOUM', 'A3LGZ8M29PBNGG_B000W3P4AQ', 'AFUVGAUNQVT0S_B004XLDE5A']
test_data = ['A26S0R5B6D0BP9_B00006LVF1', 'A3SS6919NRQ2MF_B00006I5SA', 'AYM76JWI220Z4_B0000WR5EW']

# Define the function to create unique ids for each record
def create_record_id(record):
    return record['reviewerID'] + '_' + record['asin']

def create_shingles(review):
    shingles = set()
    for i in range(len(review)-4):
        shingles.add(review[i:i+5])
    return shingles

# function to create signatures for each record
def create_signature(shingles, hash_functions):
    signature = []
    for i in range(len(hash_functions)):
        min_hash = float('inf')
        for shingle in shingles:
            hash_value = mmh3.hash(str(shingle), seed=hash_functions[i], signed=False)
            if hash_value < min_hash:
                min_hash = hash_value
        signature.append(min_hash)
    return signature


def get_bands(sig, n_rows):
    bands = []
    for i in range(0, len(sig), n_rows):
        bands.append(sig[i:(i+n_rows)])
    return bands


def hash_bands(bands):
    h_bands = []
    for i in range(len(bands)):
        h_bands.append(mmh3.hash(''.join(map(str, bands[i])), signed=False, seed=i) % 1000000)
    return h_bands


def compare_sigs(sig1, sig2):
    return (np.array(sig1) == np.array(sig2)).any()


def jaccard_similarity(set1, set2):
    num_intersect = len(set1.intersection(set2))
    num_union = len(set1.union(set2))
    sim = num_intersect / num_union 
    return sim

def signatures(sigrdd1):
    for record_id in data_main:
        sigrdd2 = sigrdd1.filter(lambda rec: rec[0] == record_id).map(lambda x: (x[0], x[1], create_shingles(x[1])))
        hash_functions1 = range(1000)
        hash_functions1 = sc.broadcast(hash_functions1)
        sigrdd2 = sigrdd2.mapValues(lambda shingles: create_signature(shingles, hash_functions1.value)).collect()

        print(f"Record ID: {record_id}, Signature: {sigrdd2[0][1]}")


#Create New Spark Session to make use of RDDs
spark = SparkSession.builder.appName("NewSession").getOrCreate()

# Load the JSON file into an RDD
filename=sys.argv[1]
sc = spark.sparkContext
rdd=sc.textFile(str(filename))

if rdd.count() < 3000 : data_main = trial_data
else : data_main = test_data

#convert into JSON object
rdd = rdd.map(lambda x: json.loads(x))

#make tuple of ID and record
rdd = rdd.map(lambda each: (create_record_id(each), each.get('reviewText', '')))

# Reduce to unique record ids
rdd = rdd.reduceByKey(lambda a, b: a)

signatures(rdd)

# Create 5-shingles for each review
shingles_rdd = rdd.map(lambda x: (x[0], x[1], create_shingles(x[1])))
print(shingles_rdd.take(5))


# broadcast hash functions
hash_functions = range(1000)
hash_functions = sc.broadcast(hash_functions)
sigrdd = shingles_rdd.mapValues(lambda shingles: create_signature(shingles, hash_functions.value))


# for record_id in data_main:
#     sig = sigrdd.filter(lambda rec: rec[0] == record_id).collect()
#     print(f"Record ID: {record_id}, Signature: {sig[0][1]}")



br = 1000
n_bands = 20
n_rows = br // n_bands

bandrdd = sigrdd.map(lambda x: (x[0], get_bands(x[1], n_rows)))
hbandrdd = bandrdd.map(lambda x: (x[0], hash_bands(x[1])))


testing_data1 = hbandrdd.filter(lambda x: x[0] in data_main).collect()
testing_data2 = sigrdd.filter(lambda x: x[0] in data_main).collect()
hbandrdd = hbandrdd.filter(lambda x: x[0] not in data_main)


output=set()
temp=defaultdict(list)
i=0
for r, sig in testing_data1:
    output=hbandrdd.map(lambda x: (x[0], compare_sigs(x[1], sig))).filter(lambda x: x[1] == True).map(lambda x: x[0]).collect()
    output=sigrdd.filter(lambda x : x[0] in output).collect()

    for out in output:
        temp1=jaccard_similarity(set(testing_data2[i][1]),set(out[1]))
        if temp1>=0.8: temp[r].append(out[0])
    i+=1


for match in temp:
    if len(match)<=10: print(f'For the target record {match} we have the following similar records : {temp[match]}')
    else : print(f'For the target record {match} we have the following similar records : {temp[match][:10]}')
 

