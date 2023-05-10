import random
import numpy as np
import math
import csv
import json
import pyspark
import hashlib
import sys

from pyspark import SparkConf, AccumulatorParam
from pyspark.context import SparkContext
from pyspark import StorageLevel
from pyspark.sql import SparkSession

path = 'a2_p2_Mittal_114855990_OUTPUT.txt'
sys.stdout = open(path, 'w')

filename = sys.argv[1]
if "Music" in filename:
    filterList = ["A26S0R5B6D0BP9_B00006LVF1", "A3SS6919NRQ2MF_B00006I5SA", "AYM76JWI220Z4_B0000WR5EW"]
else:
    filterList = ["A34A1UP40713F8_B000NCTOUM", "A3LGZ8M29PBNGG_B000W3P4AQ", "AFUVGAUNQVT0S_B004XLDE5A"] 

session = SparkSession.builder\
            .appName("Assignment2")\
            .config("spark.driver.memory", "15g") \
            .config("spark.dynamicAllocation.enabled", "true") \
            .config("spark.shuffle.service.enabled", "true") \
            .config("spark.executor.cores", "4") \
            .config("spark.rpc.message.maxSize", "256") \
            .config("spark.executor.memory", "14g") \
            .config("spark.executor.instances", "4") \
            .getOrCreate()
    
sc = session.sparkContext
# sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

class SetAccum(AccumulatorParam):
    def zero(self, zeroValue = set()):#overwrite this
        return set()
    def addInPlace(self, v1, v2):#overwrite this
        v1.update(v2)
        return v1
SetAcc = sc.accumulator(set(), SetAccum())

seeds = []
for i in range(1000):
    random.seed(i)
    a = random.randint(0, 10000)
    b = random.randint(0, 10000)
    l = [a, b]
    seeds.append(l)

seedListSC = sc.broadcast(seeds)

## Taken fron CHATGPT using PROMPT "python code to create k-shingles of string"
def create_k_shingles(string, k):
    """
    Creates k-shingles of a string.
    
    Parameters:
    string (str): The input string.
    k (int): The length of the shingles.
    
    Returns:
    list: A list of k-shingles of the input string.
    """
    shingles = set()
    n = len(string)
    p = n - k + 1
    for i in range(0,p):
        shingles.add(string[i:i+k])
    SetAcc.add(shingles)
    return shingles

def getReviewText(allData):
    for v in allData:
        rec = json.loads(v)
        k = rec['reviewerID']+'_'+ rec['asin']
        a = ""
        try:
            a = rec['reviewText']
        except KeyError:
            a = ""
        yield (k, a)

# dataRDD = sc.textFile('hdfs:/data/Appliances_5.json')
dataRDD = sc.textFile(filename)
dataRDD.count()
# dataRDD = sc.textFile('Appliances_5.json')

jsonRDD = dataRDD.mapPartitions(getReviewText).reduceByKey(lambda v1,v2: v1)
# jsonRDD = dataRDD.map(lambda rec: getReviewText).reduceByKey(lambda v1,v2: v1)
jsonRDD.persist(StorageLevel.MEMORY_AND_DISK)
reviewReduceRDD = jsonRDD.map(lambda v: (v[0], v[1], create_k_shingles(v[1], 5)))
reviewReduceRDD.persist(StorageLevel.MEMORY_AND_DISK)

################## Checkpoint 2.1 ###################
print("Checkpoint2.1: ")
print(reviewReduceRDD.take(5))
reviewReduceRDD.count()
################## Checkpoint 2.1 ###################

set_size = len(SetAcc.value)
# print(len(SetAcc.value))
hash_rangeSC = sc.broadcast(set_size)
hashShingles = dict()
for k, i in enumerate(SetAcc.value):
    hf = k%len(seedListSC.value)
    hashShingles[i] = (hf, (seedListSC.value[hf][0] * hash(i) + seedListSC.value[hf][1])%1001)
    
hashShinglesSC = sc.broadcast(hashShingles)
num_bucketsSC = sc.broadcast(100)

class DictAccum(AccumulatorParam):
    def zero(self, zeroValue = dict()):#overwrite this
        return dict()
    def addInPlace(self, v1, v2):#overwrite this
        for i in v2.keys():
            v1[i] = v2[i]
        return v1
DictAcc = sc.accumulator(dict(), DictAccum())

class ListAccum(AccumulatorParam):
    def zero(self, zeroValue = []):#overwrite this
        return []
    def addInPlace(self, v1, v2):#overwrite this
        return v1+v2

listAcc = sc.accumulator([], ListAccum())

rows = sc.broadcast(10)
bands = sc.broadcast(100)

def listRemoveInf(l):
    ans  = []
    for i, v in enumerate(l):
        if v!= float('inf'):
            ans.append(str(i)+"$"+str(v))
    return ans

def AddToBucketsNik(key, signatures):
    allSetList = []
    t = [hash('|'.join(listRemoveInf(signatures[i:i+rows.value])))%num_bucketsSC.value for i in range(0, 1000, rows.value)]
    l = 0
    for r in t:
        if r == "":
            continue
        allSetList.append(str(l)+"_"+str(r)+"_"+str(key))
        l+=rows.value
    listAcc.add(allSetList)
    return

def minhash_signature_filtered(key, shingles):
    signature = [float('inf')] * len(seedListSC.value)
    # hash_range = len(shingles)
    for shingle in shingles:
        hf = hashShinglesSC.value[shingle][0]
        value = hashShinglesSC.value[shingle][1]
        signature[hf] = min(signature[hf], value)
    d = dict()
    d[key] = shingles
    DictAcc.add(d)
    # print((key,signature))
    AddToBucketsNik(key, signature)
    return (key,signature,shingles)

################### Checkpoint 2.2 ###################
filterListSC = sc.broadcast(filterList)
filterRDD = reviewReduceRDD.filter(lambda v: v[0] in filterListSC.value).map(lambda v: (minhash_signature_filtered(v[0], v[2])))
filterRDD.persist(StorageLevel.MEMORY_AND_DISK)
# print("Checkpoint2: ", filterRDD.collect())
# print("Checkpoint2.2: ", filterRDD.count())
checkpoint2 = filterRDD.map(lambda v: (v[0], v[1]))
print("Checkpoint2.2: ", checkpoint2.collect())
################### Checkpoint 2.2 ###################
filterDict = sc.broadcast(DictAcc.value)
selectedHashesSC = sc.broadcast(listAcc.value)

def createSignatureFromSet_Unfiltered(key, shingles):
    signature = [float('inf')] * len(seedListSC.value)
    for shingle in shingles:
        hf = hashShinglesSC.value[shingle][0]
        value = hashShinglesSC.value[shingle][1]
        signature[hf] = min(signature[hf], value)
    return (key, signature, shingles)

def calculateJaccard(k1,k2,m1,m2):
    i = m1.intersection(m2)
    u = m1.union(m2)
    return (k1, float(len(i)/len(u)))

def checkForCandidates(key, signatures, shingles):
    t = [hash('|'.join(listRemoveInf(signatures[i:i+rows.value])))%num_bucketsSC.value for i in range(0, 1000, rows.value)]
    l = 0
    d = dict()
    ans = []
    for k in filterListSC.value:
        d[k] = 0
    for r in t:
        if r == "":
            continue
        for k in filterListSC.value:
            if d[k]==0:
                if str(l)+"_"+str(r)+"_"+str(k) in selectedHashesSC.value:
                    d[k]+=1
            else:
                continue   
        l+=10
    for k in filterListSC.value:
        if d[k]>0:
            h = calculateJaccard(k, key, filterDict.value[k], shingles)
            if h[1]>=float(4/5):
                ans.append(h)
    return (key, ans)

nonFilteredRDD = reviewReduceRDD.filter(lambda v: v[0] not in filterListSC.value).map(lambda v: createSignatureFromSet_Unfiltered(v[0], v[2]))
selectedRDD = nonFilteredRDD.map(lambda v: checkForCandidates(v[0], v[1], v[2])).filter(lambda v: len(v[1])>0)
# selectedRDD.persist(StorageLevel.MEMORY_AND_DISK)

################### Checkpoint 2.3 ###################
print("Checkpoint 2.3 : ", selectedRDD.collect())
# print(selectedRDD.collect())
################### Checkpoint 2.3 ###################

