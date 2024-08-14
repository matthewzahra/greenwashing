'''
Data is cleaned in following steps:
1. context's that contain "illegal" characters and the phrase "Logo" get removed
2. Nostril looks and deletes any rows with gibberish context
3. pre-trained n-gram does the same
3. links are deleted from contexts (but rest of the text in the context is kept)
4. rows whose contexts have been made blank by step 3. are deleted

Note: Lines like the following are picked up by neither the translator or the 2 gibberish detectors (but there don't seem to be too many of them)
9558,... 茶農家 ささら屋 / とりどり工房 / torico Art Beads Collection Namry / ouchi-ilo cf tocotoco / Original Color / Cashatto Photo カラフル☆コットンテイル ...,International Labour Organization (ILO) - Labour Standards ,0
'''
import pandas as pd
import re
import pickle
from datasets import Dataset, load_dataset

#https://github.com/casics/nostril
from nostril import nonsense_detector as nd

# no, above model is outdated, use this: https://github.com/amitness/Gibberish-Detector/blob/master/gib_detect_train.py
from gibDetector import gib_detect_train


#delete specified characters or keywords
# columnName to be specifed!
def deleteSpecial(ds): 
    ds = ds.filter(lambda row: "Logo" not in row['context'] and "" not in row['context'])

    #following special character breaks a lot of functions so should be removed and agreed to reove all context's with "Logo"
    pattern = "|Logo+" 
    ds = ds.filter(lambda row: )
    mask = df[columnName].str.contains(pattern, na=False,regex=True)
    return df[~mask]


#use Nostril
def detectNonsense(row):
    sixLetters = "[a-zA-Z].*[a-zA-Z].*[a-zA-Z].*[a-zA-Z].*[a-zA-Z].*[a-zA-Z]"   #Nostril only works if there are at least 6 letter characters present
    if re.search(sixLetters,row['context']) != None:
        return nd.nonsense(row['context']) #True if gibberish
    return True    #if less than 6 letters, assume to be a bad context (historically accurate)

def deleteGibberishNostril(ds):
    ds = ds.filter(lambda example: not detectNonsense(example))
    return ds


#the model data for the n-gram from pre-trained model
model_data = pickle.load(open('gibDetector/gib_model.pki', 'rb'))

def gibDetector(row):
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    return gib_detect_train.avg_transition_prob(row['context'], model_mat) > threshold #False if gibberish

def deleteGibberishNGram(ds):
    ds = ds.filter(gibDetector)
    return ds


   
def removeLinks(ds):
    df = Dataset.to_pandas(ds)
    pattern = "http+.*[ |.|,]"    #matches "http..... followed by a comma, space or fullstop "
    df[columnName] = df[columnName].str.replace(pattern,"",regex=True)  #replaces above matches by empty string
    new_ds = Dataset.from_pandas(df)
    return ds

#remove links may leave some contexts blank
def deleteBlanks(ds):
    ds = ds.map(lambda row: {'context': float("NaN") if row['context']=="" else row['context']})
    return ds

    
df = pd.read_csv('prep_output.csv')

# the combined gibberish removal function
def removeGibberish(ds):
    deleteBlanks(deleteGibberishNGram(deleteGibberishNostril(deleteSpecial(ds))))


ds = deleteSpecial(ds)  #delete rows that contain specified phrases/characters

ds = deleteGibberishNostril(ds) #use Nostril library to look for gibberish

ds = deleteGibberishNGram(ds)   #use n-gram gibberish detector

ds = removeLinks(ds)

ds = deleteBlanks(ds)

df = Dataset.to_pandas(ds)
df.to_csv("cleanedFinal2.csv",index=False)  #write results to .csv file