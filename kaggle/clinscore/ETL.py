#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:27:06 2022

@author: hal
"""
import pandas as pd
import numpy as np
from IPython.display import display
import re

from sklearn.model_selection import train_test_split

import spacy
from spacy import displacy
from spacy.tokens import DocBin
import json
from tqdm import tqdm

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

class spacy_prep:
    def __init__(self, feature_desc, location_desc, note_corpus):
        self.feature_desc = feature_desc
        self.location_desc = location_desc
        self.note_corpus = note_corpus
        self.nlp = spacy.blank('en')
        
    def start_prep(self):
        location_dict = {}
        rels = []
        feature_keys = self.feature_desc[:, 1]
        
        for key in feature_keys:
            location_dict[key] = []
        
        for entry in self.location_desc:
            for feat in self.feature_desc:
                if entry[0] == feat[0]:
                    if entry[1] !='[]':
                        stripper = entry[1][0:len(entry[1])-1]
                        stripper = stripper[1:]
                        #stripper = re.sub(' ', '-', stripper)
                        #stripper = re.sub("'", "", stripper)
                        #stripper = re.split(";", stripper)
                        
                        #stripper = re.split("t", stripper)
                        stripper = re.split(",", stripper)
                        stripper = [[int(s) for s in re.findall(r'\b\d+\b', sentry)] for sentry in stripper]
                        #print(stripper)

                        #stripper = re.split("," , stripper)
                        #stripper = [re.sub(",","", ent) for ent in stripper]
                        #stripper = [re.split(";", entity) for entity in stripper]
                        #stripper = [[int(e) for e in entu] for entu in stripper]
                        indices = []
                        for given_list in stripper:
                            for list_entry in given_list:
                                #print(list_entry)
                                
                                indices.append(list_entry)
                        indices = list(divide_chunks(indices, 2)) #Paired chunks
                        #print(indices)
                        note_num = entry[2]
                        #print(note_num)
                        #indices = np.split(indices, 2)
                        
                        #print(indices)
                        location_dict[feat[1]].append(tuple([note_num, indices]))
                        
        for feat in self.feature_desc:
            for (note_num, indexes) in location_dict[feat[1]]:
                for entry in indexes:
                    start = entry[0]
                    stop = entry[1]
                    my_note = self.note_corpus.loc[note_num]
                    rels.append([my_note, [start, stop], feat[1]])
        return rels
    
    def training_prep(self):
        preproccd_data = self.start_prep()
        collective_dict = {'TRAINING_DATA': [], 
                           'VALIDATION_DATA': []}
        
        
        for note in self.note_corpus.values:
            entities = []
            for entry in preproccd_data:
                
                if entry[0] == note:
                    #print("yes")
                    start = entry[1][0]
                    stop = entry[1][1]
                    key = entry[2]
                    entities.append((start, stop, key))
                            
            results = [note, {"entities": entities}]
            if results[1]['entities'] == []:
                del results[1]
                del results[0]
                
            #print(results)
            collective_dict['TRAINING_DATA'].append(results)
            
        collective_dict['TRAINING_DATA'] = [x for x in collective_dict['TRAINING_DATA'] if x != []]
        
        collective_dict['TRAINING_DATA'], collective_dict['VALIDATION_DATA'] = train_test_split(collective_dict['TRAINING_DATA'] 
                                                                                                , test_size=0.2, random_state=42)
        json_string = json.dumps(collective_dict)
        
        with open('clin_data.json', 'w') as outfile:
            outfile.write(json_string)
            
        return collective_dict
    
    def create_training(self):
        coll_dict = self.training_prep()
        TRAIN_DATA = coll_dict['TRAINING_DATA']
        db = DocBin()
        for text, annot in tqdm(TRAIN_DATA):
            doc = self.nlp.make_doc(text)
            ents = []
    
            # create span objects
            for start, end, label in annot["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract") 
    
                # skip if the character indices do not map to a valid span
                if span is None:
                    #print("start: {}, end: {}, label: {}".format(start, end, label))
                    print("Skipping entity.")
                else:
                    #print("start: {}, end: {}, label: {}".format(start, end, label))
                    ents.append(span)
                    # handle erroneous entity annotations by removing them
                    try:
                        doc.ents = ents
                    except:
                        # print("BAD SPAN:", span, "\n")
                        ents.pop()
            doc.ents = ents
    
            # pack Doc objects into DocBin
            db.add(doc)
            
        return db
    
    def create_validation(self):
        coll_dict = self.training_prep()
        VAL_DATA = coll_dict['VALIDATION_DATA']
        db = DocBin()
        for text, annot in tqdm(VAL_DATA):
            doc = self.nlp.make_doc(text)
            ents = []
    
            # create span objects
            for start, end, label in annot["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract") 
    
                # skip if the character indices do not map to a valid span
                if span is None:
                    #print("start: {}, end: {}, label: {}".format(start, end, label))
                    print("Skipping entity.")
                else:
                    #print("start: {}, end: {}, label: {}".format(start, end, label))
                    ents.append(span)
                    # handle erroneous entity annotations by removing them
                    try:
                        doc.ents = ents
                    except:
                        # print("BAD SPAN:", span, "\n")
                        ents.pop()
            doc.ents = ents
    
            # pack Doc objects into DocBin
            db.add(doc)
            
        return db
    
def submission(feature_desc, note_corpus):
    #sub_tup = tuple([feature_desc[:, 1], feature_desc[:, 0]] )
    sub_tup = []
    
    for entry in feature_desc:
        sub_tup.append((entry[1], entry[0], entry[2]))
        
    #sub_dict = dict(sub_tup)
    #print(sub_dict)
    return sub_tup
    
    
    
                                 
def load_data():
    #Load raw data
    feature_frame = pd.read_csv('data/features.csv')
    note_frame = pd.read_csv('data/patient_notes.csv')
    train_frame = pd.read_csv('data/train.csv')
    print("Feature frame columns:\n{}\nNote frame columns:\n{}\nTrain frame columns:\n{}\n\n".format(feature_frame.columns, note_frame.columns, train_frame.columns))
    
    note_frame.set_index('pn_num', inplace=True)
    note_corpus = note_frame['pn_history']
    feature_frame = feature_frame.drop_duplicates('feature_text')
    
    feature_desc = feature_frame[['feature_num', 'feature_text', 'case_num']].values
    location_desc = train_frame[['feature_num', 'location', 'pn_num']].values
    
    '''
    prepper = spacy_prep(feature_desc, location_desc, note_corpus)
    
    TRAIN_DATA_DOC = prepper.create_training()
    TRAIN_DATA_DOC.to_disk("./TRAIN_DATA/TRAIN_DATA.spacy")
    
    VAL_DATA_DOC = prepper.create_validation()
    VAL_DATA_DOC.to_disk("./TRAIN_DATA/VAL_DATA.spacy")
    '''
    sub_dict = submission(feature_desc, note_corpus)
    
def tester():
    note_frame = pd.read_csv('data/patient_notes.csv')
    note_corpus = note_frame[['pn_history', 'pn_num']]
    
    notes = note_corpus.values
    
    feature_frame = pd.read_csv('data/features.csv')
    feature_desc = feature_frame[['feature_num', 'feature_text', 'case_num']].values
    
    
    model_test = notes[50][0]
    #print(model_test)
    
    nlp_output = spacy.load("output/model-best")
    doc = nlp_output(model_test)
    displacy.render(doc, style="ent")
    
    entity_list = []

    for ent in doc.ents:
        print("Label: {}, Span: {}:{}".format(ent.label_, ent.start_char, ent.end_char))
        entity_list.append([ent.label_, ent.start_char, ent.end_char])
        
    sub_tup = submission(feature_desc, note_corpus)
    
    final_sub = []
    
    for (feat_name, feat_num, case_num) in sub_tup:
        for entity in entity_list:
            label = entity_list[0]
            start = entity_list[1]
            end = entity_list[2]
            
            if label == feat_name:
                final_sub.append([feat_num, case_num, start, end])
                
    print(final_sub[0])
    
def setup():
    load_data()
    
if __name__ =="__main__":
    setup()
    
    
    
    