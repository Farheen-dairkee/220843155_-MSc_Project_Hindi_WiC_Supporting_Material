# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 04:01:15 2023

@author: fdair
"""

#Import Libraries
import regex as re
import pandas as pd
from itertools import product


#Create a list of all the words
list1=[]
with open('all_words.txt', 'r', encoding='utf-8') as file:
    content = file.readlines()


def replace_non_utf8(string):
    # Define regex pattern to match non-UTF-8 characters
    pattern = r'[^\x00-\x7F\u0900-\u097F]'
    
    # Replace non-UTF-8 characters with an empty string
    cleaned_string = re.sub(pattern, '', string)
    
    return cleaned_string


def replace_word_lemmas(word,sentence):
    words=sentence.split(" ")
    end=0
    start=0
    for j in range(0,len(words)):
        found1=words[j].find(word)
        #print(found1)
        if found1!=-1:
            start+=found1
            #print(start)
            end=start+len(word)
            #print(end)
            target_word=words[j]
            #print(target_word)
            replace_word=target_word[:found1]+" "+target_word[found1:end]
            #print(replace_word)
            if replace_word.lstrip()==target_word:
                se=end-start
                replace_word=target_word[:se]+" "+target_word[se:]
                print(replace_word)
            sentence=sentence.replace(target_word,replace_word)
            sentence=sentence.replace("  "," ")
            start-=found1
        start+=len(words[j])+1
    return start,end,sentence

#From the corpus, create a dictionary of sense, context sentence for every word
file_path=r'C:\Users\fdair\Documents\College Work\MSc Project\483531Sense_Annotated_Hindi_Corpus\Sense Annotated Hindi Corpus'


list3=[]
list4=[]
sense_cnt=0
for l1 in list1:
    print(l1)
    sense_cnt=0
    word_ind=index_d[l1]
    with open(f'{file_path}\{l1}\\No_of_Senses.txt', 'r', encoding='utf-8') as file:
        no_of_senses=int(file.read())
    
    sense_cnt+=1
    sum_context=0
    lists = [[] for _ in range(no_of_senses)]
    lists_1 = [[] for _ in range(no_of_senses)]
    while sense_cnt<no_of_senses+1:
        #print(sense_cnt)
        str1='ContextSenses00'+str(sense_cnt) #name of text file with context
        str2='Senses00'+str(sense_cnt) #name of text file with sense
        #senses=[]
        context=[]
        with open(f'{file_path}\{l1}\{str1}.txt', 'r', encoding='utf-16') as file:
            content1 = file.read().split('>')
            for c1 in content1:
                c2=re.sub(r'<|>|\d+|\n|I|-|।', '', c1)
                c2=re.sub(r'  ', '', c2)    
                start,end,c2=replace_word_lemmas(l1,c2)
                if start!=0 and start!=-1 and c2!='':context.append(c2+"*"+str(start)+"*"+str(end))
        sum_context+=len(context)
        lists_1[sense_cnt-1]=len(context)
        lists[sense_cnt-1]=context
        sense_cnt+=1
    
    for lists1 in lists:
        prod1=list(product(lists1,lists1))
        length = len(prod1)
        for p1 in prod1:
            if p1[0]!=p1[1]:
                p10=p1[0].replace(","," ").replace("  "," ").split("*")
                p11=p1[1].replace(","," ").replace("  "," ").split("*")
                if len(p1[0])<1000 and len(p1[1])<1000:
                    list3.append([l1,word_ind,replace_non_utf8(p10[0]),replace_non_utf8(p11[0]),p10[1],p10[2],p11[1],p11[2],1])
    for i in range(0,len(lists)):
        for j in range(0,len(lists)):
            if i!=j:
                prod1=list(product(lists[i],lists[j]))
                length = len(prod1)
                for p1 in prod1:
                    p10=p1[0].replace(","," ").replace("  "," ").split("*")
                    p11=p1[1].replace(","," ").replace("  "," ").split("*")
                    if len(p1[0])<1000 and len(p1[1])<1000:
                        list3.append([l1,word_ind,replace_non_utf8(p10[0]),replace_non_utf8(p11[0]),p10[1],p10[2],p11[1],p11[2],0])
    #break
    list4.append([l1,no_of_senses,sum_context,lists_1])
    
    
#Remove any bad lines from the dataframe
bad=[]
for i in range(0,len(list3)):
    l1=list3[i]
    if l1[2]==" " or l1[3]==" ":
        bad.append(i)

bad.sort(reverse=True)
for i in bad:
    del list3[i]


df=pd.DataFrame(list3,columns=["target_word","word_index","context_instance1","context_instance2","start1","end1","start2","end2","labels"])

df_train=df.iloc[0:100000]
df_val=df.iloc[100001:153508]

label_0_data = df_train[df_train['labels'] == 0].sample(n=3500)
label_1_data = df_train[df_train['labels'] == 1].sample(n=3500)
df1 = pd.concat([label_0_data, label_1_data])

# Drop the rows from df to create df3
df_test = df.drop(df1.index)

df3=df_test.sample(n=3000)
    
df2=df_val.sample(n=3000)
df2['labels'].value_counts(normalize = True)


df1.to_csv(r'C:\Users\fdair\Documents\College Work\MSc Project\483531Sense_Annotated_Hindi_Corpus\Sense Annotated Hindi Corpus\hindi-wsd_sample_v7_1000.csv',index = None,header=True)
df2.to_csv(r'C:\Users\fdair\Documents\College Work\MSc Project\483531Sense_Annotated_Hindi_Corpus\Sense Annotated Hindi Corpus\hindi-wsd_sample_val_v6_1000.csv',index = None,header=True)
df3.to_csv(r'C:\Users\fdair\Documents\College Work\MSc Project\483531Sense_Annotated_Hindi_Corpus\Sense Annotated Hindi Corpus\hindi-wsd_sample_test_v3_1000.csv',index = None,header=True)
