import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

import scipy

from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
# import demoji
# from pyvi import ViPosTagger, ViTokenizer
import string

import streamlit as st
import pickle
import speech_recognition as sr
from gtts import gTTS
import playsound
from datetime import date,datetime
import datetime
from time import strftime
import os
import warnings
warnings.filterwarnings("ignore")

#@st.cache_data
def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’", '')
    document = regex.sub(r'\.+', ".", document)
    # Remove punctuation
    document = regex.sub('[^\w\s]', ' ', document)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        document = document.replace(char, ' ')

    # Remove numbers, only keep letters
    document = regex.sub(r'[\w]*\d+[\w]*', "", document) # document.replace('[\w]*\d+[\w]*', '', regex=True)

    # Some lines start with a space, remove them
    document = regex.sub('^[\s]{1,}', '', document)

    # # Remove multiple spaces with one space
    document = regex.sub('[\s]{2,}', ' ', document)

    # Some lines end with a space, remove them
    document = regex.sub('[\s]{1,}$', '', document)

    # Remove end of line characters
    document = regex.sub(r'[\r\n]+', ' ', document)

    # Remove HTTP links
    document = regex.sub(
        r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '',
        document)

    new_sentence = ''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word] + ' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence + sentence + '. '
    document = new_sentence
    # print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    # ...
    return document

# Chuẩn hóa unicode tiếng việt
#@st.cache_data
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

#@st.cache_data
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

#@st.cache_data
def process_special_word(text):
    new_text = ''
    list_word=["không",'hông', "chẳng", "chả",'tránh','mà','khiến','chớ','lại','dẫu','nhưng','dù','tuy','bị','còn','thiếu']
    text_lst = text.split()
    i= 0
    if any(word in text_lst for word in list_word):
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if word in list_word:
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

#@st.cache_data
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

#@st.cache_data
def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document


#@st.cache_data
def format_bullet(list):
    st.markdown("""
        <style>
            ul.bullet {
                font-family: 'Tahoma', sans-serif;
                list-style-type: none;
                padding-left: 0;
            }
            ul.bullet li:before {
                content: '\\1F539';
                padding-right: 5px;
            }
        </style>""", unsafe_allow_html=True)
       
       # Sử dụng phương thức markdown để tạo danh sách
    st.markdown("""
            <ul class="bullet" style="text-align: justify;">
                """ + "\n".join([f"<li>{item}</li>" for item in list]) + """
            </ul>""", unsafe_allow_html=True)
    st.write('\n\n')


#@st.cache_data
def read_data():
    data=pd.read_csv('Products_Shopee_comments.csv')
    data.loc[data['rating']<=2,'class']=0
    data.loc[data['rating']==3,'class']=1
    data.loc[data['rating']>=4,'class']=2
    df=data.loc[(data['class'] <1) | (data['class'] >1)]
    df["class"] = df["class"].apply(lambda x: 1 if x > 1 else x)
    return df

df=read_data()
#@st.cache_data
def review_data(data):
    st.dataframe(data.head())
    st.write('Số dòng dữ liệu: ',len(data))
    data=data.dropna()
    data=data.drop_duplicates()
    st.write('Số dòng dữ liệu sau khi xoá dòng trống và bị trùng: ',len(data))
    data=data.dropna()
    data=data.drop_duplicates()
    st.write('Số dòng dữ liệu thuộc lớp tiêu cực: ',len(data.loc[(data['class'] ==0)]))
    st.write('Số dòng dữ liệu thuộc lớp tích cực: ',len(data.loc[(data['class'] ==1)]))
    st.write(data.groupby('class').count())

#@st.cache_data
def hear():
    import speech_recognition as sr
    ear = sr.Recognizer()
    for i in range(1):
        with sr.Microphone() as sourse:
            st.write("Đang lắng nghe...")
            audio = ear.listen(sourse,phrase_time_limit=2)
                
        try:                   
            text = ear.recognize_google(audio,language="vi-VI")               
            if text!="":
                return text.lower()
            else:
                st.write("Không nghe rõ")
                return ""
        except:
            st.write("Không nghe rõ")
            return ""
        #time.sleep(2)