#encoding:utf-8
#reader.py 只看含有一个动词的句子(十分之一左右)

import numpy as np
import word2vec
import re
import time
import os
import pickle
import random
import requests
import json
from queue import Queue
#bug:shorten和shorten_front不一样的话,每一遍都得重新计算而不是直接从队列里拿出来!


class reader(object):
    def printtag(self,number):
        #names=['现在式','现在式第三人称单数','现在式非第三人称单数','过去式','过去分词','现在分词']
        if number==len(self.verbtags):
            return 'NN'
        return self.verbtags[number]
        #return names[number]
    def parse(self,text):
        if(text==''):
            raise NameError
        url = 'http://166.111.139.15:9000'
        params = {'properties' : r"{'annotators': 'tokenize,ssplit,pos,lemma,parse', 'outputFormat': 'json'}"}
        while True:
            try:
                temptime=time.time()
                resp = requests.post(url, text, params=params).text
                content=json.loads(resp)
                self.numtopos=[]
                for i in content['sentences']:
                    temppos=[0]*len(i['tokens'])
                    for j in i['tokens']:
                        temppos[j['index']-1]=[j['characterOffsetBegin'],j['characterOffsetEnd']]
                    self.numtopos.append(temppos)


                #print( re.sub('\s+',' ',content['sentences'][:]['parse'].replace('\n',' ')))
                #return re.sub('\s+',' ',content['sentences'][:]['parse'].replace('\n',' '))
                return [i['parse'] for i in content['sentences']]
            except:# ConnectionRefusedError:
                print('Stnlp connection refused. Retrying...',time.time()-temptime)
                print(resp)
    def __init__(self,\
                content,\
                patchlength=3,\
                maxlength=700,\
                embedding_size=100,\
                num_verbs=1,\
                allinclude=True,\
                passnum=0):   #几句前文是否shorten #是否输出不带tag,只有单词的句子 

#patchlength:每次输入前文额外的句子的数量.
#maxlength:每句话的最大长度.(包括前文额外句子).超过该长度的句子会被丢弃.
#embedding_size:词向量维度数.
        self.url = 'http://166.111.139.15:9000'
        self.patchlength=patchlength
        self.maxlength=maxlength
        self.embedding_size=embedding_size
        self.num_verbs=num_verbs
        self.allinclude=allinclude
        self.passnum=passnum
        self.verbtags=['VB','VBZ','VBP','VBD','VBN','VBG'] #所有动词的tag

        
        dir0='papersmith/editor/grammar/tense/'
        dir0='tense/'
        #print('0')
        #self.model=[]
        #print('1')
        self.oldqueue=Queue()

        #parse
        self.resp=self.parse(content)
        self.readlength=len(self.resp)
        #print('rdlng',self.readlength)
        self.pointer=0
#        self.pointer=45521*50+4363449
        for _ in range(self.patchlength):
            self.oldqueue.put(self.resp[0])

#加载原型词典(把动词变为它的原型)
        with open(dir0+'ldict2', 'rb') as f:
            self.ldict = pickle.load(f)
        with open(dir0+'tagdict', 'rb') as f:
            self.tagdict = pickle.load(f)
        with open(dir0+'cldict', 'rb') as f:
            self.cldict = pickle.load(f)
        with open(dir0+'verbset', 'rb') as f:
            self.verbset = pickle.load(f)
        
    def isverb(self,verb):
        if verb not in self.verbset:
            if self.isverb2(verb)==True:
                print('not verb in verb2',verb)
            return False
        else:
            if self.isverb2(verb)==False:
                print('is verb not in verb2',verb)
            return True


    def isverb2(self,verb):
        if verb not in self.ldict: return False
        for i in self.verbtags:
            if (self.ldict[verb]+'('+i) not in self.cldict: return False
        return True

    def clean(self,sentence):
        initial=''
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in self.verbtags:
                    vbflag=1
                else:
                    vbflag=0
            else:
                node=re.match('([^\)]+)(\)*)',tag.strip())
                if node:
                    if vbflag==1:
                        initial+=' ('+node.group(1)+')'
                    elif self.isverb(node.group(1)):
                        initial+=' ['+node.group(1)+']'
                    else:
                        initial+=' '+node.group(1)
        return initial




    def lemma(self,verb):
        if verb in self.ldict:
            return self.ldict[verb]
        else:
            params = {'properties' : r"{'annotators': 'lemma', 'outputFormat': 'json'}"}
            resp = requests.post(self.url, verb, params=params).text
            content=json.loads(resp)
            word=content['sentences'][0]['tokens'][0]['lemma']
            print('errverb',verb)
            self.ldict[verb]=word
            return word

    def list_tags(self,batch_size):
        while True:#防止读到末尾
            inputs=[]
            pads=[]
            poses=[]
            words=[]
            answers=[]
            count=0
            while len(inputs)<batch_size:
                if self.pointer==self.readlength:
                    self.pointer=0
                    return None,None,None,None,None,None
                sentence=self.resp[self.pointer]
                self.pointer+=1

                outword=[]
                answer=[]
                word=[]
                pose=[]
                total=0
#筛选只有一个动词的句子                
                for tag in sentence.split():
                    if tag[0]!='(':
                        node=re.match('([^\)]+)(\)*)',tag.strip())
                        if node:
                            if self.isverb(node.group(1)):
                                total+=1
                if total==0:
                    self.oldqueue.put(sentence)
                    self.oldqueue.get()
                    continue
#前文句子
                newqueue=Queue()
                for _ in range(self.patchlength):
                    oldsentence=self.oldqueue.get()
                    newqueue.put(oldsentence)
                    for tag in oldsentence.split():
                        if tag[0]=='(':
                            if tag not in self.tagdict:
                                self.tagdict[tag]=len(self.tagdict)
                                print('tagdicterror:',len(self.tagdict),tag)
                            tagword=[0]*self.embedding_size
                            tagword[self.tagdict[tag]]=1
                            outword.append(tagword)
                        else:                
                            node=re.match('([^\)]+)(\)*)',tag.strip())
                            if node:
                                #group(1) 单词
                                verb=node.group(1)
                                #group(2) 括号
                                tagword=[0]*self.embedding_size
                                tagword[0]=1
                                for _ in range(len(node.group(2))-1):
                                    outword.append(tagword)
                self.oldqueue=newqueue
                self.oldqueue.put(sentence)
                self.oldqueue.get()
                #print('point at:',self.resp.tell())

#本句                
                tagcount=-1
                for tag in sentence.split():
                    if tag[0]=='(':
#去除情态动词
                        if tag=='(MD':
                            mdflag=1
                        else:
                            mdflag=0
                            if tag[1:] in self.verbtags:
                                answer.append(self.verbtags.index(tag[1:]))
                                tag='(VB'
                                vbflag=1
                            else:
                                vbflag=0
                            if tag not in self.tagdict:
                                self.tagdict[tag]=len(self.tagdict)
                                print('tagdicterror:',len(self.tagdict),tag)
                            tagword=[0]*self.embedding_size
                            tagword[self.tagdict[tag]]=1
                            outword.append(tagword)
                    else:
                        tagcount+=1
                        if mdflag==0:
                            node=re.match('([^\)]+)(\)*)',tag.strip())
                            if node:
                                verb=node.group(1)
                                if True:
                                    if vbflag==1 or self.isverb(verb):
                                        if vbflag==0:
                                            answer.append(len(self.verbtags))

                                        pose.append(self.numtopos[self.pointer-1][tagcount])
                                        word.append(verb)
#去除时态
                                        node2=self.lemma(verb)
                                else:
                                    outword.append([0]*self.embedding_size)
                                tagword=[0]*self.embedding_size
                                tagword[0]=1
                                for _ in range(len(node.group(2))-1):
                                    outword.append(tagword)
                outword=np.array(outword)
#句子过长
                if outword.shape[0]>self.maxlength:
                    print('passed too long sentence')
                    continue
#补零
                pads.append(outword.shape[0])
                outword=np.pad(outword,((0,self.maxlength-outword.shape[0]),(0,0)),'constant')
                inputs.append(outword)
                answers.append(answer)
                poses.append(pose)
                words.append(word)
                print(self.clean(sentence),[self.printtag(x) for x in answer])

#            inputs=np.array(inputs)
#构建输出
#用完整个输入,从头开始
#continue the 'while True' loop
            return inputs,pads,poses,words,total,answers

if __name__ == '__main__':
    with open('tense/combine.txt') as f:
        for i in range(10):
            model = reader(f.readline())
            model.list_tags(1)

