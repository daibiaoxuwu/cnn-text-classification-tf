# -*- coding: utf-8 -*-
 
from issue import Issue
import single_three
import tense
 
 
tensechecker=tense.Tense()
#djl
def checkTense(content):
    suggests=tense.tensecheck(content)
    issuesOfArticle = []
    for i in suggests:
        issue = Issue(2, 2, [i[0]], [i[1]], i[2], i[3])
        print(i)
        issuesOfArticle.append(issue)
    return issuesOfArticle

#gjy
def checkThirdPersonSingular(content):
    res = single_three.check(content)
    issues = []
    for sentence_entries in res:
        for entry in sentence_entries:
            le = entry[0]
            ri = entry[1]
            rep = entry[2]
            issue = Issue(2, 1, [le], [ri], rep, 4)
            issues.append(issue)
    return issues
 
 
def check(content):
    '检查内容中的语法错误'
 
    global tensechecker
    issues = []
    issues += checkThirdPersonSingular(content)
    issues += tensechecker.work(content)
    issues += tensechecker.work(content)
    # Issue(category, itype, start(list), end(list), replacement, exp_id), 参见 ../issue.py
 
    #issues = [issue]
    return issues # List of issues'''
 
 
'''def check(content):
    '样例：检查内容中的语法错误'
 
    # do something with the content, which is a string
 
    issue = Issue(1, 1, [15], [19], 'replacement', 3)
        # Issue(category, itype, start(list), end(list), replacement, exp_id), 参见 ../issue.py
 
    issues = [issue]
 
    return issues # List of issues'''
 
 
print(check("The fox jump walls and grew bigger. The rat was small but runs quickly."))
