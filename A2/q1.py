# Accuracy in TRAIN  =  0.63086
# Accuracy in TEST  =  0.5996 
#Accuracy Majority Prediction = 0.4388
#Accuracy Random  = 0.2000
#Confusion Matrix [[14260, 2762, 1336, 1093,  3188], #
                  #[3941,  3362, 1751, 762,   350], 
                  #[1183,  3349, 5316, 2664,  669],
                  #[436,   1032, 5250, 17949, 15316],
                  #[349,   333,  878,  6890,  39299]]

# With Stemming
    # Accuracy in TRAIN  =  63.7// 0.64531
    # Accuracy in TEST  =   0.6068

import sys
import nltk
import json
import re
import math
from random import randint
import utils
from nltk.util import ngrams
from collections import Counter
import os
from os import path
import time
####////////// CHECK FIELD ///////////


def beautify(s):
    # only lower case
    s = s.lower()
    #unigrams
    s = s.strip()
    s1 = re.sub("[^a-zA-Z0-9]", " ", s)  # removing all accept letters and numbers
    return list((s1.split() ))

def clean(s):
    # only lower case
    s = s.lower()
    s = s.strip()
    s1 = re.sub("[^a-zA-Z0-9]", " ", s)
    return s1

    # # //////////////////// READ FILES ///////////////////////////




if (sys.argv[4]=='a'):
    data = []
    reviews = []
    ratings = []
    voacabulary = dict()
    reviews_star = [0 for x in range(5)]
    reviews_word_count = [0 for x in range(5)]
    MakeVocab = True

    if(path.exists("Vocabulary_a.json")):
        for line in open("Vocabulary_a.json", 'r'):
            voacabulary = json.loads(line)
            MakeVocab = False

        for st in xrange(5):
            reviews_word_count[st] =  voacabulary["starlen"+str(st)] 

        for st in xrange(5):
            reviews_star[st] = voacabulary["star"+str(st)]

    for line in open(sys.argv[2], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        reviews.append(data_point["text"])
        ratings.append(data_point["stars"])

    # close('train.json')
    print "reading done"
    if (MakeVocab):    
        for i in xrange(len(reviews)):
            l =  beautify(reviews[i])
            # l =  utils.getStemmedDocuments(reviews[i])

            star = int(float(ratings[i])) -1
            # print "Review ", i + 1 
            reviews_star[star]  += 1
            reviews_word_count[star] += len(l)
            if (MakeVocab):  
                for j in xrange(len(l)):
                    if l[j] in voacabulary:
                        voacabulary[l[j]][star] += 1
                    else:
                        d = [1 for x in range(5)]
                        d[star]  += 1;  
                        voacabulary[l[j]] = d 
            # print i
            else:
                # print "Vocab Present ",i
                pass



    # m = beautify("Shashwat is my name(Alongwith Banchhor shashwat).")
    # for i in xrange(len(m)):
    #     if m[i] in voacabulary:
    #         voacabulary[m[i]] += 1
    #     else:
    #         voacabulary[m[i]] = 1
    # print voacabulary
    # exit(0)
        v = len(voacabulary)
        reviews_word_count =  [reviews_word_count[i]+v for i in range(5)]

        for st in xrange(5):
            voacabulary["starlen"+str(st)] = reviews_word_count[st]

        for st in xrange(5):
            voacabulary["star"+str(st)] = reviews_star[st]



    if(not path.exists("Vocabulary_a.json")):
        utils.json_writer([voacabulary],"Vocabulary_a.json")




    # print len(reviews) , len(ratings)
    # for i in xrange(len(m)):
    #     print m[i]

    # print len(voacabulary)

    print "testing started"
    test_reviews = []
    test_ratings = []
    predition_ratings = []
    correct_count =  0


    print reviews_star

    p = 0 
    for x in xrange(len(ratings)):
        # cv = float(randint(1,5))
        # print cv , float(ratings)
        if (float(ratings[x])==float(randint(1,5))):
            p += 1

    # print "Random Acc" , float(p) / len(ratings)

    # exit(0)

    c=  0 
    for line in open(sys.argv[2], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        test_reviews.append(data_point["text"])
        actual_star = data_point["stars"]
        test_ratings.append(actual_star)
        # data_point = utils.getStemmedDocuments(data_point["text"])
        data_point = beautify(data_point["text"])
        prob_star = [0.0 for i in range(5)]
        prob_star = [math.log(reviews_star[star]/float(len(reviews))) for star in range(5)]
        for i in xrange(len(data_point)):
            for j in xrange(5):
                if data_point[i] in voacabulary:
                    prob_star[j] += math.log(voacabulary[data_point[i]][j]/float(reviews_word_count[j]))
                else:
                    prob_star[j] += math.log(1.0/float(reviews_word_count[j]))

        max = 0    
        for x in xrange(5):
            if (prob_star[max] <  prob_star[x]):
                max = x
        # print prob_star,max, actual_star
        # print c 
        c +=1
        predition_ratings.append(float(max)+1)
        if ((float(max)+1) == float(actual_star)):
            correct_count += 1
        

    print "train Accuracy : ",float(correct_count)/len(test_ratings)



    c=  0 
    test_reviews = []
    test_ratings = []
    predition_ratings = []
    correct_count =  0
    for line in open(sys.argv[3], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        test_reviews.append(data_point["text"])
        actual_star = data_point["stars"]
        test_ratings.append(actual_star)
        # data_point = utils.getStemmedDocuments(data_point["text"])
        data_point = beautify(data_point["text"])
        prob_star = [0.0 for i in range(5)]
        prob_star = [math.log(reviews_star[star]/float(len(reviews))) for star in range(5)]
        for i in xrange(len(data_point)):
            for j in xrange(5):
                if data_point[i] in voacabulary:
                    prob_star[j] += math.log(voacabulary[data_point[i]][j]/float(reviews_word_count[j]))
                else:
                    prob_star[j] += math.log(1.0/float(reviews_word_count[j]))

        max = 0    
        for x in xrange(5):
            if (prob_star[max] <  prob_star[x]):
                max = x
        # print prob_star,max, actual_star
        # print c 
        c +=1
        predition_ratings.append(float(max)+1)
        if ((float(max)+1) == float(actual_star)):
            correct_count += 1
        

    print "test Accuracy : ",float(correct_count)/len(test_ratings)


    conf_mat  = [[0 for x in range(5)] for x in range(5)]

    for i in xrange(len(test_ratings)):
        conf_mat[int(predition_ratings[i])-1][int(float(test_ratings[i]))-1] += 1

    print conf_mat

elif(sys.argv[4]=='b'):
    data = []
    reviews = []
    teratings = []
    voacabulary = dict()
    reviews_star = [0 for x in range(5)]
    reviews_word_count = [0 for x in range(5)]
    MakeVocab = True

    if(path.exists("Vocabulary_b.json")):
        for line in open("Vocabulary_b.json", 'r'):
            voacabulary = json.loads(line)
            MakeVocab = False

        for st in xrange(5):
            reviews_word_count[st] =  voacabulary["starlen"+str(st)] 

        for st in xrange(5):
            reviews_star[st] = voacabulary["star"+str(st)]
   

    for line in open(sys.argv[3], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        reviews.append(data_point["text"])
        teratings.append(data_point["stars"])

    # close('train.json')
    print "reading done"
   

    # print "testing started"
    test_reviews = []
    test_ratings = []
    predition_ratings = []
    correct_count =  0

    p = 0
    for x in xrange(len(teratings)):
        # cv = float(randint(1,5))
        # print cv , float(ratings)
        if (float(teratings[x])==float(randint(1,5))):
            p += 1

    print "Random Acc test" , float(p) / len(ratings)


    data = []
    reviews = []
    ratings = []
    voacabulary = dict()
    reviews_star = [0 for x in range(5)]
    reviews_word_count = [0 for x in range(5)]

    for line in open(sys.argv[2], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        reviews.append(data_point["text"])
        ratings.append(data_point["stars"])

    # close('train.json')
    # print "reading done"
    for i in xrange(len(reviews)):
        l =  beautify(reviews[i])
        # l =  utils.getStemmedDocuments(reviews[i])

        star = int(float(ratings[i])) -1
        # print "Review ", i + 1 
        reviews_star[star]  += 1
        reviews_word_count[star] += len(l)
        if (MakeVocab):    
            for j in xrange(len(l)):
                if l[j] in voacabulary:
                    voacabulary[l[j]][star] += 1
                else:
                    d = [1 for x in range(5)]
                    d[star]  += 1;  
                    voacabulary[l[j]] = d 
        else:
            print "Vocab Present ", i  
    max = 0
    for i in range(5):
        if (reviews_star[max] < reviews_star[i]):
            max = i
    max = max +1

    p = 0.0
    for x in xrange(len(teratings)):
        if (int(float(teratings[x]))==max):
            p +=1


    print "Majority Pred Acc " , float(p)/len(teratings)

elif(sys.argv[4]=='c'):
    data = []
    reviews = []
    ratings = []
    voacabulary = dict()
    reviews_star = [0 for x in range(5)]
    reviews_word_count = [0 for x in range(5)]
    MakeVocab = True

    if(path.exists("Vocabulary_c.json")):
        for line in open("Vocabulary_c.json", 'r'):
            voacabulary = json.loads(line)
            MakeVocab = False

        for st in xrange(5):
            reviews_word_count[st] =  voacabulary["starlen"+str(st)] 

        for st in xrange(5):
            reviews_star[st] = voacabulary["star"+str(st)]

    for line in open(sys.argv[2], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        reviews.append(data_point["text"])
        ratings.append(data_point["stars"])

    # close('train.json')
    # print "reading done"
    for i in xrange(len(reviews)):
        l =  beautify(reviews[i])
        # l =  utils.getStemmedDocuments(reviews[i])

        star = int(float(ratings[i])) -1
        # print "Review ", i + 1 
        reviews_star[star]  += 1
        reviews_word_count[star] += len(l)
        if (MakeVocab):    
            for j in xrange(len(l)):
                if l[j] in voacabulary:
                    voacabulary[l[j]][star] += 1
                else:
                    d = [1 for x in range(5)]
                    d[star]  += 1;  
                    voacabulary[l[j]] = d 
            # print i
        else:
            print "Vocab Present ", i


    # m = beautify("Shashwat is my name(Alongwith Banchhor shashwat).")
    # for i in xrange(len(m)):
    #     if m[i] in voacabulary:
    #         voacabulary[m[i]] += 1
    #     else:
    #         voacabulary[m[i]] = 1
    # print voacabulary
    # exit(0)
        v = len(voacabulary)
        reviews_word_count =  [reviews_word_count[i]+v for i in range(5)]

        for st in xrange(5):
            voacabulary["starlen"+str(st)] = reviews_word_count[st]

        for st in xrange(5):
            voacabulary["star"+str(st)] = reviews_star[st]


    if(not path.exists("Vocabulary_c.json")):
        utils.json_writer([voacabulary],"Vocabulary_c.json")



    # print len(reviews) , len(ratings)
    # for i in xrange(len(m)):
    #     print m[i]

    # print len(voacabulary)

    print "testing started"
    test_reviews = []
    test_ratings = []
    predition_ratings = []
    correct_count =  0


    # print reviews_star

    p = 0 
    for x in xrange(len(ratings)):
        # cv = float(randint(1,5))
        # print cv , float(ratings)
        if (float(ratings[x])==float(randint(1,5))):
            p += 1

    # print "Random Acc" , float(p) / len(ratings)

    # exit(0)

    # c=  0 
    # for line in open('train.json', 'r'):
    #     data.append(json.loads(line))
    #     data_point = json.loads(line)
    #     test_reviews.append(data_point["text"])
    #     actual_star = data_point["stars"]
    #     test_ratings.append(actual_star)
    #     # data_point = utils.getStemmedDocuments(data_point["text"])
    #     data_point = beautify(data_point["text"])
    #     prob_star = [0.0 for i in range(5)]
    #     prob_star = [math.log(reviews_star[star]/float(len(reviews))) for star in range(5)]
    #     for i in xrange(len(data_point)):
    #         for j in xrange(5):
    #             if data_point[i] in voacabulary:
    #                 prob_star[j] += math.log(voacabulary[data_point[i]][j]/float(reviews_word_count[j]))
    #             else:
    #                 prob_star[j] += math.log(1.0/float(reviews_word_count[j]))

    #     max = 0    
    #     for x in xrange(5):
    #         if (prob_star[max] <  prob_star[x]):
    #             max = x
    #     # print prob_star,max, actual_star
    #     # print c 
    #     c +=1
    #     predition_ratings.append(float(max)+1)
    #     if ((float(max)+1) == float(actual_star)):
    #         correct_count += 1
        

    # print "train Accuracy : ",float(correct_count)/len(test_ratings)



    c=  0 
    test_reviews = []
    test_ratings = []
    predition_ratings = []
    correct_count =  0
    for line in open(sys.argv[3], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        test_reviews.append(data_point["text"])
        actual_star = data_point["stars"]
        test_ratings.append(actual_star)
        # data_point = utils.getStemmedDocuments(data_point["text"])
        data_point = beautify(data_point["text"])
        prob_star = [0.0 for i in range(5)]
        prob_star = [math.log(reviews_star[star]/float(len(reviews))) for star in range(5)]
        for i in xrange(len(data_point)):
            for j in xrange(5):
                if data_point[i] in voacabulary:
                    prob_star[j] += math.log(voacabulary[data_point[i]][j]/float(reviews_word_count[j]))
                else:
                    prob_star[j] += math.log(1.0/float(reviews_word_count[j]))

        max = 0    
        for x in xrange(5):
            if (prob_star[max] <  prob_star[x]):
                max = x
        # print prob_star,max, actual_star
        # print c 
        c +=1
        predition_ratings.append(float(max)+1)
        if ((float(max)+1) == float(actual_star)):
            correct_count += 1
        

    # print "test Accuracy : ",float(correct_count)/len(test_ratings)


    conf_mat  = [[0 for x in range(5)] for x in range(5)]

    for i in xrange(len(test_ratings)):
        conf_mat[int(predition_ratings[i])-1][int(float(test_ratings[i]))-1] += 1

    print conf_mat

elif(sys.argv[4]=='d'):
    # print time.ctime()
    data = []
    reviews = []
    ratings = []
    voacabulary = dict()
    reviews_star = [0 for x in range(5)]
    reviews_word_count = [0 for x in range(5)]

    MakeVocab = True

    if(path.exists("Vocabulary_d.json")):
        for line in open("Vocabulary_d.json", 'r'):
            voacabulary = json.loads(line)
            MakeVocab = False

        for st in xrange(5):
            reviews_word_count[st] =  voacabulary["starlen"+str(st)] 

        for st in xrange(5):
            reviews_star[st] = voacabulary["star"+str(st)]


    for line in open(sys.argv[2], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        reviews.append(data_point["text"])
        ratings.append(data_point["stars"])

    # close('train.json')
    print "reading done"
    # print MakeVocab
    if (MakeVocab):
        for i in xrange(len(reviews)):
            # l =  beautify(reviews[i])
            l =  utils.getStemmedDocuments(reviews[i])

            star = int(float(ratings[i])) -1
            # print "Review ", i + 1 
            reviews_star[star]  += 1
            reviews_word_count[star] += len(l)
            
            if (MakeVocab):
                for j in xrange(len(l)):
                    if l[j] in voacabulary:
                        voacabulary[l[j]][star] += 1
                    else:
                        d = [1 for x in range(5)]
                        d[star]  += 1;  
                        voacabulary[l[j]] = d 
                # print i
                # if (i%10000==0):
                    # print "Vocab NotPresent ",i , " ",time.ctime()
            else :
                pass
                # /if (i%10000==0):
                    # print "Vocab Present ",i , " ",time.ctime()

        v = len(voacabulary)
        reviews_word_count =  [reviews_word_count[i]+v for i in range(5)]

        for st in xrange(5):
            voacabulary["starlen"+str(st)] = reviews_word_count[st]

        for st in xrange(5):
            voacabulary["star"+str(st)] = reviews_star[st]



    if(not path.exists("Vocabulary_d.json")):
        utils.json_writer([voacabulary],"Vocabulary_d.json")




    print "Vocab Length : " ,len(voacabulary)

    print "testing started"
    test_reviews = []
    test_ratings = []
    predition_ratings = []
    correct_count =  0





   


    c=  0 
    test_reviews = []
    test_ratings = []
    predition_ratings = []
    correct_count =  0
    for line in open(sys.argv[3], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        test_reviews.append(data_point["text"])
        actual_star = data_point["stars"]
        test_ratings.append(actual_star)
        data_point = utils.getStemmedDocuments(data_point["text"])
        # data_point = beautify(data_point["text"])
        prob_star = [0.0 for i in range(5)]
        prob_star = [math.log(reviews_star[star]/float(len(reviews))) for star in range(5)]
        for i in xrange(len(data_point)):
            for j in xrange(5):
                if data_point[i] in voacabulary:
                    prob_star[j] += math.log(voacabulary[data_point[i]][j]/float(reviews_word_count[j]))
                else:
                    prob_star[j] += math.log(1.0/float(reviews_word_count[j]))

        max = 0    
        for x in xrange(5):
            if (prob_star[max] <  prob_star[x]):
                max = x
        # print prob_star,max, actual_star
        print " test test",c 
        c +=1
        predition_ratings.append(float(max)+1)
        if ((float(max)+1) == float(actual_star)):
            correct_count += 1
        

    print "test Accuracy : ",float(correct_count)/len(test_ratings)

    c=  0 
    for line in open(sys.argv[2], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        test_reviews.append(data_point["text"])
        actual_star = data_point["stars"]
        test_ratings.append(actual_star)
        data_point = utils.getStemmedDocuments(data_point["text"])
        # data_point = beautify(data_point["text"])
        prob_star = [0.0 for i in range(5)]
        prob_star = [math.log(reviews_star[star]/float(len(reviews))) for star in range(5)]
        for i in xrange(len(data_point)):
            for j in xrange(5):
                if data_point[i] in voacabulary:
                    prob_star[j] += math.log(voacabulary[data_point[i]][j]/float(reviews_word_count[j]))
                else:
                    prob_star[j] += math.log(1.0/float(reviews_word_count[j]))

        max = 0    
        for x in xrange(5):
            if (prob_star[max] <  prob_star[x]):
                max = x
        # print prob_star,max, actual_star
        print "train test" , c 
        c +=1
        predition_ratings.append(float(max)+1)
        if ((float(max)+1) == float(actual_star)):
            correct_count += 1
        

    print "train Accuracy : ",float(correct_count)/len(test_ratings)



elif(sys.argv[4]=='e'):
    
    data = []
    reviews = []
    ratings = []
    voacabulary = dict()
    reviews_star = [0 for x in range(5)]
    reviews_word_count = [0 for x in range(5)]
    MakeVocab = True

    if(path.exists("Vocabulary_e.json")):
        for line in open("Vocabulary_e.json", 'r'):
            voacabulary = json.loads(line)
            MakeVocab = False


        for st in xrange(5):
            reviews_word_count[st] =  voacabulary["starlen"+str(st)] 

        for st in xrange(5):
            reviews_star[st] = voacabulary["star"+str(st)]
    
    c= 0

    for line in open(sys.argv[2], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        reviews.append(data_point["text"])
        ratings.append(data_point["stars"])
        # c += 1
        # if (c == 300000):
        #     break
    # close('train.json')
    print "reading done"
    
    if (MakeVocab):
        for i in xrange(len(reviews)):
            # l =  beautify(reviews[i])
            # print 'a'
            


            # Done Stemming and Cleaned the Data
            
            l =  utils.getStemmedDocuments(clean(reviews[i]))
            # Just clean Data
            # l =  clean(reviews[i])


            star = int(float(ratings[i])) -1
            # print "Review ", i + 1 
            reviews_star[star]  += 1
            reviews_word_count[star] += len(l)
            
            if(MakeVocab):
                for j in xrange(len(l)):
                    if l[j] in voacabulary:
                        voacabulary[l[j]][star] += 1
                    else:
                        d = [1 for x in range(5)]
                        d[star]  += 1;  
                        voacabulary[l[j]] = d 
                # print "Sample ", i

                # # Added bigrams
                # if (j< len(l)-1):
                #     print len(l) ," ", i," ",j
                #     if (len(l[j+1]+l[j]) >= 10):    
                #         if (l[j]+l[j+1]) in voacabulary:
                #             voacabulary[l[j]+l[j+1]][star] +=1
                #         else:
                #             d = [1 for x in range(5)]
                #             d[star]  += 1; 
                #             voacabulary[l[j]+l[j+1]] =d

               # # Added trigrams
               #  if (j<len(l)-2):
               #      if (l[j]+l[j+1] + l[j+2]) in voacabulary:
               #          voacabulary[l[j]+l[j+1]+l[j+2]][star] +=1
               #      else:
               #          d = [1 for x in range(5)]
               #          d[star]  += 1; 
               #          voacabulary[l[j]+l[j+1]+l[j+2]] = d

            else:
               pass # print "Vocab is present ", i
            

        v = len(voacabulary)
        reviews_word_count =  [reviews_word_count[i]+v for i in range(5)]

        for st in xrange(5):
            voacabulary["starlen"+str(st)] = reviews_word_count[st]

        for st in xrange(5):
            voacabulary["star"+str(st)] = reviews_star[st]
    

    v = len(voacabulary)
    reviews_word_count =  [reviews_word_count[i]+v for i in range(5)]

    # utils.json_writer([voacabulary],"Vocabulary.json")
    if(not path.exists("Vocabulary_e.json")):
        utils.json_writer([voacabulary],"Vocabulary_e.json")

    print "testing started"
    test_reviews = []
    test_ratings = []
    predition_ratings = []
    correct_count =  0





    # c=  0 
    # for line in open('train.json', 'r'):
    #     data.append(json.loads(line))
    #     data_point = json.loads(line)
    #     test_reviews.append(data_point["text"])
    #     actual_star = data_point["stars"]
    #     test_ratings.append(actual_star)
    #     data_point = utils.getStemmedDocuments(clean(data_point["text"]))
    #     # data_point = beautify(data_point["text"])
    #     prob_star = [0.0 for i in range(5)]
    #     prob_star = [math.log(reviews_star[star]/float(len(reviews))) for star in range(5)]
    #     for i in xrange(len(data_point)):
    #         for j in xrange(5):
    #             if data_point[i] in voacabulary:
    #                 prob_star[j] += math.log(voacabulary[data_point[i]][j]/float(reviews_word_count[j]))
    #             else:
    #                 prob_star[j] += math.log(1.0/float(reviews_word_count[j]))

    #     max = 0    
    #     for x in xrange(5):
    #         if (prob_star[max] <  prob_star[x]):
    #             max = x
    #     # print prob_star,max, actual_star
    #     print c 
    #     c +=1
    #     predition_ratings.append(float(max)+1)
    #     if ((float(max)+1) == float(actual_star)):
    #         correct_count += 1
        

    # print "train Accuracy : ",float(correct_count)/len(test_ratings)



    c=  0 
    test_reviews = []
    test_ratings = []
    predition_ratings = []
    correct_count =  0
    for line in open(sys.argv[3], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        test_reviews.append(data_point["text"])
        actual_star = data_point["stars"]
        test_ratings.append(actual_star)
        data_point = utils.getStemmedDocuments(clean(data_point["text"]))
        # data_point = beautify(data_point["text"])
        prob_star = [0.0 for i in range(5)]
        prob_star = [math.log(reviews_star[star]/float(len(reviews))) for star in range(5)]
        for i in xrange(len(data_point)):
            for j in xrange(5):
                if data_point[i] in voacabulary:
                    prob_star[j] += math.log(voacabulary[data_point[i]][j]/float(reviews_word_count[j]))
                else:
                    prob_star[j] += math.log(1.0/float(reviews_word_count[j]))

        max = 0    
        for x in xrange(5):
            if (prob_star[max] <  prob_star[x]):
                max = x
        # print prob_star,max, actual_star
        print "test_test ", c 
        c +=1
        predition_ratings.append(float(max)+1)
        if ((float(max)+1) == float(actual_star)):
            correct_count += 1
        

    print "test Accuracy : ",float(correct_count)/len(test_ratings)

    conf_mat  = [[0 for x in range(5)] for x in range(5)]

    for i in xrange(len(test_ratings)):
        conf_mat[int(predition_ratings[i])-1][int(float(test_ratings[i]))-1] += 1

    print conf_mat

    c=  0 
    for line in open(sys.argv[2], 'r'):
        data.append(json.loads(line))
        data_point = json.loads(line)
        test_reviews.append(data_point["text"])
        actual_star = data_point["stars"]
        test_ratings.append(actual_star)
        data_point = utils.getStemmedDocuments(clean(data_point["text"]))
        # data_point = beautify(data_point["text"])
        prob_star = [0.0 for i in range(5)]
        prob_star = [math.log(reviews_star[star]/float(len(reviews))) for star in range(5)]
        for i in xrange(len(data_point)):
            for j in xrange(5):
                if data_point[i] in voacabulary:
                    prob_star[j] += math.log(voacabulary[data_point[i]][j]/float(reviews_word_count[j]))
                else:
                    prob_star[j] += math.log(1.0/float(reviews_word_count[j]))

        max = 0    
        for x in xrange(5):
            if (prob_star[max] <  prob_star[x]):
                max = x
        # print prob_star,max, actual_star
        print c 
        c +=1
        predition_ratings.append(float(max)+1)
        if ((float(max)+1) == float(actual_star)):
            correct_count += 1
        

    print "train Accuracy : ",float(correct_count)/len(test_ratings)






# Checking Code
elif (sys.argv[4]=='t'):
   

    
    print path.exists("Vocabulary_d.json")

    d = dict()
    d["shashwat"] = 2
    utils.json_writer([d],"V.json")

    # p = dict()
    # p[1] = 3 

    for line in open("V.json", 'r'):
        p = json.loads(line)
    # p =  utils.json_reader("V.json")
    # print p["shashwat"] 
    exit(0)
    m = utils.getStemmedDocuments(clean("Shashwat is my Burger king name(Alongwith Banchhor shashwat)."))
    print m
    voacabulary = dict()
    for i in xrange(len(m)):
        if m[i] in voacabulary:
            voacabulary[m[i]] += 1
        else:
            voacabulary[m[i]] = 1
        if (i!= len(m)-1):
            if (m[i]+m[i+1]) in voacabulary:
                voacabulary[m[i]+m[i+1]] +=1
            else:
                voacabulary[m[i]+m[i+1]] =1
    print voacabulary

