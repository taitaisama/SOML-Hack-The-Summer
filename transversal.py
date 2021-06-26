from PIL import Image, ImageOps
import math
import csv

path = "SoML-50/SoML-50/data/"
#file = open("SoML-50/SoML-50/annotations.csv")
#s = (file.readline())
firstnums = [0]*996
secnums = [0]*996
opers = [0]*996
p = 0
for i in range(10):
    for j in range(10):
        flag = 0
        if j == 0 :
            flag = 1
        elif i == 0:
            flag = 0
        elif j > i:
            flag = 1
        elif i % j != 0 :
            flag = 1
        else:
            flag = 0
        if flag == 1 :
            for k in range(9):
                firstnums[p+k] = i
                secnums[p+k] = j
                opers[p+k] = math.floor(k/3)
            p += 9
        else:
            for k in range(12):
                firstnums[p+k] = i
                secnums[p+k] = j
                opers[p+k] = math.floor(k/3)
            p += 12

#print (p)
rows = []
for i in range(1, 50001):
    #newpath = path + str(i) + ".jpg"
    #im = Image.open(newpath)
    #s = file.readline() 1  
    num = (i-1) % 996
    oper = ""
    retult = 0
    if opers[num] == 0: # add
        oper = "+"
        result = firstnums[num] + secnums[num]
    elif opers[num] == 1: # subtract
        oper = "-"
        result = firstnums[num] - secnums[num]
    elif opers[num] == 2: # multiply
        oper = "*"
        result = firstnums[num] * secnums[num]
    else: # divide
        oper = "/"
        result = int(firstnums[num] / secnums[num])

    fix = (num % 3)
    if fix == 0:
        fields = [oper, str(firstnums[num]) , str(secnums[num])]
    elif fix == 1:
        fields = [str(firstnums[num]), oper, str(secnums[num])]
        
    fields = [str(fix) ,  oper , str(firstnums[num]) , str(secnums[num]) ,  str(result)]
    rows.append(fields)

    
    #distance = len(str(i)) + 6
    
    # if num % 3 == 0: #prefix
    #     distance += 6
    # elif num % 3 == 1:#postfix
    #     distance += 7
    # else: #infix
    #     distance += 5

    # retult = 0
    # if opers[num] == 0: # add
    #     result = firstnums[num] + secnums[num]
    # elif opers[num] == 1: # subtract
    #     result = firstnums[num] - secnums[num]
    # elif opers[num] == 2: # multiply
    #     result = firstnums[num] * secnums[num]
    # else: # divide
    #     result = int(firstnums[num] / secnums[num])

    # numstr = str(result)
    # actnum = s[distance:distance+len(numstr)]
    # if (numstr != actnum):
    #     print ("error ", i, ", ", result, ", ", actnum, ", ", firstnums[num], ", ", opers[num], ", ", secnums[num])

