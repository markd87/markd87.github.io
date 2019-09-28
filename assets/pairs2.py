import numpy as np

structs=[]
pairs=[]


def words(l,r,ss):

  if (l == 0 and r == 0):
    structs.append(ss)
  if (l > 0):
    words(l-1, r+1 , ss+"1")
  if (r > 0):
    words(l, r-1, ss+"0")


def convert():

 for s in structs:
   news=[]
   for i,l in enumerate(s):
      if l=="0":
        continue
      count=1
      news.append(i+1)
      for j in range(i+1,len(s)):
       if s[j]=="0":
         count=count-1
         if count==0:
          news.append(j+1)
          break 
       else:
         count=count+1
   pairs.append(news)



##input
num=input("Number of points: ")
while (num%2 != 0):
 num=input("Number of points: ")
ss=""
words(num/2,0,ss)
convert()

print len(pairs),pairs

fname='structs'+str(num)+'.csv'
f=open(fname,'w')

f.write(str(len(pairs)))
f.write('\n')

for i,s in enumerate(pairs):
  for jj in s:
   f.write(str(jj))
   if (jj!=s[len(s)-1]):
     f.write(', ')
  f.write('\n')

