import random

name = "webcam"
f = open(name+"_list.txt", "r")
a = f.read().split("\n")
f.close()

random.shuffle(a)
num = len(a)
f = open(name+"/train_5.txt", "w")
f.write("\n".join(a[:int(num/20)]))
f.close()
f = open(name+"/test_5.txt", "w")
f.write("\n".join(a[int(num/20):int(num/10)]))
f.close()

random.shuffle(a)
num = len(a)
f = open(name+"/train_10.txt", "w")
f.write("\n".join(a[:int(num/10)]))
f.close()
f = open(name+"/test_10.txt", "w")
f.write("\n".join(a[int(num/10):int(num/5)]))
f.close()


random.shuffle(a)
num = len(a)
f = open(name+"/train_20.txt", "w")
f.write("\n".join(a[:int(num/5)]))
f.close()
f = open(name+"/test_20.txt", "w")
f.write("\n".join(a[int(num/5):int(num/2.5)]))
f.close()