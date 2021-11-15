
#SOURCE: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do

print("-------ITERABLES---------")
mylist = [1, 2, 3]
for i in mylist:
	print(i)

print("----------------")
mylist = [x*x for x in range(3)]		#STORED IN MEMORY
for i in mylist:
	print(i)
# #CAN CALL TWICE
# for i in mylist:
# 	print(i)

#GENERATORS = ITERATORS  YOU CAN ONLY ITERATE OVER ONCE. 
	# DO NOT STORE ALL THE VALUES IN MEMORY, THEY GENERATE THE VALUES ON THE FLY:
print("-------GENERATORS---------")
mygenerator = (x*x for x in range(3))
print(mygenerator)
for i in mygenerator:
	print(i)

#CANT CALL A SECOND TIME
print("----------------")
for i in mygenerator:
	print(i)

# It is just the same except you used () instead of []. BUT, you cannot perform 
#for i in mygenerator a second time since generators can only be used once: 
#they calculate 0, then forget about it and calculate 1, and end calculating 4, one by one.


print("--------YIELD--------")
# YIELD IS SIMILAR TO RETURN, EXCEPT THE FUNCTION WILL RETURN A GENERATOR.
# USE WHEN YOU KNOW YOUR FUNCTION WILL RETURN A HUGE SET OF VALUES THAT YOU WILL ONLY NEED TO READ ONCE.

def create_generator():
	mylist = range(4)
	for i in mylist: yield i*i

mygenerator = create_generator() # create a generator
print(mygenerator) # mygenerator is an object! generator object create_generator at 0xb7555c34>
for i in mygenerator:
	print(i)

print("----------------")

mygenerator = create_generator() # create a generator
print("A",next(mygenerator))

#CANT CALL A SECOND TIME
for i in mygenerator:
	print("B",i)



# To master yield, you must understand that when you call the function, the code you have written in 
#the function body does not run. The function only returns the generator object, this is a bit tricky.

# Then, your code will continue from where it left off each time for uses the generator.

# Now the hard part:

# The first time the for calls the generator object created from your function, it will run 
#the code in your function from the beginning until it hits yield, then it'll return the first 
#value of the loop. Then, each subsequent call will run another iteration of the loop you have 
#written in the function and return the next value. This will continue until the generator is
# considered empty, which happens when the function runs without hitting yield. That can be because 
#the loop has come to an end, or because you no longer satisfy an "if/else".
