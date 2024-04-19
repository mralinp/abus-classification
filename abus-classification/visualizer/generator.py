def looper():
    print("Hey, how are you?")
    yield 1
    yield 2
    yield 3
    yield 4
    

x = looper()

print(x.__next__())
print(x.__next__())
print(x.__next__())
print(x.__next__())


