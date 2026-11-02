def is_sentence_palindromic(s):
    i=0
    j=len(s)-1
    print(s.lower())
    if not s:
        return False
    elif len(s) == 1:
        return True
    while i<=j:
        if s[i].lower() == s[j].lower():
            i+=1
            j-=1
        else:
            return False
    return True
        



#print(is_sentence_palindromic("hIih"))


# # S1
# def add_item(item, bucket=[]):
#     bucket.append(item)
#     return bucket
# print(add_item(1)); print(add_item(2))

# A1
# [1] [1,2]


# # S2
# funcs = [lambda: i for i in range(3)]
# print([f() for f in funcs])

# A2
# [0,1,2]


# # S3
# a = [[0] * 3] * 3
# a[0][0] = 1
# print(a)



# A3
# [[1,0,0],[0,0,0],[0,0,0]]


# # S4
# print(0.1 + 0.2 == 0.3)
#A4
# True

# # S5
# nums = [2, 4, 6, 8, 1]
# for n in nums:
#     if n % 2 == 0:
#         nums.remove(n)
# print(nums)
#  A5
# [4,8,1]


# # S6
# print(-7 // 2, -7 % 2, round(2.5), round(3.5), int(-2.7))
# A6 
# -3 1 2 3 -2

# # S7
# print(True + True, [] == False, bool([]) == False)
# A7
# True True True


# # S8
# x = [1, 2, 3]; y = x; y += [4]; z = x[:]; z.append(5)
# print(x, y, z)

# A8 
# [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]