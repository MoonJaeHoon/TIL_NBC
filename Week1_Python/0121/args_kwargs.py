# def index_dict(*args):
#     my_dict = {}
#     for ind,val in enumerate(args):
#         my_dict[ind] = val
#     return my_dict

# print(index_dict('어머니','아버지','누나','형','나'))


# from itertools import chain
# print(list(chain('ABC', 'DEF')))

# def how_old(**kwargs):
#     for k,v in kwargs.items():
#         if v<=27:
#             print(f'{k}는 {v}살입니다. 어리네요')
#         else:
#             print(f'{k}는 {v}살입니다. 먹을만큼 먹었네요')

# print(how_old(Jerry=30,Tom=35,SpongeBob=21,ZZANG9=7))


def index_dict(*args):
    my_dict = {}
    for ind,val in enumerate(args):
        my_dict[ind] = val
    return my_dict

my_dict1 = index_dict("A","B","C","D","E","F","G","H","I",)
# print(my_dict1)

my_dict2 = index_dict("AAA","BBB","CCC")
# print(my_dict2)

my_dict1.update(my_dict2)
print(my_dict1)