import timeit
xs=range(100)
print(timeit.timeit(stmt=f'map(hex, {xs})', number=10_000_000))
print(timeit.timeit(stmt=f'[hex(x) for x in {xs}]', number=10_000_000))

print(timeit.timeit(stmt=f'map(lambda x: x+10, {xs})', number=10_000_000))
print(timeit.timeit(stmt=f'[x+10 for x in {xs}]', number=10_000_000))


print(timeit.timeit(stmt=f'list(map(lambda x: x+10, {xs}))', number=10_000_000))
print(timeit.timeit(stmt=f'[x+10 for x in {xs}]', number=10_000_000))
