original_price = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 33]
# print(len(origianl_price))
# print(list(enumerate(origianl_price)))

price = {i + 1: p for i, p in enumerate(original_price)}


# print(price)
# print(price[10])


# 返回木材最高售价
def r(n):
    candidates = []
    # print(list(range(1,n)))

    # 木材分割售卖收入
    for i in range(1, n):
        # print(i, n-i)
        a = r(i)
        b = r(n - i)
        candidates.append((a + b, i))

        # 木材不分割售卖收入
    candidates.append((price[n], 0))
    # print(candidates)

    # 木材最高售卖收入，及分割情况
    max_price, split_point = max(candidates, key=lambda x: x[0])
    return max_price


res = r(5)
print(res)