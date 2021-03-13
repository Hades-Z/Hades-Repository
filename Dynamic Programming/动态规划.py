'''
木材长度-价格对应表price采用dict格式，当木材长度超过original_price中记录的已知长度价格范围时，规划算法中求木材不分割售卖收入会因为范围溢出而报错
此问题有两种解决方法：
                1、if n not in price判断
                2、将price改为defaultdict类型
方法2效率更高（运行速度快）已于采用
'''
original_price = [1,5,8,9,10,17,17,20,24,30,33]
# print(len(origianl_price))
# print(list(enumerate(origianl_price)))
from collections import defaultdict

price = defaultdict(int)
for i,p in enumerate(original_price): price[i+1] = p
# print(price)
# print(price[15])


# 返回木材最高售价
def r(n):
    candidates = []
    # print(list(range(1,n)))

    # 木材分割售卖收入
    for i in range(1,n):
        # print(i, n-i)
        a = r(i)
        b = r(n-i)
        candidates.append( (a+b, i) )

    # 木材不分割售卖收入
    candidates.append((price[n], 0))
    # print(candidates)

    # 木材最高售卖收入，及分割情况
    max_price, split_point = max(candidates, key=lambda x:x[0])
    return max_price

res = r(5)
print(res)
