'''
3.0
在2.0基础上，为解决重复子项带来的重复计算问题，引入动态表记录已有结果
'''

'''
2.0
木材长度-价格对应表price采用dict格式，当木材长度超过original_price中记录的已知长度价格范围时，规划算法中求木材不分割售卖收入会因为范围溢出而报错
此问题有两种解决方法：
                1、if n not in price判断
                2、将price改为defaultdict类型
方法2效率更高（运行速度快）已于采用
'''
# 定义初始化变量
original_price = [1,5,8,9,10,17,17,20,24,30,33]
# print(len(origianl_price))
# print(list(enumerate(origianl_price)))
from collections import defaultdict

price = defaultdict(int)
for i,p in enumerate(original_price): price[i+1] = p
# print(price)
# print(price[15])


# 定义动态表装饰器
def memo(func):
    cache = {}
    def _wrap(n): ## ? *args, **kwargs
        if n in cache: result = cache[n]
        else:
            result = func(n)
            cache[n] = result
        return result
    return _wrap


# 返回木材最高售价
@memo
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
    solution[n] = (split_point, n - split_point)
    return max_price

#
def parse_solution(target_length, revenue_solution):
    left, right = revenue_solution[target_length]
    if left==0: return [right]
    return parse_solution(left, revenue_solution) + parse_solution(right, revenue_solution)


# Main
solution = {}
n = 95
res = r(n)
par = parse_solution(n,solution)
print(res, par)