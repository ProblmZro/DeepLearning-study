#step24 : 복잡한 함수의 미분
import numpy as np
from dezero import Variable

def sphere(x, y):
  z = x**2 + y**2
  return z

def matyas(x, y):
  z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
  return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
      (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
# z = sphere(x,y)
# z = matyas(x,y)
z = goldstein(x,y)
z.backward()
print(x.grad, y.grad)


'''
[칼럼] Define-and-Run vs Define-by-Run

1. Define-and-Run : 정적 계산 그래프
계산 그래프 정의 -> 컴파일 -> 데이터 흘려보내기

<예시>
# 계산 그래프 정의
a = Variable('a') 
b = Variable('b') 
c = a * b
d = c + Constant(1)

# 컴파일
f = compile(d)

# 데이터 흘려보내기
d = f(a=np.array(2), b=np.array(3))


2. Define-by-Run : 동적 계산 그래프
데이터 흘려보내기, 계산 그래프 구축 동시에 이뤄짐

<예시>
import numpy as np
from dezero import Variable

a = Variable(np.ones(10))
b = Variable(np.ones(10) * 2) 
c = b * a
d = c + 1
print(d)
'''