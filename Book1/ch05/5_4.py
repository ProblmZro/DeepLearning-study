class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None
  def forword(self, x, y):
    self.x = x
    self.y = y
    return x * y
  def backword(self, dout):
    dx = dout * self.y
    dy = dout * self.x
    return dx, dy

class AddLayer:
  def __init__(self):
    pass
  def forword(self, x, y):
    return x + y
  def backword(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forword(apple, apple_num) # 1
orange_price = mul_orange_layer.forword(orange, orange_num) # 2
all_price = add_apple_orange_layer.forword(apple_price, orange_price) # 3
price = mul_tax_layer.forword(all_price, tax) # 4

# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backword(dprice) # 4
dapple_price, dorange_price = add_apple_orange_layer.backword(dall_price) # 3
dorange, dorange_num = mul_orange_layer.backword(dorange_price) # 2
dapple, dapple_num = mul_apple_layer.backword(dapple_price) # 1

print(print)
print(dapple_num, dapple, dorange, dorange_num, dtax)