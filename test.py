
from tinyengine import Value
from model import *
n=MLP(3,[6,6,1])
xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
    
    ]
ys=[1.0,-1.0,-1.0,1.0]

for k in range(50):
    # forwar pass
    ypred = [n(x) for x in xs]
    loss=sum([(ypr[0]-yg)**2 for yg, ypr in zip(ys,ypred)])
    # backward pass
    for p in n.parameters():
        p.grad=0
    loss.backward()
    # updata weights
    for p in n.parameters():
        p.data+=-0.1*p.grad
        print(f"Number of epochs {k}, Loss {loss.data}" )
print(ypred)