from tinyengine import Value
import random
class Neuron:
    def __init__(self, nin):
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b=Value(random.uniform(-1,1))
        # print(self.w)
        # print(self.b)
    def __call__(self, x):
        out = sum((wi*xi for wi, xi in zip(self.w, x)),self.b) 
        act=out.tanh()
        # print(act)
        return act
    def parameters(self):
        return self.w+[self.b]
class Layers:
    def __init__(self,nin, nout):
        self.neuron= [Neuron(nout) for _ in range(nout)]
    def __call__(self, x):
        out=[neuron(x) for neuron in self.neuron]
        return out
    def parameters(self):
        return [p for n in self.neuron for p in n.parameters()]
    
class MLP:
    def __init__(self,nin,nout):
        size=[nin]+nout
        self.layers=[Layers(size[i], size[i+1] ) for i in range(len(nout))]
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]