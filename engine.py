import numpy as np

try:
    import cupy as _cp
    xp = _cp
    ON_GPU = True
except Exception:
    xp = np
    ON_GPU = False

DTYPE = xp.float32
def sync():
    if ON_GPU:
        _cp.cuda.Stream.null.synchronize()

class Tensor:
    def __init__(self, data, _children=(),_op=''):
        self.data = xp.asarray(data, dtype = DTYPE)
        self.grad = xp.zeros_like(self.data)
        #gradient
        self._backward = lambda:None
        #backprop function
        self.children = set(_children)
        self._op = _op
    @property
    def shape(self):
        return self.data.shape
    def numpy(self):
        return _cp.asnumpy(self.data) if ON_GPU else self.data
    @staticmethod
    def _unbroadcast(g, shape):
        while g.ndim > len(shape):
            g = g.sum(axis = 0)
        for i , s in enumerate(shape):
            if s==1 and g.shape[i] !=1:
                g = g.sum(axis=i, keepdims=True)
        return g
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
    
        def _backward():
            self.grad  += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
    
        def _backward():
            self.grad  += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(self.data  * out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    def __radd__ (self, other): 
        return self + other
    def __rmul__(self, other):
        return self * other
    def __pow__(self, other):
        other = other if isinstance (other, (int, float)) else float (other)
        out = Tensor(self.data**other,(self,), '**')
        def _backward():
            self.grad += other*(self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)
    def __truediv__(self, other):
        return self * (other**-1)
    def __rtruediv__(self, other):
        return other *(self**-1)
    
        
    def tanh(self):
            t = xp.tanh(self.data)
            out = Tensor(t, (self,), 'tanh')
    
            def _backward():
                self.grad += (1 - t * t) * out.grad
            out._backward = _backward
            return out
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad  += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')
    
        def _backward():
            self.grad += xp.ones_like(self.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = xp.ones_like(self.data) 
        for v in reversed(topo):
            v._backward()
    def __repr__(self):
            return f'{self.data}'
    
class Layer:
    def __init__(self, nin, nout, act=True):
        self.w = Tensor(xp.random.uniform(-1, 1, (nin, nout)) * nin ** -0.5)
        self.b = Tensor(xp.zeros(nout))
        self.act = act

    def __call__(self, x):
        out = x @ self.w + self.b
        return out.tanh() if self.act else out

    def parameters(self):
        return [self.w, self.b]


class MLP:
    def __init__(self, nin, nouts, afunc = True):
        sz = [nin] + list(nouts)
        last = len(nouts) - 1
        self.layers = [Layer(sz[i], sz[i + 1], act=(i != last or afunc)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


def bench(X=  xp.random.uniform(0, 1, (1024, 784)), T = xp.random.uniform(-1, 1, (1024, 10)), bnum = 32, nin=784, hidden = [256], nout=10, epoch = 20, seed = 0, actout = False):
    import time
    size = X.shape[0]
    X = X.reshape(size, -1) 
    assert X.shape[0] == T.shape[0]
    steps = size//bnum
    sz = hidden + [nout]
    xp.random.seed(seed)
    
    m = MLP(nin, sz, afunc = actout)
    P = m.parameters()
    #warmup for the engine on the GPU
    for _ in range(3):             
        loss = ((m(Tensor(X[:bnum])) - Tensor(T[:bnum])) ** 2).sum()
        for p in P:
            p.grad = xp.zeros_like(p.data)
        loss.backward()
    sync()
    t0 = time.perf_counter()
    for ep in range (epoch):
        t = time.perf_counter()
        perm = np.random.permutation(size)
        print(f"epoch{ep+1}")
        for i in range(0,size,bnum):
            itx = perm[i:i+bnum]
            Xb = Tensor(X[itx])
            Tb = Tensor(T[itx])
            loss = ((m(Xb) - Tb) ** 2).sum()
            print(f"batch{i+1},loss:{loss}")
            for p in P:
                p.grad = xp.zeros_like(p.data)
            loss.backward()
            for p in P:
                p.data -= 0.01 * p.grad / len(itx)
        sync()
        print(f"epoch {ep+1} complete, time: {time.perf_counter() - t}")
    dt = time.perf_counter() - t0

    where = 'GPU (cupy)' if ON_GPU else 'CPU (numpy)'
    print(f'{where:12}  {bnum}x{nin} -> {hidden} -> {nout}, dtype {xp.dtype(DTYPE).name}')
    print(f'      total steps{steps*epoch} over {epoch}epochs and {steps}step each')
    print(f'              {steps*epoch} steps in {dt:.3f}s  =  {dt/(steps*epoch)*1000:.2f} ms/step')
    print(f'              {bnum*steps*epoch/dt:,.0f} images/sec')
    return m


if __name__ == '__main__':
    print('cupy available:', ON_GPU)
    if not ON_GPU:
        print('  -> running on numpy. pip install cupy-cuda12x for the GPU path.')
    bench()
