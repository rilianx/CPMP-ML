import random
import numpy as np
from copy import deepcopy
from statistics import mean

def load_model(model_file, S, H):
  import tensorflow as tf
  from tensorflow.keras import (layers, Input, Sequential, Model, optimizers)
  from tensorflow.keras.losses import BinaryCrossentropy
  device_name = tf.test.gpu_device_name()
  with tf.device(device_name):
    model=generate_model(S, H) # predice steps
    model.compile(
          loss=BinaryCrossentropy(),
          optimizer=optimizers.Adam(learning_rate=0.001),
          metrics=['mse']
    )
    try:
      model.load_weights(model_file)
    except:
      raise RuntimeError("Invalid model")
  return model

def validate_model(model_file, S, H, N, n = 1000, cvs_class=None):
  model = load_model(model_file, S, H)

  lays = []
  if cvs_class is None:
    for i in range(n):
      lays.append(generate_random_layout(S,H,N))
  else:
    n=40
    for i in range(1,n+1):
      lay = read_file(f"benchmarks/CVS/{cvs_class}/data{cvs_class}-{i}.dat",5)
      lays.append(lay)

  lays1 = deepcopy(lays)
  costs1 = greedy_model(model, lays1, max_steps=N*2)
  costs2 = greedys(lays)

  valid_costs1 = [v for v in costs1 if v!=-1]
  valid_costs2 = [v for v in costs2 if v!=-1]

  print("success ann model (%):", len(valid_costs1)/n*100., mean(valid_costs1))
  if len(valid_costs2)==0:
    print("success heuristic (%):", len(valid_costs2)/n*100.)
  else:
    print("success heuristic (%):", len(valid_costs2)/n*100., mean(valid_costs2))


def compute_sorted_elements(stack):
    if len(stack) == 0: return 0

    sorted_elements=1

    while(sorted_elements < len(stack) and 
            stack[sorted_elements] <= stack[sorted_elements-1]):
        sorted_elements +=1

    return sorted_elements   
    
class Layout:
    def __init__(self, stacks, H):
        self.stacks = stacks
        self.sorted_elements = []
        self.total_elements = 0
        self.sorted_stack = []
        self.unsorted_stacks = 0
        self.steps = 0
        self.H = H
        self.G=0
        self.reduced_stack = -1

        j=0
        
        for stack in stacks:
            if len(stack) > 0:
              g = max(stack)
              if self.G<g: self.G=g

            self.total_elements += len(stack)
            self.sorted_elements.append(compute_sorted_elements(stack))

            if not self.is_sorted_stack(j):

                self.unsorted_stacks += 1
                self.sorted_stack.append(False)

            else: self.sorted_stack.append(True)
            j += 1

    def permutate(self,perm):
      self.stacks=[self.stacks[i] for i in perm]
      self.sorted_elements=[self.sorted_elements[i] for i in perm]
      self.sorted_stack=[self.sorted_stack[i] for i in perm]

    
    def move(self,move):
        i = move[0]; j=move[1]
        
        if i==j: return None
        if len(self.stacks[i]) == 0: return None
        if len(self.stacks[j]) == self.H: return None

        if not self.is_BG_action(move): self.reduced_stack = i
        
        c = self.stacks[i][-1]

        if self.is_sorted_stack(i):
            self.sorted_elements[i] -= 1
   
        if self.is_sorted_stack(j) and self.gvalue(j) >= c:
            self.sorted_elements[j] += 1
            
        self.stacks[i].pop(-1)
        self.stacks[j].append(c)

        if len(self.stacks[i]) == 0: self.reduced_stack = -1
        
        self.is_sorted_stack(i)
        self.is_sorted_stack(j)
        self.steps += 1
        
        return c
                       
    def is_sorted_stack(self, j):
        sorted = len(self.stacks[j]) == self.sorted_elements[j]

        if (j < len(self.sorted_stack) and
                self.sorted_stack[j] != sorted): 

            self.sorted_stack[j] = sorted

            if sorted == True: self.unsorted_stacks -= 1

            else: self.unsorted_stacks += 1

        return sorted

    def gvalue(self, i):
        if len(self.stacks[i]) == 0: return self.G
        else: return self.stacks[i][-1]

def read_file(file, H):
    with open(file) as f:
        S, C = [int(x) for x in next(f).split()] # read first line
        stacks = []
        for line in f: # read rest of lines
            stack = [int(x) for x in line.split()[1::]]
            #if stack[0] == 0: stack.pop()
            stacks.append(stack)
            
        layout = Layout(stacks,H)
    return layout

def reachable_height(layout, i):
    if not layout.is_sorted_stack(i): return -1;
    
    top = layout.gvalue(i)
    h = len(layout.stacks[i])
    if h==layout.H: return h;
    
    stack=layout.stacks[i]
    all_stacks = True #True: all the bad located tops can be placed in stack
    
    for k in range(len(layout.stacks)):
        if k==i: continue
        if layout.is_sorted_stack(k): continue
            
        stack_k=layout.stacks[k]
        unsorted = len(stack_k)-layout.sorted_elements[k]
        prev = 1000;
        for j in range (1,unsorted+1):
            if stack_k[-j] <= prev and stack_k[-j] <=top:
                h += 1
                if h==layout.H: return h
                prev = stack_k[-j]
            else: 
                if j==1: all_stacks=False
                break
                
    if all_stacks: return layout.H
    else: return h

def generate_random_layout(S,H,N, feasible = False):
    stacks = []
    for i in range(S):
        stacks.append([])
    
    for j in range(N):
        s=random.randint(0,S-1);
        while len(stacks[s])==H: s=s=random.randint(0,S-1);
        g = random.randint(1,N)
        if feasible: g=N-j
        stacks[s].append(g);

    return Layout(stacks,H)

# GREEDY ##
def is_valid_BG_move(layout, s_o, s_d):
    if (s_o != s_d  and len(layout.stacks[s_o]) > 0
    and  len(layout.stacks[s_d]) < layout.H
    and layout.is_sorted_stack(s_o)==False
    and layout.is_sorted_stack(s_d)==True
    and layout.gvalue(s_o) <= layout.gvalue(s_d)):
      return True

    else: return False

def select_bg_move(layout):
  bg_move = None
  S=len(layout.stacks)
  min_diff = 100
  for s_o in range(S):
     for s_d in range(S):
       if is_valid_BG_move(layout, s_o, s_d):
          diff = layout.gvalue(s_d) - layout.gvalue(s_o)
          if min_diff > diff:
            min_diff = diff
            bg_move = (s_o,s_d)
  return bg_move

def greedy(layout, basic=True) -> int:
    steps = 0
    while layout.unsorted_stacks>0:
        bg_move=select_bg_move(layout)
        if bg_move is not None:
            layout.move(bg_move)
        else:
            return -1 # no lo resuelve
        steps +=1

    if layout.unsorted_stacks==0: 
        return steps
    return -1
###############





#Obtiene matriz a partir de Layout.
#- Los valores se normalizan y se elevan.
#- Los 2s quieren decir que no hay elementos.
#- El primer valor de cada pila indica si está ordenada o no.
#- Luego del estado, tenemos un arreglo indicando, 
#  para cada movimiento, si la pila de destino queda ordenada o no.
def get_ann_state(layout):
  S=len(layout.stacks)
  b = 2. * np.ones([S,layout.H+1])
  for i,j in enumerate(layout.stacks):
     b[i][layout.H-len(j)+1:] = [k/layout.total_elements for k in j]
     b[i][0] = layout.is_sorted_stack(i)

  mtype = []
  for i in range(5):
    for j in range(5):
      if i==j: continue
      m = layout.move((i,j))

      if m!=None:
        if layout.is_sorted_stack(j): mtype.append(1.)
        else: mtype.append(0.)
        m = layout.move((j,i)); layout.steps-=2
        if layout.is_sorted_stack(i): mtype.append(1.)
        else: mtype.append(0.)
      else:
        mtype.append(-1.); mtype.append(-1.)
  
  b.shape=(S*(layout.H+1),)
  b = np.concatenate((b,np.array(mtype)))

  return b

def get_layout_from_ann_state(b, S, H, N):
    # Reconstruir las pilas
    stacks = []
    for i in range(S):
        stack = b[i * (H + 1) : (i + 1) * (H + 1)]
        # Ignorar el primer elemento que es is_sorted_stack
        stack = stack[1:]
        # Recuperar los elementos de la pila
        stack_elements = [int(k * N) for k in stack if k != 2.0]
        stacks.append(stack_elements)
    return stacks

## INITIAL DATA GENERATION
# lay es un **estado resolubles óptimamete** en $N$ 
# pasos por un ***lazy greedy*** y. La función genera un 
# vector $A$ de salidas por movimiento $k$:

# Si el estado obtenido al aplicar el movimiento se puede resolver en $N-1$ pasos por el greedy $A_k=1$
# En cualquier otro caso: $A_k=0$

def costs_to_y(costs, parent_cost):
    #print (costs,parent_cost)
    mincost = np.inf
    y = []
    for c in costs:
        if c != -1 and c < mincost:
            mincost = c

    if c != -1 and mincost >= parent_cost:
        return None

    for c in costs:
        if c == mincost:
            y.append(1)
        else:
            y.append(0)
    #print (y); error()
    return y


def generate_y(layout, p_cost, v = False, basic = True):
  S=len(layout.stacks)
  A=np.zeros(S*(S-1))
  l = deepcopy(layout)
  n=0; costs = []
  for i in range(S):
    for j in range(S):
      if(i!=j):
        l.move((i,j))
        # print(f"Move {i} {j}")
        costs.append(greedy(l, basic=basic))
        l = deepcopy(layout)
        n+=1

  return costs_to_y(costs, p_cost)

def gen_movement_matrix(y, S):
    m = np.zeros(shape = (S, S));
    n=0
    for i in range(S):
        for j in range(S):
            if i == j: 
                continue
            m[i, j] = y[n]
            n+=1
    return m

def permutate_y(y, S, perm):
    m = gen_movement_matrix(y, S)
    # print(m)
    m = m[perm].T[perm].T
    # print(m)
    A=np.zeros(shape= (S*(S-1)))
    n=0
    for i in range(S):
        for j in range(S):
            if i == j: continue
            A[n] = m[i, j]
            n+=1
    return A

## GREEDY+MODEL
def get_move(act, S=5,H=5):
  k=0
  for i in range(S):
    for j in range(H):
      if(i==j): continue
      if k==act: return (i,j)
      k+=1


def random_perturbate_layout(lay, moves=5):
  S=len(lay.stacks)

  last_moves = []
  for m in range(moves):
    i=random.randint(0,S-1)
    j=random.randint(0,S-1)

    while (i,j) in last_moves or lay.move((i,j)) == None: 
      i=random.randint(0,S-1)
      j=random.randint(0,S-1)

    last_moves.append ((i,j))

def generate_data(
    S=5, H=5, N=10, 
    sample_size=1000, 
    lays=None, 
    perms_by_layout=5, 
    verbose=False,
    from_feasible=False, moves=5,
    basic=True
):
    x = []
    y = []
    n = 0
    while n < sample_size:
        layout = generate_random_layout(S, H, N, feasible=from_feasible)
        if from_feasible: random_perturbate_layout(layout, moves=moves)
        copy_lay = deepcopy(layout)
        p_cost = greedy(layout, basic=basic)
        if p_cost > -1:
            #for _ in range(perms_by_layout):
            #enum_stacks = list(range(S))
            #perm = random.sample(enum_stacks, S)
            #copy_lay.permutate(perm)
            y_ = generate_y(copy_lay, p_cost=p_cost, basic=basic)
            if y_ is None: continue

            for k in range(perms_by_layout):
                enum_stacks = list(range(S))
                perm = random.sample(enum_stacks, S)
                copy_lay.permutate(perm)
                y_ = permutate_y(y_, S, perm)

                x.append(get_ann_state(copy_lay))
                y.append(deepcopy(y_))
                if len(x) == sample_size:
                    return x, y
                n=n+1
                if n%5000==0: print(n)
                if n >= sample_size: break

                
    return x, y


# generate new data by using the model to solve layouts
def generate_data2(
    model,
    S=5,
    H=5,
    N=10,
    sample_size=1000,
    max_steps=20,
    batch_size=1000,
    perms_by_layout=20,
):
    x = []
    y = []

    while True:
        lays = []
        for i in range(batch_size):
            lays.append(generate_random_layout(S, H, N))
            # print ("Layout generado:", lays[i].stacks)

        lays0 = deepcopy(lays)
        costs = greedy_model(model, lays, max_steps=max_steps)

        # lays that cannot be solved by the model
        # lays0 = [lays0[i] for i in range(batch_size) if costs[i]==-1]
        # lays = [lays[i] for i in range(batch_size) if costs[i]==-1]
        # print("Costo obtenido por modelo:", len(lays))

        # for each lay we generate children clays
        clays = []
        for p in range(len(lays)):
            for i in range(S):
                for j in range(S):
                    if i == j:
                        continue
                    clay = deepcopy(lays0[p])
                    clay.move((i, j))
                    clays.append(clay)
            # print("len clays", len(clays))
        # print (f"clays generados {len(clays)}")
        # clays are solved

        ccosts = greedy_model(model, clays, max_steps=max_steps)
        # print("costs", ccosts)

        # f = lambda parent, k: (parent * (S*(S-1))) + k

        # Para cada padre
        for p in range(len(lays)):
            # print(lays[p].stacks)
            # print (ccosts[p*S*(S-1):(p+1)*S*(S-1)])
            A = []
            mincost = np.inf
            for c in range(p * (S * (S - 1)), (p + 1) * (S * (S - 1))):
                if ccosts[c] != -1 and ccosts[c] < mincost:
                    mincost = ccosts[c]

            if costs[p] != -1 and mincost >= costs[p]:
                continue

            for c in range(p * (S * (S - 1)), (p + 1) * (S * (S - 1))):
                if ccosts[c] != -1 and ccosts[c] == mincost:
                    A.append(1)
                else:
                    A.append(0)

            if (
                sum(A) > 0
            ):  # otherwise no action was succesful, we simply discard the data
                for k in range(perms_by_layout):
                    enum_stacks = list(range(S))
                    perm = random.sample(enum_stacks, S)
                    lays0[p].permutate(perm)
                    A = permutate_y(A, S, perm)

                    x.append(get_ann_state(lays0[p]))
                    y.append(deepcopy(A))
                    if len(x) == sample_size:
                        return x, y



## THE MODEL

# Useful for importing a pre-trained model
def create_model(S=5, H=5):
    import tensorflow as tf
    from tensorflow.keras import (layers, Input, Sequential, Model, optimizers)
    from tensorflow.keras.losses import BinaryCrossentropy

    # init tf
    device_name = tf.test.gpu_device_name()
    print("device_name", device_name)
    with tf.device(device_name):
      Fmodel=generate_model(S, H) # predice steps
      Fmodel.compile(
              loss=BinaryCrossentropy(),
              optimizer=optimizers.Adam(learning_rate=0.001),
              metrics=['mse']
        )
      return Fmodel

def generate_model(S=5, H=5):
   import tensorflow as tf
   from tensorflow.keras import layers, Input, Sequential, Model, optimizers
   model = tf.keras.Sequential()

   model.add(layers.Dense(256, activation='relu',
                          input_shape=(S*(H+1)+2*(S*(S-1)),)))

   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dense(64, activation='relu'))
   model.add(layers.Dense(S*(S-1), activation='sigmoid'))
   return model

def generate_model2(S=5, H=5):
  x = Input(shape=(S*(H+1)+2*(S*(S-1)),)) #recibe el estado + tipo de movs

  sensors = []
  for i in range(S): sensors.append(x[:,i*S:i*S+H+1])

  sensor_model2 = Sequential([
    #layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu')
  ])

  ## state encoding
  sensors_encodings=[]
  for i in range(S): 
    sensors_encodings.append(sensor_model2(sensors[i]))
  state_encoding = layers.Average()(sensors_encodings)

  sensor_model = Sequential([
    layers.Dense(256, activation='relu'),
    #layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])

  k=0
  pairwise_encodings=[]
  for i in range(S): 
    for j in range(S): 
      if i==j: continue
      pairwise_encodings.append(sensor_model(layers.Concatenate()([sensors[i],sensors[j],state_encoding,x[:,S*(H+1)+2*k:S*(H+1)+2*k+2]])))
      k+=1

  h = layers.Concatenate()(pairwise_encodings)
  #h = layers.Flatten()(h)

  model = Model(inputs=x, outputs=[h])

  return model



#return a vector with the number of steps that
#the model solved each of the layouts
#-1 means the model cannot solve the layout in less than 10 steps
def greedy_model(model, layouts, max_steps=10):
  costs = -np.ones(len(layouts))

  for steps in range(max_steps):
    x = []
    for i in range(len(layouts)):
      if layouts[i].unsorted_stacks==0: 
        if costs[i] ==-1: costs[i]=steps
        continue
      x.append(get_ann_state(layouts[i]))
    
    if len(x)==0: break
    actions = model.predict(np.array(x), verbose=False)
    k=0
    for i in range(len(layouts)):
      if costs[i] != -1: continue
      act = np.argmax(actions[k])
      move = get_move(act)
      layouts[i].move(move)
      k+=1
  return costs

def greedys(layouts):
  costs = -np.ones(len(layouts))
  for k in range(len(layouts)):
    steps = greedy(layouts[k])
    costs[k]=steps
  return costs


