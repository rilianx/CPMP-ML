import random
import numpy as np
from copy import deepcopy

imported = False
def import_tf():
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Sequential, Model, optimizers
    imported = True
    return imported;

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
        
        c = self.stacks[i][-1]

        if self.is_sorted_stack(i):
            self.sorted_elements[i] -= 1
   
        if self.is_sorted_stack(j) and self.gvalue(j) >= c:
            self.sorted_elements[j] += 1
            
        self.stacks[i].pop(-1)
        self.stacks[j].append(c)
        
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

def greedy(layout) -> int:
    steps = 0
    while layout.unsorted_stacks>0:
        bg_move=select_bg_move(layout)
        if bg_move is not None:
            # print("bg_move:", bg_move)
            layout.move(bg_move)
        else:
            return -1 # no lo resuelve
            # print("no hay movimiento BG posibles")
            # print("elementos mal ubicados:",
                # layout.total_elements - sum(layout.sorted_elements))
        steps +=1

    if layout.unsorted_stacks==0: 
        # print("resuelto!")
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

## INITIAL DATA GENERATION
# lay es un **estado resolubles óptimamete** en $N$ 
# pasos por un ***lazy greedy*** y. La función genera un 
# vector $A$ de salidas por movimiento $k$:

# Si el estado obtenido al aplicar el movimiento se puede resolver en $N-1$ pasos por el greedy $A_k=1$
# En cualquier otro caso: $A_k=0$

def generate_y(layout, N=15, v = False):
  S=len(layout.stacks)
  A=np.zeros(S*(S-1))
  l = deepcopy(layout)
  n=0
  for i in range(S):
    for j in range(S):
      if(i!=j):
        l.move((i,j))
        # print(f"Move {i} {j}")
        steps=greedy(l)
        if(steps < 0): A[n]=0
        else:
            A[n] = (1 if steps < N else 0)
            # print("SIRVE:D")
        l = deepcopy(layout)
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

       

def generate_data(S=5, H=5, N=10, sample_size=1000, lays=None, perms_by_layout=5):
  x = []; y= [];
  n=0
  while(n<sample_size):
    layout = generate_random_layout(S,H,N)
    copy_lay = deepcopy(layout)
    val=greedy(layout)
    if(val>-1):
      for k in range(perms_by_layout):
        enum_stacks = list(range(S))
        perm = random_permutation = random.sample(enum_stacks, S)
        copy_lay.permutate(perm)

        y_ = generate_y(copy_lay,S,val)
        x.append(get_ann_state(copy_lay))
        y.append(y_)
        if lays is not None: lays.append(deepcopy(copy_lay))
        n=n+1
  return x, y


## THE MODEL
import_tf();

def generate_model(S=5, H=5):
   model = tf.keras.Sequential()
   model.add(layers.Dense(256, activation='relu', input_shape=(S*(H+1)+2*(S*(S-1)),)))
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


