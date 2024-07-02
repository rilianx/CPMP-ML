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
            self.sorted_elements.append(self.compute_sorted_elements(stack))

            if not self.is_sorted_stack(j):

                self.unsorted_stacks += 1
                self.sorted_stack.append(False)

            else: self.sorted_stack.append(True)
            j += 1

    def compute_sorted_elements(self, stack):
        if len(stack) == 0: return 0

        sorted_elements=1

        while(sorted_elements < len(stack) and 
                stack[sorted_elements] <= stack[sorted_elements-1]):
            sorted_elements +=1

        return sorted_elements 

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

    def get_actions(self):
        actions =[]
        for i in range(len(self.stacks)):
            for j in range(len(self.stacks)):
                if i!=j and len(self.stacks[i]) > 0 and len(self.stacks[j]) < self.H:
                        actions.append((i,j))
        return actions

    def is_BG_action(self, action):
        s_o = action[0]; s_d = action[1]
        if (self.is_sorted_stack(s_o)==False
        and self.is_sorted_stack(s_d)==True
        and self.gvalue(s_o) <= self.gvalue(s_d)):
          return True

        else: return False