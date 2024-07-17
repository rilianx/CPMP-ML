from cpmp_ml.utils import generate_random_layout
from cpmp_ml.optimizer import OptimizerStrategy
from matplotlib import pyplot
import numpy as np

def percentage_per_container(optimizer: OptimizerStrategy, S: int, H: int, sample_size: int = 1000, **kwargs):
  x = [i for i in range(S*2, (S*(H-2))+1)] #Limites de S*2 hasta S*(H-2)
  y = [] #Resultados

  lays_N = [[generate_random_layout(S,H, n) for _ in range(sample_size)] for n in x]

  for n in x:
    costs = optimizer.solve(lays=np.array(lays_N[n-(S*2)]), **kwargs)
    valid_costs = [v for v in costs if v!=-1]
    results_model = len(valid_costs) / sample_size * 100.
    y.append(results_model)

  # Crear el gráfico de línea
  pyplot.plot(x, y, marker='o')

  # Agregar etiquetas y título
  pyplot.xlabel('N - Cantidad de contenedores')
  pyplot.ylabel('Porcentaje')
  pyplot.title('Porcentaje de acierto en relación N')

  # Mostrar el gráfico
  pyplot.grid(True)
  pyplot.show()