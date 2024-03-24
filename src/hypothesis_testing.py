import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison

def hypothesis_testing(scores_models, cv=5, alpha=0.01):
    """
    Realiza pruebas estadísticas para comparar las puntuaciones de precisión de varios modelos.
    
    Parámetros:
        scores_models: lista de tuplas, donde cada tupla contiene el nombre del modelo y sus puntuaciones de precisión.
        cv: número de divisiones de validación cruzada.
        alpha: nivel de significancia para las pruebas estadísticas.
    """
    # Preparar datos para pruebas estadísticas
    scores_data = [model_scores[1] for model_scores in scores_models]  # Directamente la lista de valores
    model_names = [model_scores[0] for model_scores in scores_models]
    
    # Prueba de Kruskal-Wallis
    F_statistic, pVal = stats.kruskal(*scores_data)
    
    # ANOVA de un factor
    F_statistic2, pVal2 = stats.f_oneway(*scores_data)
    
    print('p-valor Kruskal-Wallis:', pVal)
    print('p-valor ANOVA:', pVal2)
    
    if pVal <= alpha:
        print('Rechazamos la hipótesis nula: los modelos son estadísticamente diferentes\n')
        
        # Preparación de los datos para comparaciones múltiples
        stacked_data = np.hstack(scores_data)
        stacked_model = np.hstack([np.repeat(model_name, cv) for model_name in model_names])
        
        # Comparaciones múltiples
        MultiComp = MultiComparison(stacked_data, stacked_model)
        comp = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
        print(comp[0])
        
        # Prueba de Tukey HSD
        print(MultiComp.tukeyhsd(alpha=0.05))
    else:
        print('Aceptamos la hipótesis nula: no hay diferencias estadísticas significativas entre los modelos')

    
    # guardar en csv resultados de las pruebas estadísticas
    results = []
    for i in range(len(scores_data)):
        results.append([model_names[i], np.mean(scores_data[i]), np.std(scores_data[i])])
    results = np.array(results)

    np.savetxt('results/model_results/hypothesis_testing.csv', results, mode='a', delimiter=',', fmt='%s')



    
        
