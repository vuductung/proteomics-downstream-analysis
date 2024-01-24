import matplotlib.pyplot as plt
import pandas as pd
from itertools import compress
from upsetplot import plot

class Intersection():

    def __init__(self, sets, set_names):

        self.sets = sets
        self.set_names = set_names

    def upsetplot(self, figsize=(8, 4), savefig=False):
        
        fig = plt.figure(figsize=figsize)

        upset_data = self.create_upset_data()
        
        upset_data = upset_data.groupby(self.set_names).size()

        plot(upset_data, orientation='horizontal',show_percentages=True, facecolor="lightblue", fig=fig,
            show_counts=True, element_size=None)
        
        if savefig == True:
            
            fig.savefig('upset_plot.pdf', bbox_inches='tight', transparent=True)

    def create_upset_data(self):

        all_elems = list(set().union(*self.sets))

        upset_data = pd.DataFrame([[e in st for st in self.sets] for e in all_elems], columns = self.set_names)
        
        return upset_data

    def analyze_set_relationships(self):
        
        # create upset_data
        all_elems = list(set().union(*self.sets))
        upset_data = self.create_upset_data()
        upset_data['Genes'] = all_elems

        upset_data_series = [upset_data[i] for i in upset_data.select_dtypes('bool').columns]

        upset_data['bool'] = [[*i] for i in zip(*upset_data_series)]

        intersections = []

        for i in upset_data['bool']:
            intersections.append(tuple(compress(self.set_names, i)))

        upset_data['intersections'] = intersections

        upset_list = [upset_data[upset_data['intersections']==i]['Genes'].to_list() for i in upset_data['intersections'].unique()]

        intersec_data = pd.DataFrame(upset_list).T

        intersec_data.columns = [' & '.join(i) for i in upset_data['intersections'].unique()]

        return intersec_data