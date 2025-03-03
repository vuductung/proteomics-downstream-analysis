import matplotlib.pyplot as plt
import pandas as pd
from itertools import compress
from upsetplot import plot


class Intersection:
    def __init__(self, sets, set_names):
        """
        Initialize the Intersection class.
        
        Args:
            sets (list): List of sets containing the data
            set_names (list): List of names corresponding to each set
        """
        self.sets = sets
        self.set_names = set_names
        
        # Validate inputs
        if len(sets) != len(set_names):
            raise ValueError("Number of sets must match number of set names")
        
    def create_upset_data(self):
        """
        Create a DataFrame for the UpSet plot.
        
        Returns:
            pandas.DataFrame: DataFrame containing set membership information
        """
        # Get all unique elements from all sets
        all_elems = list(set().union(*self.sets))
        
        # Create membership matrix
        upset_data = pd.DataFrame(
            [[elem in st for st in self.sets] for elem in all_elems],
            columns=self.set_names
        )
        upset_data.index = all_elems
        return upset_data
    
    def upsetplot(self, figsize=(8, 4), savefig=False, title=None):
        """
        Create and display an UpSet plot.
        
        Args:
            figsize (tuple): Figure size (width, height)
            savefig (bool): Whether to save the figure to a file
        """
        # Create figure
        fig = plt.figure(figsize=figsize)

        if title:
            fig.suptitle(title)
        
        # Get and process data
        upset_data = self.create_upset_data()
        upset_data = upset_data.groupby(self.set_names).size()
        
        # Create plot
        plot(
            upset_data,
            orientation='horizontal',
            show_percentages=True,
            facecolor="lightblue",
            fig=fig,
            show_counts=True,
            element_size=None
        )
        
        # Save if requested
        if savefig:
            fig.savefig(savefig, bbox_inches='tight', transparent=True)
            
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