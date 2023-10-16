import math

class ParallelProcessing:

    """
    Methods for parallel processing
    
    """

    def split_data_for_parallel_processing(self, data, numbers_of_datasets=10, subset_column='Protein.Ids',):
            
        # split data in 10 different subsets
        proteins = data[subset_column].unique()

        idx = 0
        subsets =[]
        steps = math.ceil(len(proteins)/numbers_of_datasets)

        for _ in range(numbers_of_datasets):
                if (idx + steps) < len(proteins):
                        subsets.append(proteins[idx:idx+steps])
                        idx += steps
                
                else:
                        subsets.append(proteins[idx:])
        
        datasets = [data[data['Protein.Ids'].isin(i)] for i in subsets]

        return datasets