import re


# create dictionary that stores vocabulary
class Dictionary:
    """A class to represent a dictionary."""

    def __init__(self, dataset):
        """
        Instantiates the inverse dictionary.

        Parameters:
            dataset: ImageFolder that stores images from training folder
        """
        self.dataset = dataset
        self.inverse_dict = self.create_inverse()

    def create_inverse(self) -> dict:
        """Creates the inverse of the dictionary. Stores "int: class" pairs
        
        Returns: An inverse dictionary (int: class)

        Parameters
        ----------

        Returns
        -------

        """
        return dict((v, k) for k, v in self.dataset.class_to_idx.items())

    def simple_print(self, idx=50):
        """Shows first N integers and their respective classes.

        Parameters
        ----------
        idx :
            int (Default value = 50)

        Returns
        -------
        type
            

        """
        # show all classes from training folder
        print('\t\t\t\t\t\tClasses & Indexes')
        for i, v in enumerate(self.dataset.class_to_idx.values()):
            if i == idx: break
            print(f'{v}:\t{self.get_content(v)}')

    def get_item(self, item: int) -> str:
        """Returns a class name from a dictionary.

        Parameters
        ----------
        item :
            int
        item: int :
            

        Returns
        -------
        type
            

        """
        return self.inverse_dict[item]

    def get_content(self, index) -> str:
        """Gets the indices and outputs a beautiful representation of class names.

        Parameters
        ----------
        index :
            list or int

        Returns
        -------
        type
            

        """

        # remove leading digits and underscores
        make_prettier = lambda x: ' '.join(re.findall('[A-Za-z]+', self.get_item(x)))

        # check if it's a single index
        if type(index) == int:
            return make_prettier(index)

        # return comma-separated representation
        return ', '.join([make_prettier(x) for x in index])

