import re


# create dictionary that stores vocabulary
class Dictionary:
    """
    A class to represent a dictionary.

        Attributes:
            dataset: ImageFolder that stores images from training folder

        Methods:
            create_inverse():   Creates the inverse of the dictionary. Stores "int: class" pairs
            simple_print(idx):  Shows first N integers and their respective classes
            get_item(item):     Returns the class name at specific index
            get_content(index): Returns string representation of a list of indices
    """

    def __init__(self, dataset):
        """
        Instantiates the inverse dictionary.

        Parameters:
            dataset: ImageFolder that stores images from training folder
        """
        self.dataset = dataset
        self.inverse_dict = self.create_inverse()

    def create_inverse(self) -> dict:
        """
        Creates the inverse of the dictionary. Stores "int: class" pairs

        Returns: An inverse dictionary (int: class)
        """
        return dict((v, k) for k, v in self.dataset.class_to_idx.items())

    def simple_print(self, idx=50):
        """
        Shows first N integers and their respective classes.

        Parameters:
            idx (int): Number of indices to show

        Returns: None
        """
        # show all classes from training folder
        print('\t\t\t\t\t\tClasses & Indexes')
        for i, v in enumerate(self.dataset.class_to_idx.values()):
            if i == idx: break
            print(f'{v}:\t{self.get_content(v)}')

    def get_item(self, item: int) -> str:
        """
        Returns a class name from a dictionary.

        Parameters:
            item (int): Index value to lookup

        Returns: Class name
        """
        return self.inverse_dict[item]

    def get_content(self, index) -> str:
        """
        Gets the indices and outputs a beautiful representation of class names.

        Parameters:
            index (list or int): List or index values to lookup

        Returns: string representation separated by commas
        """

        # remove leading digits and underscores
        make_prettier = lambda x: ' '.join(re.findall('[A-Za-z]+', self.get_item(x)))

        # check if it's a single index
        if type(index) == int:
            return make_prettier(index)

        # return comma-separated representation
        return ', '.join([make_prettier(x) for x in index])
